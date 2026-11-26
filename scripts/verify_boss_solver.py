#!/usr/bin/env python3
"""Small-scale same-initial-state Boss mistake finder & clean-engine verifier.

Scientific protocol (zero RNG ambiguity, zero false "unwinnable" claims):
  For each Boss snapshot item (mined from independent non-eval runs):
    1. Construct one fixed combat spec with deterministic combat_seed = f"SOLV_{id}".
    2. Run the frozen policy greedily on a clean Sts2Headless combat reset.
       - If it wins -> label "already_won_by_policy" (not a mistake).
    3. Otherwise (policy lost), run budgeted neural-guided search from the EXACT same
       initial state using Sts2Headless native snapshot(0) / restore(0):
         Phase A: Temperature-sampled policy rollouts (fast tactical exploration).
         Phase B: High-entropy branching search along the best/closest rollout trajectory
                  (snapshots high-uncertainty decisions and explores alternative top-k actions).
       - If no winning trajectory is found within budget/timeout -> label "unknown"
         (never falsely called "provably unwinnable").
    4. If search finds a winning action sequence, replay those exact actions from a
       FRESH clean reset_combat(spec) (no snapshot/restore) to confirm victory == True.
       - Only if (policy_lost AND search_won AND clean_replay_won) -> label "verified_solvable_mistake".

Outputs yield metrics (verified_mistakes_per_hour, rescue_rate) to decide whether
scaling up solver-filtered SolverPool practice is worthwhile.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AD_PY = ROOT / "external" / "AlphaDefect-Public" / "game-v0.114.0" / "python"
if str(AD_PY) not in sys.path:
    sys.path.insert(0, str(AD_PY))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_v1140 import load_net_weights  # noqa: E402
from sts2env.env import EngineError, Sts2Env  # noqa: E402
from sts2rl.actor import MODE_COMBAT  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402


@torch.no_grad()
def policy_eval_obs(net, device: str, obs) -> tuple[np.ndarray, float]:
    """Return legal action probabilities (float64) and scalar value for a single combat Obs."""
    rg = Ragged.from_obs([obs], np.array([MODE_COMBAT], dtype=np.int64))
    batch = collate_ragged(rg, attend=bool(net.cfg.macro_attend)).to(device)
    logits, value = net.forward_infer(batch)
    lg = logits[0, : obs.num_actions].float()
    probs = torch.softmax(lg, dim=-1).cpu().numpy().astype(np.float64)
    return probs, float(value[0].item())


def make_combat_spec(item: dict) -> dict:
    return {
        "character": item.get("character", "NECROBINDER"),
        "ascension": int(item.get("ascension", 10)),
        "seed": f"SOLV_{item['id']}",
        "act": int(item["act"]),
        "encounter": item["encounter"],
        "room": "Boss",
        "player_snapshot": item["snap"],
    }


def _run_episode(
    env: Sts2Env,
    obs,
    net,
    device: str,
    rng: np.random.Generator | None,
    temp: float,
    max_steps: int,
    first_action: int | None = None,
    record_entropy: bool = False,
) -> dict:
    """Roll out a combat from current obs until done/stuck/max_steps."""
    actions: list[int] = []
    entropies: list[tuple[int, float, list[int]]] = []
    step_i = 0
    while not obs.done and not obs.stuck and obs.num_actions > 0 and step_i < max_steps:
        if first_action is not None and step_i == 0:
            a = int(first_action)
        elif obs.num_actions == 1:
            a = 0
        else:
            probs, _ = policy_eval_obs(net, device, obs)
            if record_entropy:
                ent = float(-np.sum(np.where(probs > 1e-12, probs * np.log(np.clip(probs, 1e-12, 1.0)), 0.0)))
                order = np.argsort(-probs).tolist()
                entropies.append((step_i, ent, order))
            if temp <= 0.0 or rng is None:
                a = int(np.argmax(probs))
            else:
                # Temperature sampling over legal actions
                logits = np.log(np.clip(probs, 1e-12, 1.0)) / temp
                logits -= logits.max()
                p_t = np.exp(logits)
                p_t /= p_t.sum()
                a = int(rng.choice(obs.num_actions, p=p_t))
        actions.append(a)
        obs = env.step(a)
        step_i += 1
    return {
        "win": bool(obs.victory),
        "done": bool(obs.done),
        "stuck": bool(obs.stuck),
        "end_hp": int(obs.hp),
        "steps": len(actions),
        "actions": actions,
        "entropies": entropies,
    }


def verify_clean_replay(env: Sts2Env, spec: dict, actions: list[int], stall_turns: int) -> tuple[bool, int, str]:
    """Replay winning action sequence from a completely fresh reset_combat (no snapshot/restore)."""
    obs = env.reset_combat(spec)
    # Apply same loop_cut / stall_turns flags via raw send so reset matches exactly
    for step_i, a in enumerate(actions):
        if obs.done or obs.stuck or a >= obs.num_actions:
            return False, int(obs.hp), f"mismatch_at_step_{step_i}_done={obs.done}_nact={obs.num_actions}"
        obs = env.step(int(a))
    return bool(obs.done and obs.victory), int(obs.hp), "ok" if (obs.done and obs.victory) else f"ended_done={obs.done}_vic={obs.victory}"


def reset_combat_with_flags(env: Sts2Env, spec: dict, stall_turns: int):
    req = {"op": "reset", "mode": "combat", "spec": spec, "labels": False, "loop_cut": True, "stall_turns": stall_turns}
    env._proc.send(req)
    from sts2env.protocol import parse
    return parse(env._proc.recv())


def solve_one(
    env: Sts2Env,
    item: dict,
    net,
    device: str,
    rollouts: int,
    branch_points: int,
    branch_k: int,
    timeout_s: float,
    max_steps: int,
    stall_turns: int,
    search_seed: int,
) -> dict:
    t0 = time.time()
    spec = make_combat_spec(item)
    rng = np.random.default_rng(search_seed)

    # 1. Clean reset + snapshot root (slot 0) + frozen greedy baseline
    obs0 = reset_combat_with_flags(env, spec, stall_turns)
    env.snapshot(0)
    greedy = _run_episode(env, obs0, net, device, None, temp=0.0, max_steps=max_steps, record_entropy=True)
    if greedy["win"]:
        return {
            "id": item["id"],
            "act": item["act"],
            "encounter": item["encounter"],
            "combat_seed": spec["seed"],
            "status": "already_won_by_policy",
            "policy_win": True,
            "policy_end_hp": greedy["end_hp"],
            "policy_steps": greedy["steps"],
            "search_rollouts": 0,
            "seconds": round(time.time() - t0, 2),
        }

    # 2. Phase A: Fast temperature-sampled rollouts from exact root snapshot(0)
    temps = (0.5, 0.8, 1.1)
    used_rollouts = 0
    best_cand = greedy
    found_actions: list[int] | None = None
    found_hp = 0
    found_phase = ""

    for r_idx in range(rollouts):
        if time.time() - t0 >= timeout_s:
            break
        obs = env.restore(0)
        t_cur = temps[r_idx % len(temps)]
        res = _run_episode(env, obs, net, device, rng, temp=t_cur, max_steps=max_steps, record_entropy=(r_idx < 4))
        used_rollouts += 1
        if res["win"]:
            found_actions = res["actions"]
            found_hp = res["end_hp"]
            found_phase = f"temp_rollout_{r_idx+1}_T{t_cur}"
            break
        if res["steps"] > best_cand["steps"] and res["entropies"]:
            best_cand = res

    # 3. Phase B: High-entropy decision-point branching along greedy/best trajectory
    if found_actions is None and branch_points > 0 and time.time() - t0 < timeout_s:
        # Pick top uncertain decision steps from greedy trajectory
        ranked_pts = sorted(greedy["entropies"], key=lambda x: -x[1])[:branch_points]
        for step_idx, _ent, order in ranked_pts:
            if time.time() - t0 >= timeout_s or found_actions is not None:
                break
            alts = order[1 : 1 + branch_k]
            if not alts:
                continue
            # Replay prefix from root snapshot(0) to step_idx, store in slot 1
            obs = env.restore(0)
            prefix = greedy["actions"][:step_idx]
            ok = True
            for a in prefix:
                if obs.done or obs.stuck:
                    ok = False
                    break
                obs = env.step(a)
            if not ok or obs.done:
                continue
            env.snapshot(1)
            for alt_a in alts:
                if time.time() - t0 >= timeout_s:
                    break
                # Try greedy continuation first, then one mild temp continuation
                for cont_temp in (0.0, 0.7):
                    if time.time() - t0 >= timeout_s:
                        break
                    obs_b = env.restore(1)
                    res = _run_episode(env, obs_b, net, device, rng, temp=cont_temp, max_steps=max_steps - step_idx, first_action=alt_a)
                    used_rollouts += 1
                    if res["win"]:
                        found_actions = prefix + res["actions"]
                        found_hp = res["end_hp"]
                        found_phase = f"branch_s{step_idx}_a{alt_a}_T{cont_temp}"
                        break
                if found_actions is not None:
                    break

    if found_actions is None:
        return {
            "id": item["id"],
            "act": item["act"],
            "encounter": item["encounter"],
            "combat_seed": spec["seed"],
            "status": "unknown",
            "policy_win": False,
            "policy_steps": greedy["steps"],
            "search_rollouts": used_rollouts,
            "seconds": round(time.time() - t0, 2),
        }

    # 4. Independent clean-engine verification (fresh reset_combat, NO snapshot/restore)
    obs_v = reset_combat_with_flags(env, spec, stall_turns)
    v_ok = True
    v_note = "ok"
    for step_i, a in enumerate(found_actions):
        if obs_v.done or obs_v.stuck or a >= obs_v.num_actions:
            v_ok = False
            v_note = f"step_{step_i}_done={obs_v.done}_nact={obs_v.num_actions}"
            break
        obs_v = env.step(int(a))
    if v_ok and not (obs_v.done and obs_v.victory):
        v_ok = False
        v_note = f"final_done={obs_v.done}_vic={obs_v.victory}"

    return {
        "id": item["id"],
        "act": item["act"],
        "encounter": item["encounter"],
        "combat_seed": spec["seed"],
        "status": "verified_solvable_mistake" if v_ok else "verify_failed",
        "verify_note": v_note,
        "policy_win": False,
        "policy_steps": greedy["steps"],
        "search_win": True,
        "search_phase": found_phase,
        "search_rollouts": used_rollouts,
        "verified_end_hp": int(obs_v.hp),
        "verified_steps": len(found_actions),
        "winning_actions": found_actions if v_ok else None,
        "seconds": round(time.time() - t0, 2),
        "spec": spec if v_ok else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Small-scale same-initial-state Boss mistake finder & verifier.")
    ap.add_argument("--items", required=True, help="JSONL of failed Boss snapshots (from eval_v1140.py --seed-prefix A10MINE)")
    ap.add_argument("--ckpt", required=True, help="Frozen checkpoint (.pt)")
    ap.add_argument("--weights", choices=("ema", "raw"), default="ema")
    ap.add_argument("--out", required=True, help="Output verification results JSONL")
    ap.add_argument("--export-solver-jsonl", default=None, help="Optional SolverPool-compatible labeled.jsonl output for verified mistakes")
    ap.add_argument("--limit", type=int, default=32, help="Max items to test in small-scale verification")
    ap.add_argument("--acts", default="0,1,2", help="Comma-separated acts to include (0=A1, 1=A2, 2=A3)")
    ap.add_argument("--rollouts", type=int, default=24, help="Phase A stochastic rollouts per item")
    ap.add_argument("--branch-points", type=int, default=6, help="Phase B top-entropy decision points to branch")
    ap.add_argument("--branch-k", type=int, default=2, help="Alternative actions per branch point")
    ap.add_argument("--timeout-s", type=float, default=30.0, help="Per-item wall-clock budget (seconds)")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--stall-turns", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.set_num_threads(2)
    net, blob, src_key = load_net_weights(Path(a.ckpt), a.weights)
    net = net.to(a.device)
    net.infer_ready()
    print(f"Loaded {a.ckpt} (v{blob['version']}, {a.weights}/{src_key}) on {a.device}", flush=True)

    act_set = {int(x) for x in a.acts.split(",") if x.strip()}
    all_items = [json.loads(ln) for ln in Path(a.items).read_text(encoding="utf-8").splitlines() if ln.strip()]
    all_items = [it for it in all_items if int(it.get("act", -1)) in act_set]

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        for ln in out_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                done_ids.add(json.loads(ln)["id"])

    todo = [it for it in all_items if it["id"] not in done_ids][: a.limit if a.limit > 0 else None]
    print(f"Items matching acts={sorted(act_set)}: {len(all_items)} total, {len(done_ids)} done, {len(todo)} to test", flush=True)

    env = Sts2Env()
    t_start = time.time()
    counts = {"already_won_by_policy": 0, "verified_solvable_mistake": 0, "unknown": 0, "verify_failed": 0, "error": 0}
    try:
        with out_path.open("a", encoding="utf-8") as fh:
            for idx, it in enumerate(todo, 1):
                try:
                    rec = solve_one(
                        env, it, net, a.device,
                        rollouts=a.rollouts, branch_points=a.branch_points, branch_k=a.branch_k,
                        timeout_s=a.timeout_s, max_steps=a.max_steps, stall_turns=a.stall_turns,
                        search_seed=100003 + idx,
                    )
                except EngineError as err:
                    rec = {"id": it["id"], "act": it["act"], "encounter": it["encounter"], "status": "error", "error": str(err)[:300]}
                st = rec["status"]
                counts[st] = counts.get(st, 0) + 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if a.export_solver_jsonl and st == "verified_solvable_mistake":
                    with open(a.export_solver_jsonl, "a", encoding="utf-8") as sfh:
                        sfh.write(json.dumps({
                            "id": rec["id"],
                            "act": rec["act"],
                            "encounter": rec["encounter"],
                            "win": True,
                            "end_hp": rec["verified_end_hp"],
                            "spec": rec["spec"],
                        }, ensure_ascii=False) + "\n")
                print(
                    f"[{idx}/{len(todo)}] A{it['act']+1} {it['encounter']} ({it['id']}) -> {st} "
                    f"(rollouts={rec.get('search_rollouts', 0)}, hp={rec.get('verified_end_hp', rec.get('policy_end_hp', 0))}, {rec.get('seconds', 0):.1f}s)",
                    flush=True,
                )
    finally:
        env.close()

    wall = max(time.time() - t_start, 1e-6)
    policy_lost = counts["verified_solvable_mistake"] + counts["unknown"] + counts["verify_failed"]
    rescue = counts["verified_solvable_mistake"] / max(policy_lost, 1) * 100.0
    yield_per_hr = counts["verified_solvable_mistake"] / (wall / 3600.0)
    print(
        f"\n=== Summary ({wall:.1f}s) ===\n"
        f"tested={len(todo)} | already_won={counts['already_won_by_policy']} | policy_lost={policy_lost} | "
        f"verified_solvable_mistake={counts['verified_solvable_mistake']} ({rescue:.1f}% of policy_lost) | "
        f"unknown={counts['unknown']} | verify_failed={counts['verify_failed']} | errors={counts['error']}\n"
        f"Yield: {yield_per_hr:.1f} verified mistakes / hour (single engine worker)",
        flush=True,
    )


if __name__ == "__main__":
    main()
