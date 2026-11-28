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
import hashlib
import json
import queue
import sys
import threading
import time
from collections import defaultdict
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


class BatchInferer:
    """Dynamic micro-batching CUDA/CPU inference server shared by N concurrent Sts2Env threads."""

    def __init__(self, net, device: str, max_batch: int = 16, wait_ms: float = 1.5):
        self.net = net
        self.device = device
        self.max_batch = max(1, int(max_batch))
        self.wait_s = max(0.0, float(wait_ms) / 1000.0)
        self.use_bf16 = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.batches = 0
        self.reqs = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def eval(self, obs) -> tuple[np.ndarray, float]:
        ev = threading.Event()
        box: list = []
        self._q.put((obs, ev, box))
        ev.wait()
        if len(box) == 2:
            raise box[1]
        return box[0]

    def close(self) -> None:
        self._stop.set()
        self._q.put(None)
        self._thread.join(timeout=5.0)

    @torch.no_grad()
    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                break
            batch_items = [item]
            deadline = time.perf_counter() + self.wait_s
            while len(batch_items) < self.max_batch:
                rem = deadline - time.perf_counter()
                if rem <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=rem)
                except queue.Empty:
                    break
                if nxt is None:
                    self._stop.set()
                    break
                batch_items.append(nxt)
            try:
                obs_list = [x[0] for x in batch_items]
                n = len(obs_list)
                rg = Ragged.from_obs(obs_list, np.full(n, MODE_COMBAT, dtype=np.int64))
                batch = collate_ragged(rg, attend=bool(self.net.cfg.macro_attend)).to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    logits, values = self.net.forward_infer(batch)
                logits_f = logits.float().cpu()
                vals_f = values.float().cpu().tolist()
                self.batches += 1
                self.reqs += n
                for i, (ob, ev, box) in enumerate(batch_items):
                    lg = logits_f[i, : ob.num_actions]
                    probs = torch.softmax(lg, dim=-1).numpy().astype(np.float64)
                    box.append((probs, float(vals_f[i])))
                    ev.set()
            except Exception as err:
                for _ob, ev, box in batch_items:
                    box.extend([None, err])
                    ev.set()


@torch.no_grad()
def policy_eval_obs(net, device: str, obs, inferer: BatchInferer | None = None) -> tuple[np.ndarray, float]:
    """Return legal action probabilities (float64) and scalar value for a single combat Obs."""
    if inferer is not None:
        return inferer.eval(obs)
    rg = Ragged.from_obs([obs], np.array([MODE_COMBAT], dtype=np.int64))
    batch = collate_ragged(rg, attend=bool(net.cfg.macro_attend)).to(device)
    use_bf16 = str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
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
    inferer: BatchInferer | None = None,
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
            probs, _ = policy_eval_obs(net, device, obs, inferer=inferer)
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
    inferer: BatchInferer | None = None,
) -> dict:
    t0 = time.time()
    spec = make_combat_spec(item)
    rng = np.random.default_rng(search_seed)
    snap_hash = hashlib.sha256(
        json.dumps(
            {"act": item["act"], "encounter": item["encounter"], "snap": item["snap"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    # 1. Clean reset + snapshot root (slot 0) + frozen greedy baseline
    obs0 = reset_combat_with_flags(env, spec, stall_turns)
    env.snapshot(0)
    greedy = _run_episode(env, obs0, net, device, None, temp=0.0, max_steps=max_steps, record_entropy=True, inferer=inferer)
    if greedy["win"]:
        return {
            "id": item["id"],
            "run_seed": item.get("run_seed"),
            "snap_hash": snap_hash,
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
        res = _run_episode(env, obs, net, device, rng, temp=t_cur, max_steps=max_steps, record_entropy=(r_idx < 4), inferer=inferer)
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
                    res = _run_episode(
                        env, obs_b, net, device, rng, temp=cont_temp,
                        max_steps=max_steps - step_idx, first_action=alt_a, inferer=inferer,
                    )
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
            "run_seed": item.get("run_seed"),
            "snap_hash": snap_hash,
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
        "run_seed": item.get("run_seed"),
        "snap_hash": snap_hash,
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
    ap.add_argument("--summary-out", default=None, help="Optional summary JSON output path")
    ap.add_argument("--export-solver-jsonl", default=None, help="Optional SolverPool-compatible labeled.jsonl output for verified mistakes")
    ap.add_argument("--limit", type=int, default=32, help="Max items to test (0 = all)")
    ap.add_argument("--workers", type=int, default=1, help="Concurrent Sts2Headless engine workers sharing the frozen model")
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
    print(f"Loaded {a.ckpt} (v{blob['version']}, {a.weights}/{src_key}) on {a.device} | workers={a.workers}", flush=True)

    act_set = {int(x) for x in a.acts.split(",") if x.strip()}
    raw_items = [json.loads(ln) for ln in Path(a.items).read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Deduplicate input items by id while preserving order
    seen_in = set()
    all_items = []
    for it in raw_items:
        if int(it.get("act", -1)) in act_set and it["id"] not in seen_in:
            seen_in.add(it["id"])
            all_items.append(it)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        for ln in out_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                done_ids.add(json.loads(ln)["id"])

    todo = [it for it in all_items if it["id"] not in done_ids]
    if a.limit > 0:
        todo = todo[: a.limit]
    print(f"Items matching acts={sorted(act_set)}: {len(all_items)} unique total, {len(done_ids)} done, {len(todo)} to test", flush=True)

    work_q: queue.Queue = queue.Queue()
    for idx, it in enumerate(todo, 1):
        work_q.put((idx, it))

    inferer = BatchInferer(net, a.device, max_batch=max(1, a.workers), wait_ms=1.5) if a.workers > 1 else None
    write_lock = threading.Lock()
    counts = {"already_won_by_policy": 0, "verified_solvable_mistake": 0, "unknown": 0, "verify_failed": 0, "error": 0}
    by_enc: dict[str, dict[str, int]] = defaultdict(lambda: {"tested": 0, "already_won": 0, "policy_lost": 0, "verified": 0, "unknown": 0, "verify_failed": 0, "error": 0})
    verified_hashes: set[str] = set()
    exported_hashes: set[str] = set()
    if a.export_solver_jsonl and Path(a.export_solver_jsonl).exists():
        for ln in Path(a.export_solver_jsonl).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                er = json.loads(ln)
                exported_hashes.add(er.get("snap_hash") or er.get("id"))
    completed = [0]
    t_start = time.time()

    fh = out_path.open("a", encoding="utf-8")
    if a.export_solver_jsonl:
        Path(a.export_solver_jsonl).parent.mkdir(parents=True, exist_ok=True)
    sfh = open(a.export_solver_jsonl, "a", encoding="utf-8") if a.export_solver_jsonl else None

    def worker_fn(wid: int) -> None:
        env = Sts2Env()
        try:
            while True:
                try:
                    idx, it = work_q.get_nowait()
                except queue.Empty:
                    break
                try:
                    rec = solve_one(
                        env, it, net, a.device,
                        rollouts=a.rollouts, branch_points=a.branch_points, branch_k=a.branch_k,
                        timeout_s=a.timeout_s, max_steps=a.max_steps, stall_turns=a.stall_turns,
                        search_seed=100003 + idx,
                        inferer=inferer,
                    )
                except EngineError as err:
                    rec = {"id": it["id"], "run_seed": it.get("run_seed"), "act": it["act"], "encounter": it["encounter"], "status": "error", "error": str(err)[:300]}
                except Exception as err:
                    rec = {"id": it["id"], "run_seed": it.get("run_seed"), "act": it["act"], "encounter": it["encounter"], "status": "error", "error": f"{type(err).__name__}: {err}"[:300]}
                st = rec["status"]
                with write_lock:
                    completed[0] += 1
                    counts[st] = counts.get(st, 0) + 1
                    ec = by_enc[it["encounter"]]
                    ec["tested"] += 1
                    if st == "already_won_by_policy":
                        ec["already_won"] += 1
                    elif st == "verified_solvable_mistake":
                        ec["policy_lost"] += 1
                        ec["verified"] += 1
                        verified_hashes.add(rec.get("snap_hash") or rec["id"])
                    elif st == "unknown":
                        ec["policy_lost"] += 1
                        ec["unknown"] += 1
                    elif st == "verify_failed":
                        ec["policy_lost"] += 1
                        ec["verify_failed"] += 1
                    else:
                        ec["error"] += 1
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    if sfh is not None and st == "verified_solvable_mistake":
                        hk = rec.get("snap_hash") or rec["id"]
                        if hk not in exported_hashes:
                            exported_hashes.add(hk)
                            sfh.write(json.dumps({
                                "id": rec["id"],
                                "run_seed": rec.get("run_seed"),
                                "snap_hash": rec.get("snap_hash"),
                                "act": rec["act"],
                                "encounter": rec["encounter"],
                                "pool_eligible": True,
                                "win": True,
                                "solver_status": "verified_solvable_mistake",
                                "end_hp": rec["verified_end_hp"],
                                "spec": rec["spec"],
                            }, ensure_ascii=False) + "\n")
                            sfh.flush()
                    if completed[0] <= 10 or completed[0] % 10 == 0 or st == "verified_solvable_mistake" or completed[0] == len(todo):
                        el = max(time.time() - t_start, 1e-6)
                        y_now = len(verified_hashes) / (el / 3600.0)
                        print(
                            f"[{completed[0]:>3d}/{len(todo)} w{wid}] A{it['act']+1} {it['encounter']} ({it['id']}) -> {st} "
                            f"(rollouts={rec.get('search_rollouts', 0)}, hp={rec.get('verified_end_hp', rec.get('policy_end_hp', 0))}, {rec.get('seconds', 0):.1f}s | yield={y_now:.1f}/h)",
                            flush=True,
                        )
        finally:
            env.close()

    try:
        n_w = max(1, min(a.workers, max(1, len(todo))))
        threads = [threading.Thread(target=worker_fn, args=(w,), daemon=True) for w in range(n_w)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        if inferer is not None:
            inferer.close()
        fh.close()
        if sfh is not None:
            sfh.close()

    wall = max(time.time() - t_start, 1e-6)
    policy_lost = counts["verified_solvable_mistake"] + counts["unknown"] + counts["verify_failed"]
    lost_rate = policy_lost / max(len(todo), 1) * 100.0
    rescue = counts["verified_solvable_mistake"] / max(policy_lost, 1) * 100.0
    n_uniq = len(verified_hashes)
    yield_per_hr = n_uniq / (wall / 3600.0)
    summary = {
        "ckpt": str(a.ckpt),
        "version": int(blob["version"]),
        "weights": a.weights,
        "acts": sorted(act_set),
        "workers": max(1, a.workers),
        "rollouts": a.rollouts,
        "branch_points": a.branch_points,
        "branch_k": a.branch_k,
        "timeout_s": a.timeout_s,
        "wall_s": round(wall, 1),
        "tested": len(todo),
        "already_won_by_policy": counts["already_won_by_policy"],
        "policy_lost": policy_lost,
        "policy_lost_rate": round(lost_rate, 2),
        "verified_solvable_mistake": counts["verified_solvable_mistake"],
        "unique_verified_mistakes": n_uniq,
        "rescue_rate_of_policy_lost": round(rescue, 2),
        "unknown": counts["unknown"],
        "verify_failed": counts["verify_failed"],
        "errors": counts["error"],
        "unique_verified_per_hour": round(yield_per_hr, 1),
        "by_encounter": dict(sorted(by_enc.items())),
    }
    if a.summary_out:
        sp = Path(a.summary_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\n=== Summary ({wall:.1f}s, {a.workers} workers) ===\n"
        f"tested={len(todo)} | already_won={counts['already_won_by_policy']} | "
        f"policy_lost={policy_lost} ({lost_rate:.1f}% of tested) | "
        f"verified_solvable_mistake={counts['verified_solvable_mistake']} (unique={n_uniq}, {rescue:.1f}% of policy_lost) | "
        f"unknown={counts['unknown']} | verify_failed={counts['verify_failed']} | errors={counts['error']}\n"
        f"Yield: {yield_per_hr:.1f} unique verified mistakes / hour ({a.workers} workers)",
        flush=True,
    )
    print("Per-encounter breakdown:")
    for enc, ec in sorted(by_enc.items()):
        rr = ec["verified"] / max(ec["policy_lost"], 1) * 100.0
        print(f"  {enc:30s}: tested={ec['tested']:3d} lost={ec['policy_lost']:3d} verified={ec['verified']:2d} ({rr:4.1f}%) unknown={ec['unknown']:3d}")


if __name__ == "__main__":
    main()
