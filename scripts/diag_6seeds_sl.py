#!/usr/bin/env python3
"""6-seed surgical diagnostic for Full-Combat SL + Out-of-Combat No-SL.

Tests:
  1. Reproducibility audit (Row 3 check):
       - Replays v4879 under the original wall-clock timeout + shared-RNG protocol
         and compares against the saved 128-seed `v4879` log.
       - Replays v4879 TWICE under a deterministic protocol (no wall-clock timeout
         cutoff, fixed per-combat RNG seed, fixed batch=1 FP32 inference) to verify
         100% reproducibility before attributing any divergence to the macro policy.
  2. Exact first macro divergence (`v4879 [EMA]` vs `U08 [raw]`) under the deterministic
     protocol with `labels=True` (MAP, CARD_REWARD, REST_SITE, SHOP, EVENT, REWARDS).
  3. Single-step counterfactual swaps at the first macro divergence point:
       - `u08_with_base_step`: run U08, override ONLY step k with v4879's action.
       - `base_with_u08_step`: run v4879, override ONLY step k with U08's action.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
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

from eval_combat_sl import is_perfect_win, potions_held, rank_key  # noqa: E402
from eval_v1140 import load_net_weights  # noqa: E402
from sts2env.env import Sts2Env  # noqa: E402
from sts2env.protocol import DecisionKind, TokenKind, Zone, parse  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402

SEEDS = [
    ("A10VAL0001", 1),
    ("A10VAL0010", 10),
    ("A10VAL0054", 54),
    ("A10VAL0057", 57),
    ("A10VAL0016", 16),
    ("A10VAL0103", 103),
]
TEMPS = (0.4, 0.7, 1.0, 0.5, 0.85, 1.15)


def reset_run_labeled(env: Sts2Env, seed: str, labels: bool = True):
    req = {
        "op": "reset",
        "mode": "run",
        "character": "NECROBINDER",
        "ascension": 10,
        "seed": seed,
        "snapshots": True,
        "route_summary": True,
        "labels": labels,
        "loop_cut": True,
        "stall_turns": 20,
    }
    env._proc.send(req)
    return parse(env._proc.recv())


def deck_size(obs) -> int:
    if obs.tok_ints.shape[0] == 0:
        return 0
    return int(((obs.tok_ints[:, 0] == TokenKind.CARD) & (obs.tok_ints[:, 2] == Zone.DECK)).sum())


def combat_seed(seed: str, combat_idx: int, act: int, floor: int) -> int:
    h = hashlib.sha256(f"{seed}|c={combat_idx}|a={act}|f={floor}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFF


class DetInferer:
    """Thread-safe batch=1 FP32 inference (zero dynamic-batching / BF16 nondeterminism)."""

    def __init__(self, net, device: str):
        self.net = net
        self.device = device
        self._lock = threading.Lock()

    @torch.no_grad()
    def eval(self, obs) -> np.ndarray:
        rg = Ragged.from_obs([obs], np.array([MODE_RUN], dtype=np.int64))
        batch = collate_ragged(rg, attend=bool(self.net.cfg.macro_attend)).to(self.device)
        with self._lock:
            logits, _ = self.net.forward_infer(batch)
            lg = logits[0, : obs.num_actions].float().cpu().double()
        probs = torch.softmax(lg, dim=-1).numpy()
        probs = np.maximum(probs, 0.0)
        s = float(probs.sum())
        return (probs / s) if s > 0 else np.full(obs.num_actions, 1.0 / max(obs.num_actions, 1), dtype=np.float64)


def choose_det(obs, inferer: DetInferer, temp: float, rng: np.random.Generator | None) -> tuple[int, np.ndarray]:
    if obs.num_actions <= 1:
        return 0, np.array([1.0], dtype=np.float64)
    probs = inferer.eval(obs)
    if temp <= 0.0 or rng is None:
        return int(np.argmax(probs)), probs
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temp
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(obs.num_actions, p=p)), probs


def rollout_combat(env: Sts2Env, obs, inferer: DetInferer, temp: float, rng, start_cw: int, max_steps: int = 400):
    actions: list[int] = []
    steps = 0
    while (
        not obs.done
        and not obs.stuck
        and obs.in_combat
        and obs.combats_won == start_cw
        and obs.num_actions > 0
        and steps < max_steps
    ):
        a, _ = choose_det(obs, inferer, temp=temp, rng=rng)
        actions.append(a)
        obs = env.step(a)
        steps += 1
    if obs.stuck or (steps >= max_steps and obs.in_combat and obs.combats_won == start_cw and not obs.done):
        won, stuck = False, True
    elif obs.done:
        won, stuck = bool(obs.victory), False
    else:
        won, stuck = True, False
    return {
        "won": won,
        "stuck": stuck,
        "run_victory": bool(obs.done and obs.victory),
        "end_hp": int(obs.hp),
        "end_max_hp": int(obs.max_hp),
        "end_pots": potions_held(obs) if won else 0,
        "steps": len(actions),
        "actions": actions,
    }, obs


def trace_run(
    env: Sts2Env,
    seed: str,
    seed_idx: int,
    inferer: DetInferer,
    *,
    deterministic_sl: bool = True,
    override_macro: dict[int, int] | None = None,
    monster_trials: int = 4,
    elite_trials: int = 12,
    boss_trials: int = 20,
    potion_hp_frac: float = 0.20,
) -> dict:
    obs = reset_run_labeled(env, seed, labels=True)
    shared_rng = np.random.default_rng(770001 + seed_idx * 997)
    budgets = (
        {"Monster": (monster_trials, 9999.0), "Elite": (elite_trials, 9999.0), "Boss": (boss_trials, 9999.0)}
        if deterministic_sl
        else {"Monster": (monster_trials, 4.0), "Elite": (elite_trials, 8.0), "Boss": (boss_trials, 15.0)}
    )

    macro_log: list[dict] = []
    combat_log: list[dict] = []
    macro_idx = 0
    combat_idx = 0
    run_steps = 0
    stuck = False
    death_cause = None

    while not obs.done and not obs.stuck and obs.num_actions > 0 and run_steps < 6000:
        if not obs.in_combat:
            a, probs = choose_det(obs, inferer, temp=0.0, rng=None)
            orig_a = a
            forced = False
            if override_macro and macro_idx in override_macro:
                cand_override = int(override_macro[macro_idx])
                if 0 <= cand_override < obs.num_actions:
                    a = cand_override
                    forced = (a != orig_a)

            labels = list(obs.labels) if obs.labels else [f"a{i}" for i in range(obs.num_actions)]
            order = np.argsort(-probs)
            top1 = int(order[0])
            top2 = int(order[1]) if len(order) > 1 else None
            dec_name = DecisionKind(int(obs.decision)).name if int(obs.decision) in DecisionKind._value2member_map_ else str(obs.decision)

            macro_log.append({
                "m_idx": macro_idx,
                "act": int(obs.act),
                "floor": int(obs.floor),
                "decision": dec_name,
                "hp": int(obs.hp),
                "max_hp": int(obs.max_hp),
                "pots": potions_held(obs),
                "deck": deck_size(obs),
                "n_act": int(obs.num_actions),
                "action": a,
                "orig_action": orig_a,
                "forced": forced,
                "chosen_label": labels[a] if a < len(labels) else f"a{a}",
                "chosen_prob": round(float(probs[a]), 4),
                "top2_label": (labels[top2] if (top2 is not None and top2 < len(labels)) else None),
                "top2_prob": round(float(probs[top2]), 4) if top2 is not None else None,
                "labels": labels,
            })
            obs = env.step(a)
            macro_idx += 1
            run_steps += 1
            continue

        # Combat
        cinfo = (obs.extra or {}).get("combat") or {}
        room = str(cinfo.get("room") or "Monster")
        if room not in ("Monster", "Elite", "Boss"):
            room = "Boss" if int(obs.floor) in (17, 33, 48, 49) else "Monster"
        encounter = str(cinfo.get("encounter") or f"UNK_A{obs.act+1}_F{obs.floor}")
        act = int(cinfo.get("act", obs.act))
        floor = int(obs.floor)
        start_hp = int(obs.hp)
        start_max_hp = int(obs.max_hp)
        start_pots = potions_held(obs)
        start_cw = int(obs.combats_won)
        pot_val = potion_hp_frac * max(start_max_hp, 1)

        rng_c = np.random.default_rng(combat_seed(seed, combat_idx, act, floor)) if deterministic_sl else shared_rng
        t_c0 = time.time()
        env.snapshot(0)
        greedy_res, obs_after_greedy = rollout_combat(env, obs, inferer, temp=0.0, rng=None, start_cw=start_cw)
        best_res = greedy_res
        best_idx = 0
        best_key = rank_key(greedy_res, pot_val, is_greedy=True)
        last_idx = 0
        last_obs = obs_after_greedy

        max_extra, fight_timeout = budgets[room]
        perfect_skip = is_perfect_win(greedy_res, start_hp, start_pots)
        extra_used = 0
        hit_timeout = False

        if (not perfect_skip) and max_extra > 0:
            for ti in range(1, max_extra + 1):
                if time.time() - t_c0 >= fight_timeout:
                    hit_timeout = True
                    break
                obs_r = env.restore(0)
                t_cur = TEMPS[(ti - 1) % len(TEMPS)]
                cand_res, obs_end = rollout_combat(env, obs_r, inferer, temp=t_cur, rng=rng_c, start_cw=start_cw)
                extra_used += 1
                last_idx = ti
                last_obs = obs_end
                k = rank_key(cand_res, pot_val, is_greedy=False)
                if k > best_key:
                    best_key = k
                    best_res = cand_res
                    best_idx = ti
                    if is_perfect_win(cand_res, start_hp, start_pots):
                        break

        if best_idx == last_idx:
            obs = last_obs
        else:
            obs_re = env.restore(0)
            for act_i in best_res["actions"]:
                obs_re = env.step(int(act_i))
            obs = obs_re

        run_steps += best_res["steps"]
        if best_res["stuck"]:
            stuck = True
        if not best_res["won"]:
            death_cause = f"A{act+1}F{floor} {room} {encounter}"

        combat_log.append({
            "c_idx": combat_idx,
            "act": act,
            "floor": floor,
            "room": room,
            "encounter": encounter,
            "start_hp": start_hp,
            "start_max_hp": start_max_hp,
            "start_pots": start_pots,
            "greedy_won": greedy_res["won"],
            "greedy_hp": greedy_res["end_hp"],
            "greedy_pots": greedy_res["end_pots"],
            "final_won": best_res["won"],
            "final_hp": best_res["end_hp"],
            "final_pots": best_res["end_pots"],
            "extra_trials": extra_used,
            "hit_timeout": hit_timeout,
            "rescued": (not greedy_res["won"]) and best_res["won"],
            "improved_win": greedy_res["won"] and best_res["won"] and best_idx != 0,
            "sl_s": round(time.time() - t_c0, 2),
        })
        combat_idx += 1
        if stuck:
            break

    if obs.done and not obs.victory and death_cause is None:
        death_cause = f"A{obs.act+1}F{obs.floor} NON_COMBAT"

    return {
        "seed": seed,
        "floor": int(obs.floor),
        "act": int(obs.act),
        "win": bool(obs.done and obs.victory),
        "stuck": bool(stuck),
        "death_cause": death_cause,
        "macros": macro_log,
        "combats": combat_log,
    }


def find_first_macro_div(tr_b: dict, tr_u: dict) -> dict | None:
    mb, mu = tr_b["macros"], tr_u["macros"]
    for i in range(min(len(mb), len(mu))):
        a, b = mb[i], mu[i]
        # Must be same state!
        state_same = (a["act"] == b["act"] and a["floor"] == b["floor"] and a["decision"] == b["decision"] and a["labels"] == b["labels"])
        if not state_same:
            return {"type": "STATE_DESYNC_BEFORE_MACRO_DIV", "m_idx": i, "base": a, "u08": b}
        if a["action"] != b["action"]:
            return {"type": "MACRO_DIV", "m_idx": i, "base": a, "u08": b}
    return None


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net_b, _, _ = load_net_weights(Path("runs/step1-v3965-solver-a2-s0/latest.pt"), "ema")
    net_u, _, _ = load_net_weights(Path("runs/macro-adapt-v4879-sl-s0/milestones/u0008.pt"), "raw")
    net_b = net_b.to(device).eval().infer_ready()
    net_u = net_u.to(device).eval().infer_ready()
    net_b.fast = device == "cuda"
    net_u.fast = device == "cuda"
    inf_b = DetInferer(net_b, device)
    inf_u = DetInferer(net_u, device)

    saved_b = {r["seed"]: r for r in (json.loads(l) for l in Path("runs/eval-combat-sl-v4879-a10val-128/sl_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    saved_u = {r["seed"]: r for r in (json.loads(l) for l in Path("runs/eval-combat-sl-macro-u0008-raw-a10val-128/sl_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}

    out_dir = Path("runs/diag-6seeds-v4879-vs-u08")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    lock = threading.Lock()

    def worker(seed: str, sidx: int) -> None:
        env = Sts2Env()
        try:
            # 1. Original timeout protocol replay of v4879 (check Row 3 variance vs saved 128-seed log)
            b_orig = trace_run(env, seed, sidx, inf_b, deterministic_sl=False)
            # 2. Deterministic (no timeout cutoff + per-combat RNG + batch=1 FP32) v4879 TWICE
            b_det1 = trace_run(env, seed, sidx, inf_b, deterministic_sl=True)
            b_det2 = trace_run(env, seed, sidx, inf_b, deterministic_sl=True)
            def _strip_t(cs):
                return [{k: v for k, v in c.items() if k != "sl_s"} for c in cs]
            det_identical = (
                b_det1["floor"] == b_det2["floor"]
                and _strip_t(b_det1["combats"]) == _strip_t(b_det2["combats"])
                and [(m["floor"], m["decision"], m["action"]) for m in b_det1["macros"]]
                == [(m["floor"], m["decision"], m["action"]) for m in b_det2["macros"]]
            )
            # 3. Deterministic U08
            u_det = trace_run(env, seed, sidx, inf_u, deterministic_sl=True)
            div = find_first_macro_div(b_det1, u_det)

            # 4. Single-step counterfactual swaps at first macro divergence
            cf_u_to_b = None
            cf_b_to_u = None
            if div and div["type"] == "MACRO_DIV":
                mi = div["m_idx"]
                cf_u_to_b = trace_run(env, seed, sidx, inf_u, deterministic_sl=True, override_macro={mi: div["base"]["action"]})
                cf_b_to_u = trace_run(env, seed, sidx, inf_b, deterministic_sl=True, override_macro={mi: div["u08"]["action"]})

            rec = {
                "seed": seed,
                "saved_128": {
                    "base_fl": saved_b[seed]["floor"], "base_act": saved_b[seed]["act"],
                    "u08_fl": saved_u[seed]["floor"], "u08_act": saved_u[seed]["act"],
                },
                "orig_timeout_replay_base": {"fl": b_orig["floor"], "act": b_orig["act"], "death": b_orig["death_cause"]},
                "det_v4879_identical": det_identical,
                "det_base": {"fl": b_det1["floor"], "act": b_det1["act"], "win": b_det1["win"], "death": b_det1["death_cause"], "n_combats": len(b_det1["combats"]), "n_macros": len(b_det1["macros"])},
                "det_u08": {"fl": u_det["floor"], "act": u_det["act"], "win": u_det["win"], "death": u_det["death_cause"], "n_combats": len(u_det["combats"]), "n_macros": len(u_det["macros"])},
                "first_div": div,
                "cf_u08_take_base_step": ({"fl": cf_u_to_b["floor"], "act": cf_u_to_b["act"], "death": cf_u_to_b["death_cause"]} if cf_u_to_b else None),
                "cf_base_take_u08_step": ({"fl": cf_b_to_u["floor"], "act": cf_b_to_u["act"], "death": cf_b_to_u["death_cause"]} if cf_b_to_u else None),
            }
            # Also save full traces per seed
            (out_dir / f"{seed}_det_base.json").write_text(json.dumps(b_det1, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / f"{seed}_det_u08.json").write_text(json.dumps(u_det, ensure_ascii=False, indent=2), encoding="utf-8")
            with lock:
                results[seed] = rec
                print(
                    f"[{seed}] saved128(B={saved_b[seed]['floor']}/U={saved_u[seed]['floor']}) | "
                    f"origReplayB={b_orig['floor']} | detRepeatOK={det_identical} | "
                    f"det(B={b_det1['floor']} A{b_det1['act']+1} vs U={u_det['floor']} A{u_det['act']+1}) | "
                    f"div={div['type'] if div else 'NONE'}@{div['m_idx'] if div else '-'}",
                    flush=True,
                )
        finally:
            env.close()

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(s, idx)) for s, idx in SEEDS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ordered = [results[s] for s, _ in SEEDS]
    (out_dir / "summary.json").write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDONE 6-seed diagnostic in {time.time() - t0:.1f}s -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
