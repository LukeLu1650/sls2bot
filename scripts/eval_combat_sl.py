#!/usr/bin/env python3
"""Paired full-run evaluator: Pure Greedy vs Full-Combat SL + Out-of-Combat No-SL.

Protocol:
  - Uses the EXACT same frozen policy checkpoint (e.g. v4879 EMA) and EXACT same seed set
    (e.g. A10VAL0000..0127) across two paired arms:
      1. `greedy`: Pure single-pass greedy everywhere (0 SL, 0 search).
      2. `sl`:     EVERY combat from A1F1 Monster through A3 Boss 2 may save/load from
                   combat start via Sts2Headless native snapshot(0)/restore(0);
                   EVERY out-of-combat decision (map, card rewards, shop, rest, event,
                   relics, rewards) is executed ONCE greedily (strict No-SL).

  - Per-combat SL policy:
      * Candidate 0 is always greedy (temp=0.0).
      * Early stop: if a candidate wins the final Act-3 Boss 2, OR wins with zero net HP
        loss (end_hp >= start_hp) and zero net potion loss (end_pots >= start_pots), stop
        immediately without spending additional rollouts.
      * Otherwise run up to room-tiered extra trials (Monster / Elite / Boss budgets)
        with temperature-sampled trajectories from the exact combat-start snapshot(0).
      * Candidate selection:
          1. Victory strictly first (any win beats any loss).
          2. Among winning routes, maximize pre-registered Pareto-consistent resource score:
               score = end_hp + (potion_hp_frac * start_max_hp) * end_pots
             (default potion_hp_frac = 0.20, ~15 HP/potion, preventing spending a potion
             just to save a few HP while strictly respecting Pareto dominance).
          3. Tie-break: higher end_hp -> fewer steps -> prefer greedy.
      * Replay the chosen winning (or greedy) action sequence on the live run and continue
        No-SL with actual resulting HP, potions, and game state.

Logs rich per-run and per-combat decomposition:
  - Run-level: reach Act 3 per 100 runs, A10 wins per 100 runs, mean floor, Boss killed/reached,
    paired seed deltas, and extra wall-clock seconds per run.
  - Combat-level (overall, by room Monster/Elite/Boss, and by Act 1/2/3):
      * greedy_perfect_skip, sl_triggered, extra_rollouts
      * rescued_losses (greedy lost -> SL won)
      * improved_wins (greedy won -> SL improved resource score)
      * total & per-run HP saved and potions saved on improved wins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AD_PY = ROOT / "external" / "AlphaDefect-Public" / "game-v0.114.0" / "python"
if str(AD_PY) not in sys.path:
    sys.path.insert(0, str(AD_PY))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_v1140 import load_net_weights, load_seeds  # noqa: E402
from sts2env.env import EngineError, Sts2Env  # noqa: E402
from sts2env.protocol import DecisionKind, TokenKind, Zone, parse  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402


def trial_rng_seed(run_seed: str, act: int, floor: int, floor_combat_idx: int, trial_idx: int) -> int:
    """Deterministic per-(run_seed, combat_position, trial_idx) RNG seed.

    Decouples every combat and every trial so earlier trials/early-stops never shift
    the sampling stream of later combats.
    """
    h = hashlib.sha256(f"{run_seed}|a{act}|f{floor}|fc{floor_combat_idx}|t{trial_idx}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFF


def deck_size(obs) -> int:
    if obs.tok_ints.shape[0] == 0:
        return 0
    return int(((obs.tok_ints[:, 0] == TokenKind.CARD) & (obs.tok_ints[:, 2] == Zone.DECK)).sum())


def potions_held(obs) -> int:
    if obs.tok_ints.shape[0] == 0:
        return 0
    return int(((obs.tok_ints[:, 0] == TokenKind.POTION) & (obs.tok_ints[:, 2] == Zone.ALLY)).sum())


class BatchInferer:
    """Dynamic micro-batching CUDA/CPU inference server for MODE_RUN observations."""

    def __init__(self, net, device: str, max_batch: int = 16, wait_ms: float = 1.5, use_bf16: bool = True):
        self.net = net
        self.device = device
        self.max_batch = max(1, int(max_batch))
        self.wait_s = max(0.0, float(wait_ms) / 1000.0)
        self.use_bf16 = bool(use_bf16) and str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.batches = 0
        self.reqs = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def eval(self, obs) -> np.ndarray:
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
                rg = Ragged.from_obs(obs_list, np.full(n, MODE_RUN, dtype=np.int64))
                batch = collate_ragged(rg, attend=bool(self.net.cfg.macro_attend)).to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    logits, _ = self.net.forward_infer(batch)
                logits_f = logits.float().cpu()
                self.batches += 1
                self.reqs += n
                for i, (ob, ev, box) in enumerate(batch_items):
                    lg = logits_f[i, : ob.num_actions].double()
                    probs = torch.softmax(lg, dim=-1).numpy()
                    probs = np.maximum(probs, 0.0)
                    s = float(probs.sum())
                    probs = (probs / s) if (np.isfinite(s) and s > 0.0) else np.full(ob.num_actions, 1.0 / max(ob.num_actions, 1), dtype=np.float64)
                    box.append(probs)
                    ev.set()
            except Exception as err:
                for _ob, ev, box in batch_items:
                    box.extend([None, err])
                    ev.set()


def choose_action(obs, inferer: BatchInferer, temp: float, rng: np.random.Generator | None) -> int:
    if obs.num_actions <= 1:
        return 0
    probs = inferer.eval(obs)
    if temp <= 0.0 or rng is None:
        return int(np.argmax(probs))
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temp
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(obs.num_actions, p=p))


def reset_full_run(env: Sts2Env, character: str, asc: int, seed: str, stall_turns: int, labels: bool = False):
    req = {
        "op": "reset",
        "mode": "run",
        "character": character,
        "ascension": asc,
        "seed": seed,
        "snapshots": True,
        "route_summary": True,
        "labels": bool(labels),
        "loop_cut": True,
        "stall_turns": stall_turns,
    }
    env._proc.send(req)
    return parse(env._proc.recv())


def rollout_single_combat(
    env: Sts2Env,
    obs,
    inferer: BatchInferer,
    temp: float,
    rng: np.random.Generator | None,
    start_combats_won: int,
    max_steps_fight: int,
) -> tuple[dict, object]:
    """Play from combat start until this single combat ends (win, death, stuck, or next back-to-back combat)."""
    actions: list[int] = []
    steps = 0
    while (
        not obs.done
        and not obs.stuck
        and obs.in_combat
        and obs.combats_won == start_combats_won
        and obs.num_actions > 0
        and steps < max_steps_fight
    ):
        a = choose_action(obs, inferer, temp=temp, rng=rng)
        actions.append(a)
        obs = env.step(a)
        steps += 1

    # Determine combat outcome
    if obs.stuck or (steps >= max_steps_fight and obs.in_combat and obs.combats_won == start_combats_won and not obs.done):
        won = False
        stuck = True
    elif obs.done:
        won = bool(obs.victory)
        stuck = False
    else:
        # Left combat alive OR combats_won incremented (e.g. A3 Boss 1 -> Boss 2)
        won = True
        stuck = False

    res = {
        "won": won,
        "stuck": stuck,
        "run_victory": bool(obs.done and obs.victory),
        "end_hp": int(obs.hp),
        "end_max_hp": int(obs.max_hp),
        "end_pots": potions_held(obs) if won else 0,
        "steps": len(actions),
        "actions": actions,
    }
    return res, obs


def rank_key(cand: dict, potion_hp_val: float, is_greedy: bool) -> tuple:
    won = 1 if cand["won"] else 0
    score = (cand["end_hp"] + potion_hp_val * cand["end_pots"]) if cand["won"] else 0.0
    hp_tie = cand["end_hp"] if cand["won"] else 0
    step_tie = -cand["steps"] if cand["won"] else cand["steps"]
    return (won, round(score, 6), hp_tie, step_tie, 1 if is_greedy else 0)


def is_perfect_win(cand: dict, start_hp: int, start_pots: int) -> bool:
    if not cand["won"]:
        return False
    if cand["run_victory"]:
        return True
    return cand["end_hp"] >= start_hp and cand["end_pots"] >= start_pots


def run_one_seed(
    env: Sts2Env,
    seed: str,
    seed_idx: int,
    inferer: BatchInferer,
    sl_enabled: bool,
    cfg: argparse.Namespace,
) -> dict:
    t0 = time.time()
    log_macros = bool(getattr(cfg, "log_macros", False))
    obs = reset_full_run(env, cfg.character, cfg.asc, seed, cfg.stall_turns, labels=log_macros)
    temps = (0.4, 0.7, 1.0, 0.5, 0.85, 1.15)

    budgets = {
        "Monster": (cfg.monster_trials, cfg.monster_timeout),
        "Elite": (cfg.elite_trials, cfg.elite_timeout),
        "Boss": (cfg.boss_trials, cfg.boss_timeout),
    }

    combats_log: list[dict] = []
    macros_log: list[dict] = []
    floor_combat_counts: dict[tuple[int, int], int] = defaultdict(int)
    had_anomaly_timeout = False
    run_steps = 0
    stuck = False

    while not obs.done and not obs.stuck and obs.num_actions > 0 and run_steps < cfg.max_steps_run:
        if not obs.in_combat:
            # Out-of-combat: STRICT No-SL, single greedy forward pass
            if log_macros:
                probs = inferer.eval(obs) if obs.num_actions > 1 else np.array([1.0], dtype=np.float64)
                a = int(np.argmax(probs)) if obs.num_actions > 1 else 0
                labels = list(obs.labels) if obs.labels else [f"a{i}" for i in range(obs.num_actions)]
                order = np.argsort(-probs)
                top2 = int(order[1]) if len(order) > 1 else None
                dec_name = DecisionKind(int(obs.decision)).name if int(obs.decision) in DecisionKind._value2member_map_ else str(obs.decision)
                macros_log.append({
                    "m_idx": len(macros_log),
                    "act": int(obs.act),
                    "floor": int(obs.floor),
                    "decision": dec_name,
                    "hp": int(obs.hp),
                    "max_hp": int(obs.max_hp),
                    "pots": potions_held(obs),
                    "deck": deck_size(obs),
                    "n_act": int(obs.num_actions),
                    "action": a,
                    "chosen_label": labels[a] if a < len(labels) else f"a{a}",
                    "chosen_prob": round(float(probs[a]), 4),
                    "top2_label": labels[top2] if (top2 is not None and top2 < len(labels)) else None,
                    "top2_prob": round(float(probs[top2]), 4) if top2 is not None else None,
                    "labels": labels,
                })
            else:
                a = choose_action(obs, inferer, temp=0.0, rng=None)
            obs = env.step(a)
            run_steps += 1
            continue

        # Combat start!
        cinfo = (obs.extra or {}).get("combat") or {}
        room = str(cinfo.get("room") or "Monster")
        if room not in ("Monster", "Elite", "Boss"):
            room = "Boss" if int(obs.floor) in (17, 33, 48, 49) else "Monster"
        encounter = str(cinfo.get("encounter") or f"UNK_A{obs.act+1}_F{obs.floor}")
        act = int(cinfo.get("act", obs.act))
        floor = int(obs.floor)
        fc_idx = floor_combat_counts[(act, floor)]
        floor_combat_counts[(act, floor)] += 1
        start_hp = int(obs.hp)
        start_max_hp = int(obs.max_hp)
        start_pots = potions_held(obs)
        start_cw = int(obs.combats_won)
        potion_hp_val = cfg.potion_hp_frac * max(start_max_hp, 1)

        if not sl_enabled:
            res, obs = rollout_single_combat(
                env, obs, inferer, temp=0.0, rng=None,
                start_combats_won=start_cw, max_steps_fight=cfg.max_steps_fight,
            )
            run_steps += res["steps"]
            if res["stuck"]:
                stuck = True
            combats_log.append({
                "act": act,
                "floor": floor,
                "fc_idx": fc_idx,
                "room": room,
                "encounter": encounter,
                "start_hp": start_hp,
                "start_max_hp": start_max_hp,
                "start_pots": start_pots,
                "greedy_won": res["won"],
                "greedy_hp": res["end_hp"],
                "greedy_pots": res["end_pots"],
                "final_won": res["won"],
                "final_hp": res["end_hp"],
                "final_pots": res["end_pots"],
                "sl_triggered": False,
                "perfect_skip": False,
                "extra_trials": 0,
                "stop_reason": "no_sl",
                "rescued": False,
                "improved_win": False,
                "hp_saved": 0,
                "pots_saved": 0,
                "sl_s": 0.0,
            })
            if stuck:
                break
            continue

        # Full-Combat SL enabled: snapshot combat start in slot 0
        t_c0 = time.time()
        env.snapshot(0)
        greedy_res, obs_after_greedy = rollout_single_combat(
            env, obs, inferer, temp=0.0, rng=None,
            start_combats_won=start_cw, max_steps_fight=cfg.max_steps_fight,
        )
        best_res = greedy_res
        best_idx = 0
        best_key = rank_key(greedy_res, potion_hp_val, is_greedy=True)
        last_idx = 0
        last_obs = obs_after_greedy

        max_extra, fight_timeout = budgets.get(room, (cfg.monster_trials, cfg.monster_timeout))
        perfect_skip = is_perfect_win(greedy_res, start_hp, start_pots)
        extra_used = 0
        sl_triggered = (not perfect_skip) and (max_extra > 0)
        if perfect_skip:
            stop_reason = "greedy_perfect"
        elif max_extra <= 0:
            stop_reason = "no_sl"
        else:
            stop_reason = "budget_exhausted"

        if sl_triggered:
            for ti in range(1, max_extra + 1):
                if fight_timeout > 0.0 and (time.time() - t_c0 >= fight_timeout):
                    stop_reason = "anomaly_timeout"
                    had_anomaly_timeout = True
                    break
                obs_r = env.restore(0)
                t_cur = temps[(ti - 1) % len(temps)]
                # Decoupled per-(seed, act, floor, floor_combat_idx, trial_idx) RNG
                rng_t = np.random.default_rng(trial_rng_seed(seed, act, floor, fc_idx, ti))
                cand_res, obs_end = rollout_single_combat(
                    env, obs_r, inferer, temp=t_cur, rng=rng_t,
                    start_combats_won=start_cw, max_steps_fight=cfg.max_steps_fight,
                )
                extra_used += 1
                last_idx = ti
                last_obs = obs_end
                k = rank_key(cand_res, potion_hp_val, is_greedy=False)
                if k > best_key:
                    best_key = k
                    best_res = cand_res
                    best_idx = ti
                    if is_perfect_win(cand_res, start_hp, start_pots):
                        stop_reason = "sl_perfect"
                        break

        # Ensure live engine is at exact post-combat state of `best_res`
        if best_idx == last_idx:
            obs = last_obs
        else:
            obs_re = env.restore(0)
            for a in best_res["actions"]:
                obs_re = env.step(int(a))
            obs = obs_re

        sl_s = time.time() - t_c0
        run_steps += best_res["steps"]
        if best_res["stuck"]:
            stuck = True

        rescued = (not greedy_res["won"]) and best_res["won"]
        improved_win = greedy_res["won"] and best_res["won"] and (best_idx != 0)
        hp_saved = (best_res["end_hp"] - greedy_res["end_hp"]) if improved_win else 0
        pots_saved = (best_res["end_pots"] - greedy_res["end_pots"]) if improved_win else 0

        combats_log.append({
            "act": act,
            "floor": floor,
            "fc_idx": fc_idx,
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
            "sl_triggered": sl_triggered,
            "perfect_skip": perfect_skip,
            "extra_trials": extra_used,
            "stop_reason": stop_reason,
            "rescued": rescued,
            "improved_win": improved_win,
            "hp_saved": hp_saved,
            "pots_saved": pots_saved,
            "sl_s": round(sl_s, 3),
        })
        if stuck:
            break

    wall_s = time.time() - t0
    if obs.stuck or (not obs.done and run_steps >= cfg.max_steps_run):
        stuck = True

    # Derive Boss outcomes cleanly from combats_log + final run state
    a1_b = [c for c in combats_log if c["room"] == "Boss" and c["act"] == 0]
    a2_b = [c for c in combats_log if c["room"] == "Boss" and c["act"] == 1]
    a3_b = [c for c in combats_log if c["room"] == "Boss" and c["act"] == 2]

    rec = {
        "seed": seed,
        "mode": "sl" if sl_enabled else "greedy",
        "win": bool(obs.done and obs.victory),
        "stuck": bool(stuck),
        "had_anomaly_timeout": bool(had_anomaly_timeout),
        "act": int(obs.act),
        "floor": int(obs.floor),
        "end_hp": int(obs.hp),
        "combats_won": int(obs.combats_won),
        "run_steps": run_steps,
        "wall_s": round(wall_s, 2),
        "boss1": [int(any(c["final_won"] for c in a1_b)), int(len(a1_b) > 0)],
        "boss2": [int(any(c["final_won"] for c in a2_b)), int(len(a2_b) > 0)],
        "boss3_1": [int(len(a3_b) >= 1 and a3_b[0]["final_won"]), int(len(a3_b) >= 1)],
        "boss3_2": [int(len(a3_b) >= 2 and a3_b[1]["final_won"]), int(len(a3_b) >= 2)],
        "combats": combats_log,
    }
    if log_macros:
        rec["macros"] = macros_log
    return rec


def summarize_arm(rows: list[dict], wall_total_s: float | None) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    wins = sum(1 for r in rows if r["win"])
    stuck = sum(1 for r in rows if r["stuck"])
    floors = [r["floor"] for r in rows]
    reach_a2 = sum(1 for r in rows if r["act"] >= 1 or r["win"])
    reach_a3 = sum(1 for r in rows if r["act"] >= 2 or r["win"])
    b1 = [sum(r["boss1"][0] for r in rows), sum(r["boss1"][1] for r in rows)]
    b2 = [sum(r["boss2"][0] for r in rows), sum(r["boss2"][1] for r in rows)]
    b31 = [sum(r["boss3_1"][0] for r in rows), sum(r["boss3_1"][1] for r in rows)]
    b32 = [sum(r["boss3_2"][0] for r in rows), sum(r["boss3_2"][1] for r in rows)]

    all_c = [c for r in rows for c in r.get("combats", [])]
    stop_counts = dict(Counter(c.get("stop_reason", "unknown") for c in all_c))
    anomaly_runs = sum(1 for r in rows if r.get("had_anomaly_timeout"))

    def _cstats(arr: list[dict]) -> dict:
        return {
            "combats": len(arr),
            "perfect_skip": sum(1 for c in arr if c["perfect_skip"]),
            "sl_triggered": sum(1 for c in arr if c["sl_triggered"]),
            "extra_trials": sum(c["extra_trials"] for c in arr),
            "stop_reasons": dict(Counter(c.get("stop_reason", "unknown") for c in arr)),
            "rescued_losses": sum(1 for c in arr if c["rescued"]),
            "improved_wins": sum(1 for c in arr if c["improved_win"]),
            "hp_saved_on_wins": sum(c["hp_saved"] for c in arr),
            "pots_saved_on_wins": sum(c["pots_saved"] for c in arr),
        }

    by_room = {rm: _cstats([c for c in all_c if c["room"] == rm]) for rm in ("Monster", "Elite", "Boss")}
    by_act = {f"A{a+1}": _cstats([c for c in all_c if c["act"] == a]) for a in (0, 1, 2)}
    tot_c = _cstats(all_c)
    mean_run_wall = sum(r["wall_s"] for r in rows) / n

    return {
        "n": n,
        "wins": wins,
        "wins_per_100": round(wins / n * 100.0, 2),
        "stuck": stuck,
        "anomaly_timeout_runs": anomaly_runs,
        "stop_reasons": stop_counts,
        "mean_floor": round(statistics.mean(floors), 2),
        "median_floor": statistics.median(floors),
        "max_floor": max(floors),
        "reach_act2": reach_a2,
        "reach_act2_per_100": round(reach_a2 / n * 100.0, 2),
        "reach_act3": reach_a3,
        "reach_act3_per_100": round(reach_a3 / n * 100.0, 2),
        "boss1": b1,
        "boss2": b2,
        "boss3_1": b31,
        "boss3_2": b32,
        "mean_engine_s_per_run": round(mean_run_wall, 2),
        "wall_total_s": round(wall_total_s, 1) if wall_total_s is not None else None,
        "wall_s_per_run": round(wall_total_s / n, 2) if wall_total_s is not None else None,
        "combat_sl": {
            "total": tot_c,
            "per_run_hp_saved": round(tot_c["hp_saved_on_wins"] / n, 2),
            "per_run_pots_saved": round(tot_c["pots_saved_on_wins"] / n, 3),
            "by_room": by_room,
            "by_act": by_act,
        },
    }


def run_arm(
    arm_name: str,
    sl_enabled: bool,
    seeds: list[str],
    inferer: BatchInferer,
    out_path: Path,
    cfg: argparse.Namespace,
) -> tuple[dict, dict[str, dict]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        for ln in out_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[r["seed"]] = r

    todo = [(idx, s) for idx, s in enumerate(seeds) if s not in done]
    print(f"\n=== Arm [{arm_name}] | total={len(seeds)}, cached={len(done)}, to_run={len(todo)} | workers={cfg.workers} ===", flush=True)

    t_start = time.time()
    if todo:
        work_q: queue.Queue = queue.Queue()
        for item in todo:
            work_q.put(item)
        write_lock = threading.Lock()
        completed = [0]
        fh = out_path.open("a", encoding="utf-8")

        def worker_fn(wid: int) -> None:
            env = Sts2Env()
            try:
                while True:
                    try:
                        s_idx, s = work_q.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        rec = run_one_seed(env, s, s_idx, inferer, sl_enabled=sl_enabled, cfg=cfg)
                    except EngineError as err:
                        rec = {
                            "seed": s, "mode": arm_name, "win": False, "stuck": True,
                            "act": 0, "floor": 0, "end_hp": 0, "combats_won": 0,
                            "run_steps": 0, "wall_s": 0.0, "error": str(err)[:300],
                            "boss1": [0, 0], "boss2": [0, 0], "boss3_1": [0, 0], "boss3_2": [0, 0],
                            "combats": [],
                        }
                    with write_lock:
                        done[s] = rec
                        completed[0] += 1
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        if completed[0] <= 4 or completed[0] % 16 == 0 or rec["win"] or completed[0] == len(todo):
                            el = time.time() - t_start
                            resc = sum(1 for c in rec["combats"] if c["rescued"])
                            imp = sum(1 for c in rec["combats"] if c["improved_win"])
                            hps = sum(c["hp_saved"] for c in rec["combats"])
                            ps = sum(c["pots_saved"] for c in rec["combats"])
                            print(
                                f"[{arm_name} {completed[0]:>3d}/{len(todo)} w{wid}] {s} -> fl={rec['floor']:2d} A{rec['act']+1} win={int(rec['win'])} "
                                f"resc={resc} imp={imp}(+{hps}hp,+{ps}pot) {rec['wall_s']:.1f}s | elapsed={el:.0f}s",
                                flush=True,
                            )
            finally:
                env.close()

        try:
            n_w = max(1, min(cfg.workers, len(todo)))
            threads = [threading.Thread(target=worker_fn, args=(w,), daemon=True) for w in range(n_w)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            fh.close()

    wall_total = time.time() - t_start if todo else None
    ordered = [done[s] for s in seeds if s in done]
    sm = summarize_arm(ordered, wall_total)
    return sm, done


def _fmt_ratio(pair: list[int]) -> str:
    k, r = pair
    return f"{k}/{r} ({k/r*100:4.1f}%)" if r > 0 else "0/0 (-)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired Full-Combat SL + Out-of-Combat No-SL evaluator.")
    ap.add_argument("--ckpt", required=True, help="Frozen checkpoint (.pt)")
    ap.add_argument("--weights", choices=("ema", "raw"), default="ema")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--mode", choices=("both", "greedy", "sl"), default="both")
    ap.add_argument("--character", default="NECROBINDER")
    ap.add_argument("--asc", type=int, default=10)
    ap.add_argument("--runs", type=int, default=128)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-prefix", default="A10VAL")
    ap.add_argument("--seeds-file", default=None)
    ap.add_argument("--seeds", default=None, help="Comma-separated explicit seed list (overrides --runs/--seed-prefix)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-batch", type=int, default=0, help="Override BatchInferer max_batch (0 = workers)")
    ap.add_argument("--no-bf16", action="store_true", help="Disable CUDA BF16 autocast (run FP32 for strict determinism)")
    ap.add_argument("--log-macros", action="store_true", help="Enable engine labels and log every out-of-combat decision")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--monster-trials", type=int, default=4, help="Extra SL trials per Monster fight")
    ap.add_argument("--elite-trials", type=int, default=12, help="Extra SL trials per Elite fight")
    ap.add_argument("--boss-trials", type=int, default=20, help="Extra SL trials per Boss fight")
    ap.add_argument("--monster-timeout", type=float, default=60.0, help="Anomaly wall-clock guard (s) per Monster fight (0=off)")
    ap.add_argument("--elite-timeout", type=float, default=120.0, help="Anomaly wall-clock guard (s) per Elite fight (0=off)")
    ap.add_argument("--boss-timeout", type=float, default=180.0, help="Anomaly wall-clock guard (s) per Boss fight (0=off)")
    ap.add_argument("--potion-hp-frac", type=float, default=0.20, help="Resource score weight: 1 potion = potion_hp_frac * max_hp")
    ap.add_argument("--stall-turns", type=int, default=20)
    ap.add_argument("--max-steps-run", type=int, default=6000)
    ap.add_argument("--max-steps-fight", type=int, default=400)
    a = ap.parse_args()

    torch.set_num_threads(2)
    if a.no_bf16 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    net, blob, src_key = load_net_weights(Path(a.ckpt), a.weights)
    net = net.to(a.device)
    net.fast = str(a.device).startswith("cuda")
    net.infer_ready()
    if a.seeds:
        seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]
    else:
        seeds = load_seeds(a.runs, a.seed_prefix, a.seeds_file, seed_start=a.seed_start)
    print(
        f"Loaded {a.ckpt} (v{blob['version']}, {a.weights}/{src_key}) on {a.device} (bf16={not a.no_bf16}) | "
        f"runs={len(seeds)} ({seeds[0]}..) | "
        f"budgets M={a.monster_trials} E={a.elite_trials} B={a.boss_trials} | "
        f"guards={a.monster_timeout}/{a.elite_timeout}/{a.boss_timeout}s | potion_hp_frac={a.potion_hp_frac}",
        flush=True,
    )

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mb = a.max_batch if a.max_batch > 0 else max(1, a.workers)
    inferer = BatchInferer(net, a.device, max_batch=mb, wait_ms=1.5, use_bf16=not a.no_bf16)

    results: dict[str, dict] = {}
    arm_rows: dict[str, dict[str, dict]] = {}
    try:
        if a.mode in ("both", "greedy"):
            sm_g, rows_g = run_arm("greedy", False, seeds, inferer, out_dir / "greedy_runs.jsonl", a)
            results["greedy"] = sm_g
            arm_rows["greedy"] = rows_g
        if a.mode in ("both", "sl"):
            sm_s, rows_s = run_arm("sl", True, seeds, inferer, out_dir / "sl_runs.jsonl", a)
            results["sl"] = sm_s
            arm_rows["sl"] = rows_s
    finally:
        inferer.close()

    paired = None
    if "greedy" in arm_rows and "sl" in arm_rows:
        common = [s for s in seeds if s in arm_rows["greedy"] and s in arm_rows["sl"]]
        rg, rs = arm_rows["greedy"], arm_rows["sl"]
        d_fl = [rs[s]["floor"] - rg[s]["floor"] for s in common]
        a3_gain = sum(1 for s in common if (rs[s]["act"] >= 2 or rs[s]["win"]) and not (rg[s]["act"] >= 2 or rg[s]["win"]))
        a3_loss = sum(1 for s in common if (rg[s]["act"] >= 2 or rg[s]["win"]) and not (rs[s]["act"] >= 2 or rs[s]["win"]))
        win_gain = sum(1 for s in common if rs[s]["win"] and not rg[s]["win"])
        win_loss = sum(1 for s in common if rg[s]["win"] and not rs[s]["win"])
        paired = {
            "n": len(common),
            "mean_d_floor": round(sum(d_fl) / max(len(d_fl), 1), 2),
            "floor_better": sum(1 for x in d_fl if x > 0),
            "floor_worse": sum(1 for x in d_fl if x < 0),
            "floor_tie": sum(1 for x in d_fl if x == 0),
            "act3_gained": a3_gain,
            "act3_lost": a3_loss,
            "act3_net": a3_gain - a3_loss,
            "win_gained": win_gain,
            "win_lost": win_loss,
            "win_net": win_gain - win_loss,
            "extra_engine_s_per_run": round(results["sl"]["mean_engine_s_per_run"] - results["greedy"]["mean_engine_s_per_run"], 2),
            "extra_wall_s_per_run": (
                round(results["sl"]["wall_s_per_run"] - results["greedy"]["wall_s_per_run"], 2)
                if results["sl"]["wall_s_per_run"] is not None and results["greedy"]["wall_s_per_run"] is not None
                else None
            ),
        }

    summary = {
        "config": vars(a),
        "version": int(blob["version"]),
        "total_steps": int(blob.get("total_steps", 0)),
        "arms": results,
        "paired": paired,
    }
    sp = out_dir / "summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===", flush=True)
    for k, sm in results.items():
        print(
            f"[{k:6s}] n={sm['n']} | fl={sm['mean_floor']:.2f} (med {sm['median_floor']}, max {sm['max_floor']}) | "
            f"A2={sm['reach_act2']}/{sm['n']} ({sm['reach_act2_per_100']:.1f}/100) | "
            f"A3={sm['reach_act3']}/{sm['n']} ({sm['reach_act3_per_100']:.1f}/100) | "
            f"Win={sm['wins']}/{sm['n']} ({sm['wins_per_100']:.1f}/100) | "
            f"B1={_fmt_ratio(sm['boss1'])} B2={_fmt_ratio(sm['boss2'])} B3.1={_fmt_ratio(sm['boss3_1'])} B3.2={_fmt_ratio(sm['boss3_2'])} | "
            f"eng_s/run={sm['mean_engine_s_per_run']:.2f}s wall_s/run={sm['wall_s_per_run']}s | stuck={sm['stuck']}",
            flush=True,
        )
    if "sl" in results:
        c = results["sl"]["combat_sl"]
        t = c["total"]
        print(
            f"\n[SL Decomposition] total_combats={t['combats']} | perfect_skip={t['perfect_skip']} | "
            f"sl_triggered={t['sl_triggered']} | extra_trials={t['extra_trials']} | "
            f"anomaly_timeout_runs={results['sl']['anomaly_timeout_runs']} | stop_reasons={results['sl']['stop_reasons']} | "
            f"rescued_losses={t['rescued_losses']} | improved_wins={t['improved_wins']} | "
            f"hp_saved={t['hp_saved_on_wins']} ({c['per_run_hp_saved']}/run) | "
            f"pots_saved={t['pots_saved_on_wins']} ({c['per_run_pots_saved']}/run)",
            flush=True,
        )
        for grp_name, grp in (("By Room", c["by_room"]), ("By Act", c["by_act"])):
            print(f"  {grp_name}:")
            for label, d in grp.items():
                print(
                    f"    {label:8s}: combats={d['combats']:4d} skip={d['perfect_skip']:4d} trig={d['sl_triggered']:4d} "
                    f"resc={d['rescued_losses']:3d} imp={d['improved_wins']:4d} (+{d['hp_saved_on_wins']}hp, +{d['pots_saved_on_wins']}pot)",
                    flush=True,
                )
    if paired:
        print(f"\n[Paired n={paired['n']}] {json.dumps(paired, ensure_ascii=False)}", flush=True)
    print(f"Saved -> {sp}", flush=True)


if __name__ == "__main__":
    main()
