"""Pure from-scratch PPO v2 trainer for Necrobinder (A0 -> A10, zero human data).

Key upgrades over train_aligned.py:
  * Fully trainable lightweight Entity-Transformer (no frozen random 5-layer / 5024-FFN backbone).
  * Complete player-visible entity tokens (relics, potions, deck summary, 4-layer map lookahead,
    hand keywords/enchantments/afflictions/previews) + 2-layer MLP Value Head with auxiliary heads.
  * Room-boundary discounting (gamma_step=0.9995 within room, gamma_room=0.99 across rooms) +
    separate combat/strategic GAE lambdas & advantage normalization.
  * Elite kill bonus (+0.35) and smooth Boss damage-progress shaping (+0.40 * dmg_frac).
  * Large minibatch (default 128), cosine LR schedule, temperature & entropy annealing,
    focus_rollouts >= 3 with fair pool sampling, and automatic best-checkpoint.pt selection
    on a larger fixed development panel (default 24 seeds).
  * Optional P2 CombatSolver offline BC regularization (--solver-demos / --bc-coef, zero human data).
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import math
import os
from pathlib import Path
import statistics
from threading import Lock

import numpy as np
import torch

from beta_build import CHECKOUT, STEAM_HOST, STEAM_MANIFEST, sha, verify_host
from beta_env import is_combat_decision, run_episode
from beta_eval import wilson, write_json
from ppo_v2_policy import InferenceBatcher, ModelConfig, OBSERVATION_VERSION, PRESETS, PPOv2Policy
from solver_bc import load_bc_demos
from strategy_updates import training_rng
from train_curriculum import Attempt, PRIORITIES, choose_case, read_json, retain_cases
from train_joint import show_hp
from train_strategy import panel, positive_int


TERMINAL = {"defeat", "victory"}
EXPERIMENT = "necro-ppo-v2"


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return number
COMPATIBLE_STEAM_HOSTS = {
    "3f94256154ba293f06af2c195e8958a3f1de1da7cb26fbc4a74be1f06829fd77",
    "5c0acfd4f8027335e180d1496ac0ab4f0cf21d4f0417f6169109e96bdeb92066",
    "330710141478b0dfee06321e585934e623ffd024895bd3a68ad110bedf19cfbb",
}


def start_command(seed: str, progress: dict, ascension: int = 0) -> dict:
    return {
        "cmd": "start_steam_run",
        "progress": progress,
        "run": {
            "seed": seed,
            "character": "Necrobinder",
            "ascension": ascension,
            "game_mode": "standard",
            "is_custom_game": False,
            "modifiers": [],
        },
    }


def schedule_value(start: float, end: float, round_idx: int, total_rounds: int) -> float:
    if total_rounds <= 1:
        return end
    progress = min(1.0, max(0.0, (round_idx - 1) / (total_rounds - 1)))
    return start + (end - start) * progress


def cosine_lr(base_lr: float, round_idx: int, total_rounds: int, warmup_rounds: int = 4) -> float:
    if warmup_rounds > 1 and round_idx <= warmup_rounds:
        frac = 0.25 + 0.75 * (round_idx - 1) / (warmup_rounds - 1)
        return base_lr * frac
    denom = max(1, total_rounds - warmup_rounds)
    progress = min(1.0, max(0.0, (round_idx - warmup_rounds) / denom))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


class RunTraceV2:
    """Track per-fight outcomes, Elite kills, and smooth Boss HP damage progress."""

    def __init__(self):
        self.fights: dict[tuple[int, int], dict] = {}
        self.active: tuple[int, int] | None = None
        self.confirmed = 0
        self.steps = []
        self.last_room: tuple[int, int] | None = None
        self.rooms: list[tuple[tuple[int, int], int]] = []
        self.act2_entry: dict | None = None

    @staticmethod
    def _build_entry_snapshot(state: dict, room: tuple[int, int]) -> dict:
        player = state.get("player") or {}
        deck_cards = player.get("deck") or []
        strikes = defends = curses = upgraded = 0
        deck_names: list[str] = []
        for card in deck_cards:
            if not isinstance(card, dict):
                deck_names.append(str(card))
                continue
            cid = str(card.get("id") or card.get("name") or "unknown")
            cname = str(card.get("name") or cid)
            ctype = str(card.get("type") or "")
            up = bool(card.get("upgraded"))
            if up:
                upgraded += 1
            cid_u = cid.upper()
            if "STRIKE" in cid_u or cname.lower() == "strike":
                strikes += 1
            elif "DEFEND" in cid_u or cname.lower() == "defend":
                defends += 1
            if ctype in {"Curse", "Status"}:
                curses += 1
            deck_names.append(f"{cname}+" if up and not cname.endswith("+") else cname)
        relics = [
            str(r.get("name") or r.get("id") or "unknown") if isinstance(r, dict) else str(r)
            for r in (player.get("relics") or [])
        ]
        potions = [
            str(p.get("name") or p.get("id") or "unknown")
            for p in (player.get("potions") or [])
            if isinstance(p, dict) and (p.get("name") or p.get("id"))
        ]
        return {
            "floor": room[1],
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "gold": player.get("gold"),
            "deck_size": len(deck_cards),
            "upgraded": upgraded,
            "strikes": strikes,
            "defends": defends,
            "basics": strikes + defends,
            "curses": curses,
            "relics": relics,
            "potions": potions,
            "deck": deck_names,
        }

    def observe(self, state: dict):
        self.confirmed = len(self.steps)
        context = state.get("context", {})
        room = (context.get("act", state.get("act")), context.get("floor", state.get("floor")))
        hp = state.get("player", {}).get("hp")
        if None in room or hp is None:
            return
        if room[0] == 2 and self.act2_entry is None:
            self.act2_entry = self._build_entry_snapshot(state, room)
        if room != self.last_room:
            self.rooms.append((room, self.confirmed))
            self.last_room = room
        if self.active and (room != self.active or not is_combat_decision(state, self.active)):
            fight = self.fights[self.active]
            fight["hp_out"] = hp if room[0] == self.active[0] else fight["last_hp"]
            if fight["hp_out"] > 0 and not fight["escaped"]:
                fight["kill_order"].extend(sorted(fight["alive"].elements()))
                fight["enemy_min_hp"] = 0
            fight["finished_at"] = self.confirmed
            self.active = None
        if state["decision"] != "combat_play":
            return
        enemies_list = state.get("enemies") or []
        enemies = Counter(e["name"] for e in enemies_list if e["hp"] > 0)
        total_enemy_hp = sum(max(0, e["hp"]) for e in enemies_list)
        total_enemy_max_hp = sum(max(1, e.get("max_hp") or e["hp"]) for e in enemies_list)
        self.active = room
        fight = self.fights.setdefault(
            room,
            {
                "act": room[0],
                "floor": room[1],
                "room": context.get("room_type"),
                "hp_in": hp,
                "max_hp": state["player"]["max_hp"],
                "hp_out": None,
                "alive": enemies,
                "last_hp": hp,
                "enemy_initial_hp": max(1, total_enemy_max_hp),
                "enemy_min_hp": total_enemy_hp,
                "kill_order": [],
                "potion_uses": 0,
                "escaped": False,
            },
        )
        fight["kill_order"].extend(sorted((fight["alive"] - enemies).elements()))
        fight["enemy_min_hp"] = min(fight["enemy_min_hp"], total_enemy_hp)
        fight.update(alive=enemies, last_hp=hp)

    def finalize_defeat(self):
        if self.active and self.active in self.fights:
            fight = self.fights[self.active]
            if fight["hp_out"] is None:
                fight["hp_out"] = 0
                fight["finished_at"] = self.confirmed
            self.active = None

    def record(self) -> list[dict]:
        return [
            {k: v for k, v in fight.items() if k not in {"alive", "last_hp", "finished_at", "escaped"}}
            for fight in self.fights.values()
        ]


def fight_return(fight: dict) -> float:
    """Balanced fight reward: survival + HP preservation + act-scaled Elite kill bonus & Boss progress."""
    survived = bool(fight["hp_out"] and fight["hp_out"] > 0)
    hp_delta = (fight["hp_out"] - fight["hp_in"]) / max(1, fight["max_hp"])
    score = 0.15 * survived + 0.45 * hp_delta
    act = int(fight.get("act") or 1)
    room = fight.get("room")
    if room == "Elite" and survived:
        score += {1: 0.35, 2: 0.25, 3: 0.20}.get(act, 0.35)
    if room == "Boss":
        if survived:
            dmg_frac = 1.0
        else:
            init_hp = max(1, fight.get("enemy_initial_hp") or 1)
            min_hp = max(0, fight.get("enemy_min_hp") or 0)
            dmg_frac = max(0.0, min(1.0, 1.0 - min_hp / init_hp))
        boss_weight = {1: 0.40, 2: 0.80, 3: 1.00}.get(act, 0.40)
        score += boss_weight * dmg_frac
    return float(score)


def compute_ppo_returns(
    trace: RunTraceV2,
    row: dict,
    opening: tuple[int, int] | None = None,
    *,
    gamma_step: float = 0.9995,
    gamma_room: float = 0.99,
    lam_combat: float = 0.95,
    lam_strategic: float = 0.98,
) -> list[dict]:
    """Compute room-boundary-discounted GAE advantages, returns, and auxiliary self-play targets.

    Both gamma_room and GAE lambda are applied at room transitions rather than every individual
    card play, preventing exponential credit collapse across 200-350 step runs when V(s) starts near 0.
    """
    samples = trace.steps[:trace.confirmed]
    if not samples:
        return []
    if opening and (opening not in trace.fights or trace.fights[opening]["hp_out"] is None):
        return []

    rewards = np.zeros(len(samples), dtype=np.float32)
    fight_hp_frac: dict[tuple[int, int], float] = {}
    for room, fight in trace.fights.items():
        if fight["hp_out"] is not None:
            fight_hp_frac[room] = max(0.0, fight["hp_out"] / max(1, fight["max_hp"]))
        end = fight.get("finished_at", 0)
        if 0 < end <= len(samples) and fight["hp_out"] is not None:
            rewards[end - 1] += fight_return(fight)

    previous = None
    for room, end in trace.rooms:
        if previous and 0 < end <= len(samples):
            if room[0] > previous[0]:
                rewards[end - 1] += 1.5
            else:
                rewards[end - 1] += {1: 0.01, 2: 0.06, 3: 0.10}.get(previous[0], 0.01)
        previous = room

    terminal = row["status"] in TERMINAL
    if terminal:
        rewards[-1] += 3.0 if row["victory"] else -1.0

    act = row.get("act") or 1
    floor = row.get("floor") or 0
    final_floor = min(51, (act - 1) * 17 + min(floor, 17)) if terminal else None

    next_value = (
        0.0
        if terminal
        else (trace.steps[len(samples)].value if len(trace.steps) > len(samples) else samples[-1].value)
    )
    next_room = samples[-1].room
    advantage = 0.0
    out: list[dict] = []

    for idx in range(len(samples) - 1, -1, -1):
        step = samples[idx]
        is_last = idx == len(samples) - 1
        crossed_room = (not is_last) and (step.room != next_room)
        gamma_t = gamma_room if crossed_room else gamma_step
        # Apply GAE lambda at room boundaries (1.0 within a room) so multi-card turns preserve credit
        lam_t = (lam_combat if step.combat else lam_strategic) if crossed_room else 1.0

        delta = float(rewards[idx]) + gamma_t * next_value - step.value
        advantage = delta + gamma_t * lam_t * advantage
        target = advantage + step.value

        if opening is None or (step.combat and step.room == opening):
            out.append({
                "step": step,
                "advantage": float(advantage),
                "target": float(target),
                "final_floor": final_floor,
                "combat_end_hp": fight_hp_frac.get(step.room) if step.combat else None,
            })

        next_value = step.value
        next_room = step.room

    out.reverse()
    return out


def focus_advantages(group: list[tuple[dict, list[dict]]]) -> list[dict]:
    """Leave-one-out relative baseline across repeated rollouts of the same focus opening."""
    completed = [(row["focus"]["return_"], items) for row, items in group if "return_" in row.get("focus", {})]
    if not completed:
        return []
    total = sum(score for score, _ in completed)
    merged: list[dict] = []
    for score, items in completed:
        relative = score - (total - score) / (len(completed) - 1) if len(completed) > 1 else 0.0
        for item in items:
            merged.append({**item, "advantage": item["advantage"] + relative})
    return merged


def error_signature(row: dict) -> str:
    """Stable deduplication key for any abnormal non-terminal episode (engine_error, step_limit, timeout, etc.)."""
    status = row.get("status") or ""
    stack = row.get("stack_trace") or ""
    top_frames = "\n".join(line.strip() for line in stack.splitlines()[:2] if line.strip())
    if status == "engine_error":
        payload = f"{row.get('error_code') or ''}|{row.get('error') or ''}|{top_frames}"
    else:
        actions = row.get("actions") or {}
        top_action = max(actions.items(), key=lambda kv: kv[1])[0] if actions else ""
        payload = f"{status}|{row.get('error_code') or ''}|{row.get('error') or ''}|{top_action}|{top_frames}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _diagnostic_filename(status: str, sig: str) -> str:
    prefix = "err" if status == "engine_error" else status
    return f"{prefix}-{sig}.json"


def save_error_diagnostic(
    diag_dir: Path,
    diag_lock: Lock,
    row: dict,
    *,
    actions_prefix: list[dict],
    last_state: dict | None,
    last_actions: list[dict] | None,
    progress_path: Path,
    ascension: int,
    round_idx: int,
    phase: str,
) -> Path | None:
    """Persist a deduplicated, 1-command replayable diagnostic bundle on any abnormal non-terminal episode."""
    status = row.get("status")
    if not status or status in TERMINAL:
        return None
    sig = error_signature(row)
    target = diag_dir / _diagnostic_filename(status, sig)
    with diag_lock:
        if target.exists():
            return target
        diag_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "signature": sig,
            "status": status,
            "host_manifest_sha256": sha(STEAM_MANIFEST),
            "round": round_idx,
            "phase": phase,
            "seed": row["seed"],
            "ascension": ascension,
            "act": row.get("act"),
            "floor": row.get("floor"),
            "steps": row.get("steps"),
            "error_code": row.get("error_code"),
            "error": row.get("error"),
            "decisions": row.get("decisions"),
            "actions": row.get("actions"),
            "recent_actions": actions_prefix[-30:],
            "stack_trace": row.get("stack_trace"),
            "stderr_tail": row.get("stderr_tail"),
            "progress_path": str(progress_path.resolve()),
            "run_config": {
                "seed": row["seed"],
                "character": "Necrobinder",
                "ascension": ascension,
                "game_mode": "standard",
                "is_custom_game": False,
                "modifiers": [],
            },
            "actions_prefix": actions_prefix,
            "last_state": last_state,
            "last_actions": last_actions,
        }
        write_json(target, bundle)
    return target


def aggregate_engine_errors(rows: list[dict]) -> list[dict]:
    """Group all abnormal non-terminal episodes (engine_error, step_limit, timeout, etc.) by signature."""
    groups: dict[str, dict] = {}
    for row in rows:
        status = row.get("status")
        if not status or status in TERMINAL:
            continue
        sig = error_signature(row)
        actions = row.get("actions") or {}
        top_action = max(actions.items(), key=lambda kv: kv[1])[0] if actions else ""
        entry = groups.setdefault(sig, {
            "signature": sig,
            "status": status,
            "diag_file": _diagnostic_filename(status, sig),
            "count": 0,
            "error_code": row.get("error_code"),
            "error": row.get("error"),
            "top_action": top_action,
            "stack_top": " | ".join(l.strip() for l in (row.get("stack_trace") or "").splitlines()[:2] if l.strip()),
            "seeds": [],
        })
        entry["count"] += 1
        entry["seeds"].append(f"{row['seed']}(a{row.get('act')}f{row.get('floor')})")
    return sorted(groups.values(), key=lambda x: -x["count"])


def replay_diagnostic(path: Path) -> int:
    """Deterministically replay a saved engine_error diagnostic JSON without any model weights."""
    verify_host(steam=True)
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    bundle = read_json(path)
    progress = read_json(Path(bundle["progress_path"]))
    prefix = bundle["actions_prefix"]
    cursor = {"index": 0}

    def replay_policy(state, actions):
        idx = cursor["index"]
        if idx < len(prefix):
            cmd = prefix[idx]
            cursor["index"] += 1
            if cmd in actions:
                return actions.index(cmd)
            raise ValueError(f"Recorded action {idx} is no longer legal: {cmd}")
        # Past the original crash point: pick first legal action to verify clean recovery
        return 0

    row = run_episode(
        STEAM_HOST, bundle["seed"], "standard", int(bundle["ascension"]), replay_policy,
        episode_timeout=110, max_steps=max(len(prefix) + 5, 64),
        strategic_observations=True, sequential_combat_selection=True,
        start_command={"cmd": "start_steam_run", "progress": progress, "run": bundle["run_config"]},
    )
    expected_status = bundle.get("status", "engine_error")
    reproduced = row["status"] == expected_status and error_signature(row) == bundle["signature"]
    print(f"Replay {path.name} (seed={bundle['seed']}, recorded_steps={len(prefix)}):", flush=True)
    print(f"  result: status={row['status']} act={row.get('act')} floor={row.get('floor')} error={row.get('error')}", flush=True)
    print(f"  verdict: {'REPRODUCED' if reproduced else 'RESOLVED (or diverged past crash)'}", flush=True)
    return 1 if reproduced else 0


def play_episode(
    model: PPOv2Policy,
    model_lock: Lock,
    progress: dict,
    seed: str,
    *,
    ascension: int = 0,
    training: bool = False,
    repeat: int = 0,
    case: dict | None = None,
    round_idx: int = 0,
    combat_temp: float = 0.7,
    strategy_temp: float = 0.9,
    gamma_step: float = 0.9995,
    gamma_room: float = 0.99,
    diag_dir: Path | None = None,
    diag_lock: Lock | None = None,
    progress_path: Path | None = None,
    batcher: InferenceBatcher | None = None,
    sample_eval: bool = False,
    focus_acts: tuple[int, ...] = (1,),
) -> tuple[dict, list[dict], list[dict]]:
    opening = (case["act"], case["floor"]) if case else None
    key = f"{seed}/focus/{case['opening']}/{round_idx}" if case else seed
    rng = training_rng(key, repeat) if (training or sample_eval) else None
    trace = RunTraceV2()
    reason = None
    combat_room = None
    actions_prefix: list[dict] = list(case["prefix"]) if case else []
    last_snapshot: dict[str, object] = {"state": None, "actions": None}

    def choose(state, actions):
        nonlocal reason, combat_room
        context = state["context"]
        room = (context["act"], context["floor"])
        combat = is_combat_decision(state, combat_room)
        combat_room = room if combat else None
        if combat:
            reason = None
        if len(actions) > 1:
            if batcher is not None:
                chosen, sample = batcher.decide(
                    state, actions, combat, reason, rng,
                    combat_temp=combat_temp, strategy_temp=strategy_temp,
                )
            else:
                with model_lock:
                    chosen, sample = model.decide(
                        state, actions, combat, reason, rng,
                        combat_temp=combat_temp, strategy_temp=strategy_temp,
                    )
            if sample is not None:
                trace.steps.append(sample)
        else:
            chosen = 0
        action = actions[chosen]
        actions_prefix.append(action)
        last_snapshot["state"] = state
        last_snapshot["actions"] = actions
        if action["action"] == "remove_card":
            reason = "remove_card"
        elif state["decision"] == "rest_site" and action["action"] == "choose_option":
            option = next(o for o in state["options"] if o["index"] == action["args"]["option_index"])
            reason = (
                "smith"
                if any("smith" in str(option.get(k, "")).lower() or "upgrade" in str(option.get(k, "")).lower()
                       for k in ("name", "option_id", "title"))
                else None
            )
        elif state["decision"] != "card_select":
            reason = None
        if action["action"] == "use_potion" and room in trace.fights:
            trace.fights[room]["potion_uses"] += 1
        return chosen

    attempt = Attempt(choose, case)
    row = run_episode(
        STEAM_HOST, seed, "standard", ascension, attempt,
        episode_timeout=110, max_steps=1500,
        strategic_observations=True, sequential_combat_selection=True,
        start_command=start_command(seed, progress, ascension),
        observe=trace.observe,
    )
    if row["status"] == "defeat":
        trace.finalize_defeat()
    if row["status"] not in TERMINAL and diag_dir is not None and diag_lock is not None and progress_path is not None:
        phase = "focus" if case else "train" if training else "eval"
        saved_diag = save_error_diagnostic(
            diag_dir, diag_lock, row,
            actions_prefix=actions_prefix,
            last_state=last_snapshot["state"],
            last_actions=last_snapshot["actions"],
            progress_path=progress_path,
            ascension=ascension,
            round_idx=round_idx,
            phase=phase,
        )
        if saved_diag is not None:
            row["diagnostic_file"] = str(saved_diag)
    terminal = row["status"] in TERMINAL
    row.update(
        fights=trace.record(),
        rollout=repeat + 1,
        act1_cleared=(row.get("act") or 0) > 1 or row["victory"] is True,
        act2_cleared=(row.get("act") or 0) > 2 or row["victory"] is True,
    )
    if trace.act2_entry is not None:
        row["act2_entry"] = trace.act2_entry
    row["last_floor_entry"] = attempt.policy.entries.get((row["act"], row["floor"]))
    target = trace.fights.get(opening) if opening else None
    if case:
        row["focus"] = {
            "seed": seed, "opening": case["opening"],
            "act": opening[0], "floor": opening[1],
            "reached": attempt.reached_target,
        }
    if target and target["hp_out"] is not None and attempt.reached_target:
        row["focus"].update(
            survived=target["hp_out"] > 0,
            net_hp_lost=target["hp_in"] - target["hp_out"],
            potion_uses=target["potion_uses"],
            kill_order=target["kill_order"],
            return_=fight_return(target),
        )
    if not training or not attempt.reached_target:
        return row, [], []
    cases = list(attempt.cases(row, round_idx, allowed_acts=focus_acts)) if terminal and not case else []
    samples = compute_ppo_returns(trace, row, opening, gamma_step=gamma_step, gamma_room=gamma_room)
    return row, samples, cases


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    wins = sum(r["victory"] is True for r in rows)
    act1 = sum(r["act1_cleared"] for r in rows)
    act2 = sum(r["act2_cleared"] for r in rows)
    unresolved = sum(r["status"] not in TERMINAL and not r["act1_cleared"] for r in rows)
    boss = [b["hp"] for r in rows for b in r["boss_entries"] if b["act"] == 1]
    losses = [
        f["hp_in"] - f["hp_out"]
        for r in rows for f in r["fights"]
        if f["act"] == 1 and f["room"] == "Monster" and f["hp_out"] and f["hp_out"] > 0
    ]
    elites_won = sum(
        1 for r in rows for f in r["fights"]
        if f["room"] == "Elite" and f["hp_out"] and f["hp_out"] > 0
    )
    progress_list = [
        51 if r["victory"] is True else ((r["act"] or 1) - 1) * 17 + min(r["floor"] or 0, 17)
        for r in rows
    ]
    return {
        "attempts": n,
        "wins": wins,
        "win_rate": wins / n,
        "win_wilson95": wilson(wins, n),
        "act1_clears": act1,
        "act1_clear_rate": act1 / n,
        "act1_wilson95": wilson(act1, n),
        "act2_clears": act2,
        "unresolved_before_act2": unresolved,
        "boss_entries": len(boss),
        "median_boss_hp": statistics.median(boss) if boss else None,
        "elites_won": elites_won,
        "median_normal_fight_hp_loss": statistics.median(losses) if losses else None,
        "median_progress": statistics.median(progress_list),
        "mean_progress": float(statistics.mean(progress_list)),
        "statuses": dict(Counter(r["status"] for r in rows)),
    }


def dev_rank(summary: dict) -> list[float] | None:
    """Rank complete development evaluations using a smooth composite score so a single noisy clear
    cannot override a broadly stronger checkpoint. Requires every game to terminate cleanly."""
    if not summary:
        return None
    if any(status not in TERMINAL and count > 0 for status, count in summary["statuses"].items()):
        return None
    n = max(1, int(summary.get("attempts") or sum(summary["statuses"].values())))
    win_rate = float(summary["wins"]) / n
    act2_rate = float(summary["act2_clears"]) / n
    act1_rate = float(summary["act1_clears"]) / n
    boss_rate = float(summary["boss_entries"]) / n
    mean_prog = float(summary["mean_progress"])
    composite = (
        24.0 * win_rate
        + 10.0 * act2_rate
        + 5.0 * act1_rate
        + 1.0 * boss_rate
        + 0.12 * mean_prog
    )
    return [
        round(composite, 4),
        float(summary["wins"]),
        float(summary["act2_clears"]),
        float(summary["act1_clears"]),
        round(mean_prog, 4),
    ]


def _norm_mcfg(cfg_dict: dict) -> dict:
    out = dict(cfg_dict)
    out.setdefault("value_bins", 1)
    out.setdefault("value_symlog_max", 2.6)
    return out


def parse_focus_acts(spec: str) -> tuple[int, ...]:
    """Parse comma-separated act numbers for focus case collection (e.g. '1', '2,3', '1,2,3')."""
    acts: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val not in (1, 2, 3):
            raise argparse.ArgumentTypeError(f"Focus act must be 1, 2, or 3 (got {val})")
        if val not in acts:
            acts.append(val)
    if not acts:
        raise argparse.ArgumentTypeError("At least one act must be specified in --focus-acts")
    return tuple(acts)


def update_top_k_checkpoints(
    top_k_entries: list[dict],
    *,
    k: int,
    round_idx: int,
    rank: list[float] | None,
    payload: dict,
    top_k_dir: Path,
) -> list[dict]:
    """Maintain up to K weights-only candidate checkpoints ranked by dev_rank, pruning evicted files."""
    if k <= 0 or rank is None:
        return list(top_k_entries)
    entries = [dict(e) for e in top_k_entries]
    existing_idx = next((i for i, e in enumerate(entries) if int(e["round"]) == int(round_idx)), None)
    qualifies = (
        existing_idx is not None
        or len(entries) < k
        or rank > min(entries, key=lambda e: (e["rank"], int(e["round"])))["rank"]
    )
    if not qualifies:
        return entries

    top_k_dir.mkdir(parents=True, exist_ok=True)
    fname = f"round-{int(round_idx):04}.pt"
    target = top_k_dir / fname
    tmp = top_k_dir / f"{fname}.tmp"
    torch.save(payload, tmp)
    tmp.replace(target)

    if existing_idx is not None:
        entries.pop(existing_idx)
    entries.append({"round": int(round_idx), "rank": list(rank), "file": fname})
    entries.sort(key=lambda e: (e["rank"], int(e["round"])), reverse=True)

    kept = entries[:k]
    evicted = entries[k:]
    kept_files = {e["file"] for e in kept}
    for e in evicted:
        if e["file"] not in kept_files:
            (top_k_dir / e["file"]).unlink(missing_ok=True)

    write_json(top_k_dir / "manifest.json", {"top_k": kept})
    return kept


def validate_resume_protocol(saved_proto: dict, protocol: dict) -> None:
    """Verify --resume matches model, host, Steam progress, dev panel, and observation schema."""
    saved_host = saved_proto["host"]
    if _norm_mcfg(saved_proto["model_config"]) != _norm_mcfg(protocol["model_config"]):
        raise ValueError("Model config changed since this run began")
    if saved_host != protocol["host"] and saved_host not in COMPATIBLE_STEAM_HOSTS:
        raise ValueError("Host manifest changed to an incompatible build since this run began")
    if saved_proto.get("progress") != protocol.get("progress"):
        raise ValueError("Steam progress.save snapshot changed since this run began")
    if saved_proto.get("development_seeds") != protocol.get("development_seeds"):
        raise ValueError("Development evaluation panel (--dev-games / seeds) changed since this run began")
    saved_obs = saved_proto.get("observation_version")
    if saved_obs is not None and saved_obs != protocol.get("observation_version"):
        raise ValueError(f"Observation schema changed ({saved_obs} -> {protocol.get('observation_version')})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-diagnostic", type=Path, help="Replay a saved diagnostics/err-*.json bundle and exit")
    parser.add_argument("--progress", type=Path, help="Steam progress.save snapshot")
    parser.add_argument("--out", type=Path, help="Output run directory")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="base")
    parser.add_argument("--value-bins", type=positive_int, default=1,
                        help="Value head bins: 1 = scalar Huber, >1 (e.g. 41) = DreamerV3 symlog two-hot critic")
    parser.add_argument("--value-symlog-max", type=float, default=2.6,
                        help="Symlog range [-max, +max] for distributional two-hot critic")
    parser.add_argument("--critic-warmup", type=nonnegative_int, default=None,
                        help="Initial rounds updating only the value head (auto=2 when upgrading value head via --init)")
    parser.add_argument("--ascension", type=int, choices=range(11), default=0)
    parser.add_argument("--asc-max", type=int, choices=range(11), default=0,
                        help="Optional auto-promotion target ascension (0 = stay at --ascension)")
    parser.add_argument("--asc-promote-rate", type=float, default=0.5,
                        help="Promote ascension when dev win_rate reaches this threshold")
    parser.add_argument("--init", type=Path, help="Optional PPO v2 checkpoint (.pt) for stage transfer (e.g. A0->A5->A10)")
    parser.add_argument("--seed-namespace", type=str, default=None,
                        help="Explicit training seed namespace (default: stage-unique on --init so stages never repeat seeds)")
    parser.add_argument("--seed-offset", type=nonnegative_int, default=None,
                        help="Starting round offset for training seed generation (default: parent total_rounds on --init, else 0)")
    parser.add_argument("--rounds", type=positive_int, default=64)
    parser.add_argument("--fresh-seeds", type=positive_int, default=12)
    parser.add_argument("--rollouts-per-seed", type=positive_int, default=2)
    parser.add_argument("--focus-cases", type=nonnegative_int, default=3,
                        help="Hard openings replayed per round (0 = pure fresh self-play)")
    parser.add_argument("--focus-rollouts", type=positive_int, default=3,
                        help="Rollouts per hard opening (>=3 recommended for group-relative baseline)")
    parser.add_argument("--focus-acts", type=parse_focus_acts, default=(1,),
                        help="Comma-separated acts eligible for focus-case collection, e.g. '1' or '2,3'")
    parser.add_argument("--dev-games", type=positive_int, default=24)
    parser.add_argument("--eval-every", type=positive_int, default=4)
    parser.add_argument("--eval-games", type=positive_int, default=64)
    parser.add_argument("--top-k", type=nonnegative_int, default=3,
                        help="Number of top weights-only dev checkpoints to retain under <out>/top-k/ (0 = off)")
    parser.add_argument("--skip-r0-eval", action="store_true",
                        help="Skip initial round-0 greedy dev baseline before training round 1")
    parser.add_argument("--minibatch", type=positive_int, default=128)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--combat-temp-start", type=float, default=None,
                        help="Initial combat sampling temp (default: 0.55 with --init, 0.85 from scratch)")
    parser.add_argument("--combat-temp-end", type=float, default=None,
                        help="Final combat sampling temp (default: 0.30 with --init, 0.35 from scratch)")
    parser.add_argument("--strategy-temp-start", type=float, default=None,
                        help="Initial strategic sampling temp (default: 0.65 with --init, 0.95 from scratch)")
    parser.add_argument("--strategy-temp-end", type=float, default=None,
                        help="Final strategic sampling temp (default: 0.40 with --init, 0.50 from scratch)")
    parser.add_argument("--target-kl", type=float, default=0.025)
    parser.add_argument("--gamma-step", type=float, default=0.9995)
    parser.add_argument("--gamma-room", type=float, default=0.99)
    parser.add_argument("--max-error-rate", type=float, default=0.25,
                        help="Circuit breaker: stop after saving checkpoint if a round's engine_error share exceeds this (0 = off)")
    parser.add_argument("--solver-demos", type=Path, help="Optional CombatSolver JSONL demos (zero human data)")
    parser.add_argument("--bc-coef", type=float, default=0.0, help="Optional BC regularization weight for --solver-demos")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.replay_diagnostic:
        raise SystemExit(replay_diagnostic(args.replay_diagnostic))
    if not args.progress or not args.out:
        parser.error("--progress and --out are required unless --replay-diagnostic is set")

    if args.init and args.resume:
        parser.error("Use --init for a new stage and --resume to continue an existing directory")

    torch.set_num_threads(1 if args.workers > 1 else 2)
    torch.set_num_interop_threads(1)
    verify_host(steam=True)
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")

    progress = read_json(args.progress)
    mcfg = replace(PRESETS[args.preset], value_bins=args.value_bins, value_symlog_max=args.value_symlog_max)
    model = PPOv2Policy(mcfg, args.device)
    model_lock = Lock()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-6)

    bc_samples = load_bc_demos(args.solver_demos, mcfg.vocab_size) if args.solver_demos else None
    if bc_samples is not None:
        print(f"Loaded {len(bc_samples)} CombatSolver BC demonstrations (bc_coef={args.bc_coef})", flush=True)

    namespace = f"{EXPERIMENT}/a{args.ascension}/{args.preset}"
    dev_seeds = panel("development", args.dev_games, EXPERIMENT)
    test_seeds = [s for s in panel("final", args.eval_games + len(dev_seeds), EXPERIMENT) if s not in set(dev_seeds)][:args.eval_games]
    protected = set(dev_seeds) | set(test_seeds)

    fine_tuning = bool(args.init)
    parent = torch.load(args.init, map_location=args.device, weights_only=False) if args.init else None
    if args.seed_namespace is not None:
        train_seed_namespace = args.seed_namespace
    elif fine_tuning:
        train_seed_namespace = f"{namespace}/{args.out.name}"
    else:
        train_seed_namespace = namespace
    if args.seed_offset is not None:
        seed_offset = int(args.seed_offset)
    elif parent is not None:
        seed_offset = int(parent.get("total_rounds", parent.get("round", 0)))
    else:
        seed_offset = 0

    c_temp_bounds = [
        args.combat_temp_start if args.combat_temp_start is not None else (0.55 if fine_tuning else 0.85),
        args.combat_temp_end if args.combat_temp_end is not None else (0.30 if fine_tuning else 0.35),
    ]
    s_temp_bounds = [
        args.strategy_temp_start if args.strategy_temp_start is not None else (0.65 if fine_tuning else 0.95),
        args.strategy_temp_end if args.strategy_temp_end is not None else (0.40 if fine_tuning else 0.50),
    ]
    ent_c_bounds = [0.004, 0.001] if fine_tuning else [0.008, 0.002]
    ent_s_bounds = [0.012, 0.005] if fine_tuning else [0.025, 0.012]

    protocol = {
        "experiment": EXPERIMENT,
        "observation_version": OBSERVATION_VERSION,
        "host": sha(STEAM_MANIFEST),
        "progress": sha(args.progress),
        "preset": args.preset,
        "model_config": asdict(mcfg),
        "parameter_summary": model.parameter_summary(),
        "ascension": args.ascension,
        "asc_max": args.asc_max,
        "sampling": {
            "fresh_seeds": args.fresh_seeds,
            "rollouts_per_seed": args.rollouts_per_seed,
            "focus_cases": args.focus_cases,
            "focus_rollouts": args.focus_rollouts,
            "focus_acts": list(args.focus_acts),
            "top_k": args.top_k,
            "train_seed_namespace": train_seed_namespace,
            "seed_offset": seed_offset,
        },
        "ppo": {
            "minibatch": args.minibatch,
            "epochs": args.epochs,
            "lr": args.lr,
            "target_kl": args.target_kl,
            "gamma_step": args.gamma_step,
            "gamma_room": args.gamma_room,
            "bc_coef": args.bc_coef,
            "schedule_bounds": {
                "combat_temp": c_temp_bounds,
                "strategy_temp": s_temp_bounds,
                "ent_combat": ent_c_bounds,
                "ent_strategic": ent_s_bounds,
            },
        },
        "development_seeds": dev_seeds,
        "evaluation_seeds": test_seeds,
    }
    if args.init:
        protocol["parent_checkpoint"] = sha(args.init)

    proto_path = args.out / "protocol.json"
    ckpt_path = args.out / "ppo.pt"
    best_path = args.out / "best-checkpoint.pt"
    top_k_dir = args.out / "top-k"
    diag_dir = args.out / "diagnostics"
    diag_lock = Lock()

    last_round = 0
    current_ascension = args.ascension
    pool: list[dict] = []
    best_rank: list[float] | None = None
    best_round = 0
    top_k_entries: list[dict] = []
    focus_acts: tuple[int, ...] = tuple(args.focus_acts)

    critic_warmup_rounds = args.critic_warmup or 0
    if args.resume:
        saved_proto = read_json(proto_path)
        validate_resume_protocol(saved_proto, protocol)
        saved_sampling = saved_proto.get("sampling", {})
        if args.seed_namespace is None:
            train_seed_namespace = saved_sampling.get("train_seed_namespace", namespace)
        if args.seed_offset is None:
            seed_offset = int(saved_sampling.get("seed_offset", 0))
        if "focus_acts" in saved_sampling:
            focus_acts = tuple(int(a) for a in saved_sampling["focus_acts"])
        saved_bounds = saved_proto.get("ppo", {}).get("schedule_bounds")
        if saved_bounds:
            if args.combat_temp_start is None:
                c_temp_bounds[0] = float(saved_bounds["combat_temp"][0])
            if args.combat_temp_end is None:
                c_temp_bounds[1] = float(saved_bounds["combat_temp"][1])
            if args.strategy_temp_start is None:
                s_temp_bounds[0] = float(saved_bounds["strategy_temp"][0])
            if args.strategy_temp_end is None:
                s_temp_bounds[1] = float(saved_bounds["strategy_temp"][1])
            ent_c_bounds = [float(x) for x in saved_bounds["ent_combat"]]
            ent_s_bounds = [float(x) for x in saved_bounds["ent_strategic"]]
        if ckpt_path.exists():
            saved = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            last_round = int(saved["round"])
            current_ascension = int(saved.get("ascension", args.ascension))
            pool = saved.get("pool", [])
            best_rank = saved.get("best_rank")
            best_round = int(saved.get("best_round", 0))
            top_k_entries = list(saved.get("top_k", []))
    else:
        if parent is not None:
            loaded_n, total_n, skipped_keys = model.load_compatible_state_dict(parent["model"])
            if skipped_keys:
                if args.critic_warmup is None:
                    critic_warmup_rounds = 2
                print(
                    f"Transferred {loaded_n:,}/{total_n:,} ({100.0 * loaded_n / max(1, total_n):.2f}%) weights from {args.init}; "
                    f"reinitialized {skipped_keys} (critic_warmup={critic_warmup_rounds} rounds).",
                    flush=True,
                )
            pool = [c for c in parent.get("pool", []) if c["seed"] not in protected and c.get("act", 1) in focus_acts]
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(proto_path, protocol)

    print(f"PPOv2 {args.preset}: {model.parameter_summary()} on {args.device} (A{current_ascension}, focus_acts={focus_acts}, top_k={args.top_k})", flush=True)

    def collect(jobs, *, training=False, round_idx=0, c_temp=0.7, s_temp=0.9, target_model=None):
        active_model = target_model or model
        active_lock = Lock() if target_model is not None else model_lock
        with InferenceBatcher(active_model, max_batch=max(16, args.workers * 2), max_wait_ms=1.5) as batcher:
            def one(job):
                seed, repeat, case = job
                outcome = play_episode(
                    active_model, active_lock, progress, seed,
                    ascension=current_ascension, training=training, repeat=repeat,
                    case=case, round_idx=round_idx,
                    combat_temp=c_temp, strategy_temp=s_temp,
                    gamma_step=args.gamma_step, gamma_room=args.gamma_room,
                    diag_dir=diag_dir, diag_lock=diag_lock, progress_path=args.progress,
                    batcher=batcher, focus_acts=focus_acts,
                )
                row = outcome[0]
                entry = row["last_floor_entry"] or {}
                phase = "focus" if case else "train" if training else "eval"
                label = f"{seed}/{repeat + 1}" if training else seed
                hp = show_hp(entry.get("before_travel"))
                extra = ""
                if case and "survived" in row.get("focus", {}):
                    finfo = row["focus"]
                    extra = f" surv={int(finfo['survived'])} dHP={finfo.get('net_hp_lost')} ret={finfo.get('return_', 0.0):+.2f}"
                print(f"{phase} A{current_ascension} {label} {row['status']} act={row['act']} floor={row['floor']} hp={hp}{extra}", flush=True)
                return outcome

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                return list(executor.map(one, jobs))

    if not args.resume and not args.skip_r0_eval:
        r0_rows = [row for row, *_ in collect([(s, 0, None) for s in dev_seeds], training=False)]
        r0_summary = summarize(r0_rows)
        r0_rank = dev_rank(r0_summary)
        protocol["r0_baseline"] = {"dev_summary": r0_summary, "dev_rank": r0_rank}
        write_json(proto_path, protocol)
        write_json(args.out / "round-0000.json", {
            "round": 0,
            "ascension": current_ascension,
            "schedule": None,
            "train_summary": None,
            "dev_summary": r0_summary,
            "dev_rank": r0_rank,
            "ppo_update": None,
            "pool_size": len(pool),
            "engine_errors": [],
            "best_round": 0,
            "best_rank": r0_rank,
            "train_rows": [],
        })
        print(
            f"[R0 BASELINE] rank={r0_rank} | wins={r0_summary.get('wins', 0)}/{r0_summary.get('attempts', 0)} "
            f"a2={r0_summary.get('act2_clears', 0)} a1={r0_summary.get('act1_clears', 0)} "
            f"boss={r0_summary.get('boss_entries', 0)} mProg={r0_summary.get('mean_progress', 0.0):.2f}",
            flush=True,
        )

    for number in range(last_round + 1, args.rounds + 1):
        lr_now = cosine_lr(args.lr, number, args.rounds)
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        c_temp = schedule_value(c_temp_bounds[0], c_temp_bounds[1], number, args.rounds)
        s_temp = schedule_value(s_temp_bounds[0], s_temp_bounds[1], number, args.rounds)
        ent_c = schedule_value(ent_c_bounds[0], ent_c_bounds[1], number, args.rounds)
        ent_s = schedule_value(ent_s_bounds[0], ent_s_bounds[1], number, args.rounds)

        seed_round = seed_offset + number
        fresh = [
            s for s in panel(f"train/{seed_round}", args.fresh_seeds + len(protected), train_seed_namespace)
            if s not in protected
        ][:args.fresh_seeds]
        jobs = [(seed, repeat, None) for seed in fresh for repeat in range(args.rollouts_per_seed)]

        train_rows: list[dict] = []
        samples: list[dict] = []
        for row, batch, cases in collect(jobs, training=True, round_idx=number, c_temp=c_temp, s_temp=s_temp):
            train_rows.append(row)
            samples.extend(batch)
            pool.extend(cases)

        # Fair focus selection rotating across all priority tags when pool has enough cases
        used: set[str] = set()
        focus_cases: list[dict] = []
        for idx in range(args.focus_cases):
            tag = PRIORITIES[(number - 1 + idx) % len(PRIORITIES)]
            case = choose_case(pool, tag, used)
            if case is None:
                break
            used.add(case["seed"])
            focus_cases.append(case)

        if focus_cases:
            f_jobs = [(case["seed"], repeat, case) for case in focus_cases for repeat in range(args.focus_rollouts)]
            results = iter(collect(f_jobs, training=True, round_idx=number, c_temp=c_temp, s_temp=s_temp))
            for case in focus_cases:
                group = []
                stale = False
                for _ in range(args.focus_rollouts):
                    row, batch, _ = next(results)
                    train_rows.append(row)
                    stale |= not row["focus"]["reached"]
                    group.append((row, batch))
                samples.extend(focus_advantages(group))
                case["uses"] += args.focus_rollouts
                completed_focus = [r["focus"] for r, _ in group if "survived" in r.get("focus", {})]
                hopeless = len(completed_focus) == args.focus_rollouts and all(not f["survived"] for f in completed_focus)
                trivial = len(completed_focus) == args.focus_rollouts and all(
                    f["survived"] and f.get("net_hp_lost", 1) == 0 for f in completed_focus
                )
                if (stale or hopeless or trivial) and case in pool:
                    pool.remove(case)

        pool = retain_cases(pool)
        critic_only = number <= critic_warmup_rounds
        update_stats = model.update(
            samples, optimizer, training_rng(f"{train_seed_namespace}/mb/{seed_round}", 0),
            epochs=args.epochs, minibatch=args.minibatch,
            target_kl=args.target_kl, ent_combat=ent_c, ent_strategic=ent_s,
            combat_temp=c_temp, strategy_temp=s_temp,
            bc_samples=bc_samples, bc_coef=args.bc_coef,
            critic_only=critic_only,
        )

        dev_summary = None
        if number % args.eval_every == 0 or number == args.rounds:
            dev_rows = [row for row, *_ in collect([(s, 0, None) for s in dev_seeds], training=False)]
            dev_summary = summarize(dev_rows)
            rank = dev_rank(dev_summary)
            if rank is not None:
                cand_payload = {
                    "round": number,
                    "total_rounds": seed_round,
                    "ascension": current_ascension,
                    "preset": args.preset,
                    "model_config": asdict(mcfg),
                    "rank": rank,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_round = number
                    tmp_best = args.out / "best.tmp"
                    torch.save(cand_payload, tmp_best)
                    tmp_best.replace(best_path)
                if args.top_k > 0:
                    top_k_entries = update_top_k_checkpoints(
                        top_k_entries,
                        k=args.top_k,
                        round_idx=number,
                        rank=rank,
                        payload=cand_payload,
                        top_k_dir=top_k_dir,
                    )
            if args.asc_max > current_ascension and dev_summary["win_rate"] >= args.asc_promote_rate:
                current_ascension += 1
                best_rank = None
                top_k_entries = []
                print(f"Promoted ascension to A{current_ascension} after round {number}!", flush=True)

        err_groups = aggregate_engine_errors(train_rows)
        err_count = sum(g["count"] for g in err_groups)
        err_rate = err_count / max(1, len(train_rows))

        report = {
            "round": number,
            "ascension": current_ascension,
            "schedule": {"lr": lr_now, "combat_temp": c_temp, "strategy_temp": s_temp, "ent_combat": ent_c, "ent_strategic": ent_s},
            "train_summary": summarize(train_rows),
            "dev_summary": dev_summary,
            "ppo_update": update_stats,
            "pool_size": len(pool),
            "engine_errors": err_groups,
            "best_round": best_round,
            "best_rank": best_rank,
            "top_k": top_k_entries,
            "train_rows": train_rows,
        }
        write_json(args.out / f"round-{number:04}.json", report)

        tmp = args.out / "ppo.tmp"
        torch.save({
            "round": number,
            "total_rounds": seed_round,
            "ascension": current_ascension,
            "preset": args.preset,
            "model_config": asdict(mcfg),
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "pool": pool,
            "best_rank": best_rank,
            "best_round": best_round,
            "top_k": top_k_entries,
        }, tmp)
        tmp.replace(ckpt_path)
        print(
            f"Round {number}/{args.rounds} saved (lr={lr_now:.2e}, T={c_temp:.2f}/{s_temp:.2f}, "
            f"upd={update_stats['updates']}, kl={update_stats['mean_kl']:.4f}, err={err_count}/{len(train_rows)}, best=R{best_round}).",
            flush=True,
        )
        if err_groups:
            for g in err_groups:
                detail = g["stack_top"] or f"top_action={g.get('top_action') or '-'}"
                print(
                    f"  [{g['status'].upper()} x{g['count']}] sig={g['signature']} err={g['error']} "
                    f"{detail} diag={diag_dir / g['diag_file']}",
                    flush=True,
                )
        if args.max_error_rate > 0 and err_rate > args.max_error_rate:
            print(
                f"CIRCUIT BREAKER TRIGGERED: round {number} abnormal episode rate {err_rate:.1%} "
                f"({err_count}/{len(train_rows)}) exceeded --max-error-rate {args.max_error_rate:.1%}. "
                f"Checkpoint saved to {ckpt_path}. Replay bundles under {diag_dir} with --replay-diagnostic.",
                flush=True,
            )
            raise SystemExit(2)

    # Final held-out evaluation using best-checkpoint, latest checkpoint, and any extra Top-K candidates
    final_eval = {}
    evaluated_by_round: dict[int, list[dict]] = {}
    if best_path.exists():
        best_model = PPOv2Policy(mcfg, args.device)
        best_model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=False)["model"])
        best_rows = [row for row, *_ in collect([(s, 0, None) for s in test_seeds], training=False, target_model=best_model)]
        evaluated_by_round[best_round] = best_rows
        final_eval["best"] = {"round": best_round, "summary": summarize(best_rows), "rows": best_rows}
        print("BEST EVAL:", final_eval["best"]["summary"], flush=True)

    if args.rounds in evaluated_by_round:
        final_rows = evaluated_by_round[args.rounds]
    else:
        final_rows = [row for row, *_ in collect([(s, 0, None) for s in test_seeds], training=False)]
        evaluated_by_round[args.rounds] = final_rows
    final_eval["final"] = {"round": args.rounds, "summary": summarize(final_rows), "rows": final_rows}
    print("FINAL EVAL:", final_eval["final"]["summary"], flush=True)

    if top_k_entries:
        top_k_eval = []
        for entry in top_k_entries:
            eround = int(entry["round"])
            epath = top_k_dir / entry["file"]
            if eround in evaluated_by_round:
                erows = evaluated_by_round[eround]
            elif epath.exists():
                cand_model = PPOv2Policy(mcfg, args.device)
                cand_model.load_state_dict(torch.load(epath, map_location=args.device, weights_only=False)["model"])
                erows = [row for row, *_ in collect([(s, 0, None) for s in test_seeds], training=False, target_model=cand_model)]
                evaluated_by_round[eround] = erows
            else:
                continue
            esum = summarize(erows)
            top_k_eval.append({"round": eround, "dev_rank": entry["rank"], "summary": esum, "rows": erows})
            print(f"TOP-K EVAL (R{eround}):", esum, flush=True)
        if top_k_eval:
            final_eval["top_k"] = top_k_eval

    write_json(args.out / "evaluation.json", final_eval)


if __name__ == "__main__":
    main()
