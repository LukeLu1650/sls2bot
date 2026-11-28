#!/usr/bin/env python3
"""Macro-Only PPO Adaptation under a Fixed In-Combat SL Partner.

Scientific design:
  - Initializes PolicyValueNet from a frozen checkpoint (e.g. v4879 EMA).
  - FREEZES the entire shared Transformer encoder and ALL combat heads (`cand_mlp`,
    `value_mlp`, `value_hl`, `fh_*`, embeddings, etc.).
  - Trains ONLY the out-of-combat (macro) heads: `cand_mlp_m`, `value_mlp_m`, `value_hl_m`.
    Because combat frames never read `*_m` heads, the combat policy + fixed SL partner is
    bit-for-bit stationary across the entire training run, and any improvement over
    "Original v4879 Macro + Same Fixed SL" is cleanly attributable to out-of-combat
    adaptation (card drafting, pathing, elite appetite, campfires, shops, events).
  - Each combat runs the exact deployment SL protocol (`snapshot(0)/restore(0)`, room-tiered
    budgets, Pareto-consistent resource score `end_hp + potion_hp_frac * max_hp * end_pots`)
    and submits ONLY the single chosen trajectory to the live run. Discarded SL branches
    never emit rewards or macro steps.
  - Pure on-policy GAE (`gamma=1.0, lam=0.95`) along the realized out-of-combat decision chain.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
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
import torch.nn.functional as F  # noqa: E402

from eval_combat_sl import (  # noqa: E402
    is_perfect_win,
    potions_held,
    rank_key,
    reset_full_run,
    trial_rng_seed,
)
from eval_v1140 import load_net_weights  # noqa: E402
from sts2env.env import EngineError, Sts2Env  # noqa: E402
from sts2env.protocol import ActionKind, DecisionKind  # noqa: E402
from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.features import Ragged, collate_ragged  # noqa: E402
from sts2rl.train import hl_target  # noqa: E402


MACRO_HEAD_PREFIXES = ("cand_mlp_m.", "value_mlp_m.", "value_hl_m.")


class DualInferer:
    """Dynamic micro-batching CUDA/CPU inference server returning (probs, value) for MODE_RUN."""

    def __init__(self, net, device: str, max_batch: int = 16, wait_ms: float = 1.5, use_bf16: bool = False):
        self.net = net
        self.device = device
        self.max_batch = max(1, int(max_batch))
        self.wait_s = max(0.0, float(wait_ms) / 1000.0)
        self.use_bf16 = bool(use_bf16) and str(device).startswith("cuda") and torch.cuda.is_bf16_supported()
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._pause_lock = threading.Lock()
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

    def pause_context(self):
        return self._pause_lock

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
                with self._pause_lock:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                        logits, values = self.net.forward_infer(batch)
                    logits_f = logits.float().cpu()
                    vals_f = values.float().cpu().tolist()
                for i, (ob, ev, box) in enumerate(batch_items):
                    lg = logits_f[i, : ob.num_actions].double()
                    probs = torch.softmax(lg, dim=-1).numpy()
                    probs = np.maximum(probs, 0.0)
                    s = float(probs.sum())
                    probs = (probs / s) if (np.isfinite(s) and s > 0.0) else np.full(ob.num_actions, 1.0 / max(ob.num_actions, 1), dtype=np.float64)
                    box.append((probs, float(vals_f[i])))
                    ev.set()
            except Exception as err:
                for _ob, ev, box in batch_items:
                    box.extend([None, err])
                    ev.set()


def choose_combat_action(obs, inferer: DualInferer, temp: float, rng: np.random.Generator | None) -> int:
    if obs.num_actions <= 1:
        return 0
    probs, _ = inferer.eval(obs)
    if temp <= 0.0 or rng is None:
        return int(np.argmax(probs))
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temp
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(obs.num_actions, p=p))


def rollout_combat_with_inferer(
    env: Sts2Env,
    obs,
    inferer: DualInferer,
    temp: float,
    rng: np.random.Generator | None,
    start_cw: int,
    max_steps_fight: int,
) -> tuple[dict, object]:
    actions: list[int] = []
    steps = 0
    while (
        not obs.done
        and not obs.stuck
        and obs.in_combat
        and obs.combats_won == start_cw
        and obs.num_actions > 0
        and steps < max_steps_fight
    ):
        a = choose_combat_action(obs, inferer, temp=temp, rng=rng)
        actions.append(a)
        obs = env.step(a)
        steps += 1

    if obs.stuck or (steps >= max_steps_fight and obs.in_combat and obs.combats_won == start_cw and not obs.done):
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


def collect_one_macro_run(
    env: Sts2Env,
    seed: str,
    rng_seed: int,
    inferer: DualInferer,
    cfg: argparse.Namespace,
) -> dict:
    """Run one full A10 game with fixed In-Combat SL and stochastic on-policy macro choices."""
    obs = reset_full_run(env, cfg.character, cfg.asc, seed, cfg.stall_turns)
    rng_mac = np.random.default_rng(rng_seed ^ 0x5A5A5A5A)
    temps = (0.4, 0.7, 1.0, 0.5, 0.85, 1.15)
    budgets = {
        "Monster": (cfg.monster_trials, cfg.monster_timeout),
        "Elite": (cfg.elite_trials, cfg.elite_timeout),
        "Boss": (cfg.boss_trials, cfg.boss_timeout),
    }
    floor_combat_counts: dict[tuple[int, int], int] = {}

    # Macro trajectory buffers (ONLY realized out-of-combat steps!)
    m_obs: list = []
    m_act: list[int] = []
    m_logp: list[float] = []
    m_val: list[float] = []
    m_rew: list[float] = []
    m_multi: list[bool] = []
    m_is_map: list[bool] = []

    # Telemetry
    elites_fought = 0
    rest_heal = 0
    rest_smith = 0
    card_pick = 0
    card_skip = 0
    a2_boss_entry: dict | None = None
    a3_boss1_rewarded = False
    prev_floor = int(obs.floor)
    prev_act = int(obs.act)
    run_steps = 0
    stuck = False

    def add_reward(val: float) -> None:
        if m_rew and val != 0.0:
            m_rew[-1] += val

    while not obs.done and not obs.stuck and obs.num_actions > 0 and run_steps < cfg.max_steps_run:
        if not obs.in_combat:
            # Realized out-of-combat macro decision
            n_act = int(obs.num_actions)
            if n_act <= 1:
                _, v = inferer.eval(obs)
                a = 0
                lp = 0.0
            else:
                probs, v = inferer.eval(obs)
                if cfg.macro_temp != 1.0:
                    lg = np.log(np.clip(probs, 1e-12, 1.0)) / cfg.macro_temp
                    lg -= lg.max()
                    p_s = np.exp(lg)
                    p_s /= p_s.sum()
                else:
                    p_s = probs / probs.sum()
                a = int(rng_mac.choice(n_act, p=p_s))
                lp = float(np.log(max(probs[a], 1e-12)))

            # Behavioral telemetry
            dec = int(obs.decision)
            if n_act > 1 and a < obs.cand_ints.shape[0]:
                ak = int(obs.cand_ints[a, 0])
                if dec == int(DecisionKind.CARD_REWARD):
                    if ak == int(ActionKind.PICK_CARD_REWARD):
                        card_pick += 1
                    elif ak == int(ActionKind.CARD_REWARD_ALT):
                        card_skip += 1
                elif dec == int(DecisionKind.REST_SITE) and ak == int(ActionKind.REST_OPTION):
                    # Heuristic by label if available or cand slot 0=heal/1=smith not guaranteed; track hp jump below
                    pass

            m_obs.append(obs)
            m_act.append(a)
            m_logp.append(lp)
            m_val.append(v)
            m_rew.append(0.0)
            m_multi.append(n_act > 1)
            m_is_map.append(dec == int(DecisionKind.MAP))

            hp_before = int(obs.hp)
            ak_chosen = int(obs.cand_ints[a, 0]) if a < obs.cand_ints.shape[0] else -1
            if ak_chosen == int(ActionKind.DISCARD_POTION):
                add_reward(-cfg.potion_discard_penalty)

            obs = env.step(a)
            run_steps += 1

            # Rest site telemetry (heal increases HP, smith leaves HP unchanged)
            if dec == int(DecisionKind.REST_SITE) and ak_chosen == int(ActionKind.REST_OPTION):
                if int(obs.hp) > hp_before:
                    rest_heal += 1
                else:
                    rest_smith += 1

            # Realized progression rewards
            nf = int(obs.floor)
            na = int(obs.act)
            if nf > prev_floor:
                add_reward(cfg.floor_reward * (nf - prev_floor))
                prev_floor = nf
            if na > prev_act:
                if prev_act == 0 and na >= 1:
                    add_reward(cfg.act1_boss_reward)
                if prev_act <= 1 and na >= 2:
                    add_reward(cfg.act2_boss_reward)
                prev_act = na
            continue

        # Combat: fixed frozen v4879 + fixed SL budget (ONLY realized outcome enters run!)
        cinfo = (obs.extra or {}).get("combat") or {}
        room = str(cinfo.get("room") or "Monster")
        if room not in ("Monster", "Elite", "Boss"):
            room = "Boss" if int(obs.floor) in (17, 33, 48, 49) else "Monster"
        act = int(cinfo.get("act", obs.act))
        floor = int(obs.floor)
        fc_idx = floor_combat_counts.get((act, floor), 0)
        floor_combat_counts[(act, floor)] = fc_idx + 1
        start_hp = int(obs.hp)
        start_max_hp = int(obs.max_hp)
        start_pots = potions_held(obs)
        start_cw = int(obs.combats_won)
        potion_hp_val = cfg.potion_hp_frac * max(start_max_hp, 1)

        if room == "Elite":
            elites_fought += 1
        if room == "Boss" and act == 1 and a2_boss_entry is None:
            a2_boss_entry = {"hp_pct": start_hp / max(start_max_hp, 1), "pots": start_pots}

        t_c0 = time.time()
        env.snapshot(0)
        greedy_res, obs_after_greedy = rollout_combat_with_inferer(
            env, obs, inferer, temp=0.0, rng=None, start_cw=start_cw, max_steps_fight=cfg.max_steps_fight
        )
        best_res = greedy_res
        best_idx = 0
        best_key = rank_key(greedy_res, potion_hp_val, is_greedy=True)
        last_idx = 0
        last_obs = obs_after_greedy

        max_extra, fight_timeout = budgets.get(room, (cfg.monster_trials, cfg.monster_timeout))
        if (not is_perfect_win(greedy_res, start_hp, start_pots)) and max_extra > 0:
            for ti in range(1, max_extra + 1):
                if fight_timeout > 0.0 and (time.time() - t_c0 >= fight_timeout):
                    break
                obs_r = env.restore(0)
                t_cur = temps[(ti - 1) % len(temps)]
                rng_t = np.random.default_rng(trial_rng_seed(seed, act, floor, fc_idx, ti))
                cand_res, obs_end = rollout_combat_with_inferer(
                    env, obs_r, inferer, temp=t_cur, rng=rng_t, start_cw=start_cw, max_steps_fight=cfg.max_steps_fight
                )
                last_idx = ti
                last_obs = obs_end
                k = rank_key(cand_res, potion_hp_val, is_greedy=False)
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
            for a in best_res["actions"]:
                obs_re = env.step(int(a))
            obs = obs_re

        run_steps += best_res["steps"]
        if best_res["stuck"]:
            stuck = True

        # Realized post-combat rewards attributed to the preceding macro decision
        nf = int(obs.floor)
        na = int(obs.act)
        if nf > prev_floor:
            add_reward(cfg.floor_reward * (nf - prev_floor))
            prev_floor = nf
        if na > prev_act:
            if prev_act == 0 and na >= 1:
                add_reward(cfg.act1_boss_reward)
            if prev_act <= 1 and na >= 2:
                add_reward(cfg.act2_boss_reward)
            prev_act = na
        # Act 3 Boss 1 cleared (went directly to Boss 2 or won)
        if room == "Boss" and act == 2 and best_res["won"]:
            if obs.done and obs.victory:
                add_reward(cfg.victory_reward + cfg.act3_boss2_reward)
            elif not a3_boss1_rewarded:
                add_reward(cfg.act3_boss_reward)
                a3_boss1_rewarded = True
        if stuck:
            break

    # Compute GAE along the realized macro chain (gamma=1.0, lam=cfg.macro_lam, terminal boot=0.0)
    n_m = len(m_obs)
    if n_m > 0:
        vals = np.array(m_val, dtype=np.float32)
        rews = np.array(m_rew, dtype=np.float32)
        next_v = np.append(vals[1:], 0.0).astype(np.float32)
        delta = rews + next_v - vals
        adv = np.zeros(n_m, dtype=np.float32)
        acc = 0.0
        for i in range(n_m - 1, -1, -1):
            acc = float(delta[i]) + cfg.macro_lam * acc
            adv[i] = acc
        ret = adv + vals
    else:
        adv = np.zeros(0, dtype=np.float32)
        ret = np.zeros(0, dtype=np.float32)

    return {
        "seed": seed,
        "win": bool(obs.done and obs.victory),
        "stuck": bool(stuck),
        "act": int(obs.act),
        "floor": int(obs.floor),
        "elites": elites_fought,
        "rest_heal": rest_heal,
        "rest_smith": rest_smith,
        "card_pick": card_pick,
        "card_skip": card_skip,
        "a2_boss_entry": a2_boss_entry,
        "obs": m_obs,
        "actions": np.array(m_act, dtype=np.int64),
        "logp": np.array(m_logp, dtype=np.float32),
        "values": np.array(m_val, dtype=np.float32),
        "adv": adv,
        "ret": ret,
        "multi": np.array(m_multi, dtype=bool),
        "is_map": np.array(m_is_map, dtype=bool),
        "total_return": float(sum(m_rew)),
    }


def save_compatible_ckpt(
    path: Path,
    net,
    ema_dict: dict[str, torch.Tensor],
    base_blob: dict,
    update_idx: int,
    total_macro_steps: int,
    total_runs: int,
    cfg: argparse.Namespace,
    opt_state: dict | None = None,
) -> None:
    """Save checkpoint in exact AlphaDefect format directly loadable by eval_combat_sl/eval_v1140/NecroPilot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    state_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    ema_cpu = {k: (ema_dict[k].to(v.dtype) if k in ema_dict else v).detach().cpu() for k, v in net.state_dict().items()}
    out_blob = {
        "version": int(base_blob["version"]) + update_idx,
        "base_version": int(base_blob.get("base_version", base_blob["version"])),
        "total_steps": int(base_blob.get("total_steps", 0)) + total_macro_steps,
        "base_total_steps": int(base_blob.get("base_total_steps", base_blob.get("total_steps", 0))),
        "macro_sl_update": update_idx,
        "macro_sl_runs": total_runs,
        "macro_sl_steps": total_macro_steps,
        "obs_format": base_blob.get("obs_format", 28),
        "catalog_version": base_blob.get("catalog_version"),
        "vocab_n": base_blob.get("vocab_n"),
        "content_names": base_blob.get("content_names"),
        "config": base_blob["config"],
        "macro_sl_config": vars(cfg),
        "state": state_cpu,
        "ema": ema_cpu,
        "ema_state": ema_cpu,
    }
    if opt_state is not None:
        out_blob["opt_state"] = opt_state
    torch.save(out_blob, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Macro-Only PPO Adaptation under Fixed In-Combat SL.")
    ap.add_argument("--init-ckpt", required=True, help="Frozen base checkpoint (e.g. v4879 latest.pt)")
    ap.add_argument("--init-weights", choices=("ema", "raw"), default="ema")
    ap.add_argument("--run-dir", required=True, help="Output training directory")
    ap.add_argument("--character", default="NECROBINDER")
    ap.add_argument("--asc", type=int, default=10)
    ap.add_argument("--updates", type=int, default=24, help="Number of macro-only PPO updates")
    ap.add_argument("--batch-runs", type=int, default=32, help="Completed full SL runs per PPO update")
    ap.add_argument("--workers", type=int, default=12, help="Concurrent Sts2Headless workers")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed-prefix", default="SLMAC")
    ap.add_argument("--seed", type=int, default=20261127)
    # Fixed combat SL budget (MUST match deployment/eval!)
    ap.add_argument("--monster-trials", type=int, default=4)
    ap.add_argument("--elite-trials", type=int, default=12)
    ap.add_argument("--boss-trials", type=int, default=20)
    ap.add_argument("--monster-timeout", type=float, default=60.0)
    ap.add_argument("--elite-timeout", type=float, default=120.0)
    ap.add_argument("--boss-timeout", type=float, default=180.0)
    ap.add_argument("--potion-hp-frac", type=float, default=0.20)
    # Macro PPO hyperparameters (conservative head-only fine-tuning)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--ent-coef-macro", type=float, default=0.02)
    ap.add_argument("--ent-coef-map", type=float, default=0.04)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--hl-coef", type=float, default=0.1)
    ap.add_argument("--hl-sigma", type=float, default=0.75)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--macro-lam", type=float, default=0.95)
    ap.add_argument("--macro-temp", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.80)
    ap.add_argument("--milestone-every", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="Resume from <run-dir>/latest.pt (optimizer, EMA, update counter)")
    ap.add_argument("--infer-bf16", action="store_true", help="Use BF16 for rollout sampling inference (default: FP32 matching --no-bf16 eval)")
    # Rewards (matching standard v0.114.0 config)
    ap.add_argument("--floor-reward", type=float, default=0.02)
    ap.add_argument("--act1-boss-reward", type=float, default=0.5)
    ap.add_argument("--act2-boss-reward", type=float, default=0.5)
    ap.add_argument("--act3-boss-reward", type=float, default=1.5)
    ap.add_argument("--act3-boss2-reward", type=float, default=2.0)
    ap.add_argument("--victory-reward", type=float, default=2.0)
    ap.add_argument("--potion-discard-penalty", type=float, default=0.1)
    ap.add_argument("--stall-turns", type=int, default=20)
    ap.add_argument("--max-steps-run", type=int, default=6000)
    ap.add_argument("--max-steps-fight", type=int, default=400)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    if (not a.infer_bf16) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(a), ensure_ascii=False, indent=2), encoding="utf-8")

    resume_ckpt = run_dir / "latest.pt"
    resuming = bool(a.resume and resume_ckpt.exists())
    if resuming:
        net, base_blob, src_key = load_net_weights(resume_ckpt, "raw")
        # Restore true base_version / base_total_steps so version stays monotonic on further saves
        if "base_version" in base_blob:
            base_blob["version"] = int(base_blob["base_version"])
        if "base_total_steps" in base_blob:
            base_blob["total_steps"] = int(base_blob["base_total_steps"])
    else:
        net, base_blob, src_key = load_net_weights(Path(a.init_ckpt), a.init_weights)
    net = net.to(device)
    net.fast = device.type == "cuda"

    # STRICT FREEZE: ONLY macro heads (`cand_mlp_m`, `value_mlp_m`, `value_hl_m`) are trainable!
    trainable = []
    frozen_n = 0
    for name, p in net.named_parameters():
        if name.startswith(MACRO_HEAD_PREFIXES):
            p.requires_grad_(True)
            trainable.append(p)
        else:
            p.requires_grad_(False)
            frozen_n += p.numel()
    train_n = sum(p.numel() for p in trainable)
    assert train_n > 0, "No trainable macro head parameters found!"

    opt = torch.optim.Adam(trainable, lr=a.lr, eps=1e-5)
    start_upd = 0
    total_macro_steps = 0
    total_runs = 0
    if resuming:
        start_upd = int(base_blob.get("macro_sl_update", 0))
        total_macro_steps = int(base_blob.get("macro_sl_steps", 0))
        total_runs = int(base_blob.get("macro_sl_runs", 0))
        if "opt_state" in base_blob:
            opt.load_state_dict(base_blob["opt_state"])
            for g in opt.param_groups:
                g["lr"] = a.lr
        ema_src = base_blob.get("ema") or base_blob.get("ema_state") or {}
        ema = {
            k: (ema_src[k].to(device=device, dtype=torch.float32).clone() if k in ema_src else v.detach().float().clone())
            for k, v in net.state_dict().items()
            if torch.is_floating_point(v) and k != "hl_beta"
        }
    else:
        ema = {k: v.detach().float().clone() for k, v in net.state_dict().items() if torch.is_floating_point(v) and k != "hl_beta"}

    print(
        f"{'Resumed' if resuming else 'Loaded'} {resume_ckpt if resuming else a.init_ckpt} "
        f"(base_v={base_blob['version']}, start_upd={start_upd}, src={src_key}) -> {device}\n"
        f"Trainable macro-only params: {train_n:,} ({len(trainable)} tensors) | Frozen trunk+combat params: {frozen_n:,}\n"
        f"Plan: updates {start_upd + 1}..{a.updates} x {a.batch_runs} SL runs = {max(0, a.updates - start_upd) * a.batch_runs} new SL runs | "
        f"SL budgets M={a.monster_trials} E={a.elite_trials} B={a.boss_trials} guards={a.monster_timeout}/{a.elite_timeout}/{a.boss_timeout}s pot={a.potion_hp_frac} | "
        f"infer_bf16={a.infer_bf16} learner_bf16={device.type == 'cuda' and torch.cuda.is_bf16_supported()}",
        flush=True,
    )
    net.eval().infer_ready()

    inferer = DualInferer(net, a.device, max_batch=max(1, a.workers), wait_ms=1.5, use_bf16=a.infer_bf16)
    log_path = run_dir / "log.jsonl"
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    t_global = time.time()

    try:
        for upd in range(start_upd + 1, a.updates + 1):
            t_u0 = time.time()
            # ---- 1. Parallel collection of `batch_runs` completed SL runs
            work_q: queue.Queue = queue.Queue()
            for ri in range(a.batch_runs):
                seed_str = f"{a.seed_prefix}_U{upd:03d}_R{ri:03d}"
                work_q.put((ri, seed_str))

            collected: list[dict] = []
            col_lock = threading.Lock()

            def col_worker() -> None:
                env = Sts2Env()
                try:
                    while True:
                        try:
                            ri, s_str = work_q.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            tr = collect_one_macro_run(env, s_str, a.seed + upd * 10000 + ri * 31, inferer, a)
                        except EngineError as err:
                            print(f"[warn] EngineError on {s_str}: {err}", flush=True)
                            continue
                        with col_lock:
                            collected.append(tr)
                finally:
                    env.close()

            threads = [threading.Thread(target=col_worker, daemon=True) for _ in range(min(a.workers, a.batch_runs))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            t_col = time.time() - t_u0
            if not collected:
                raise RuntimeError(f"Update {upd}: collected 0 runs!")

            # ---- 2. Concatenate realized macro decisions
            all_obs = [o for tr in collected for o in tr["obs"]]
            n_m = len(all_obs)
            actions = np.concatenate([tr["actions"] for tr in collected])
            old_logp = np.concatenate([tr["logp"] for tr in collected])
            old_val = np.concatenate([tr["values"] for tr in collected])
            adv = np.concatenate([tr["adv"] for tr in collected])
            ret = np.concatenate([tr["ret"] for tr in collected])
            multi = np.concatenate([tr["multi"] for tr in collected])
            is_map = np.concatenate([tr["is_map"] for tr in collected])

            # Normalize advantages over multi-choice macro steps
            adv_use = adv[multi]
            adv_mean = float(adv_use.mean()) if len(adv_use) else 0.0
            adv_std = float(adv_use.std()) if len(adv_use) > 1 else 1.0
            adv_norm = (adv - adv_mean) / max(adv_std, 1e-6)

            # Explained variance of macro value head before update
            var_y = float(np.var(ret)) if n_m > 1 else 0.0
            ev_macro = float(1.0 - np.var(ret - old_val) / var_y) if var_y > 1e-8 else 0.0

            rg_all = Ragged.from_obs(all_obs, np.full(n_m, MODE_RUN, dtype=np.int64))
            act_t = torch.from_numpy(actions).to(device)
            olp_t = torch.from_numpy(old_logp).to(device)
            adv_t = torch.from_numpy(adv_norm.astype(np.float32)).to(device)
            ret_t = torch.from_numpy(ret.astype(np.float32)).to(device)
            multi_t = torch.from_numpy(multi).to(device)
            map_t = torch.from_numpy(is_map).to(device)

            # ---- 3. Macro-Only PPO update (pause inference thread cleanly while updating weights)
            t_lrn0 = time.time()
            stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy_macro": 0.0, "entropy_map": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
            nb = 0
            epochs_done = 0

            with inferer.pause_context():
                net.train()
                for ep in range(a.epochs):
                    perm = np.random.permutation(n_m)
                    ep_kl, ep_nb = 0.0, 0
                    for start in range(0, n_m, a.minibatch):
                        idx = np.sort(perm[start : start + a.minibatch])
                        if len(idx) < 4:
                            continue
                        b = collate_ragged(rg_all.select(idx), attend=bool(net.cfg.macro_attend)).to(device)
                        idx_t = torch.from_numpy(idx).to(device)
                        A_b = adv_t[idx_t]
                        R_b = ret_t[idx_t]
                        a_b = act_t[idx_t]
                        olp_b = olp_t[idx_t]
                        m_b = multi_t[idx_t].float()
                        mp_b = (multi_t[idx_t] & map_t[idx_t]).float()

                        opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                            logits, v, _phi, ex = net.forward_all(b)
                        lg = logits.float()
                        valid = torch.isfinite(lg)
                        lg_fin = torch.where(valid, lg, torch.full_like(lg, -1e9))
                        logp_all = F.log_softmax(lg_fin, dim=-1)
                        lp = logp_all.gather(1, a_b[:, None]).squeeze(1)
                        logp_safe = torch.where(valid, logp_all, torch.zeros_like(logp_all))
                        p_all = torch.where(valid, logp_all.exp(), torch.zeros_like(logp_all))
                        ent_all = -(p_all * logp_safe).sum(-1)

                        ratio = torch.exp((lp - olp_b).clamp(-20.0, 20.0))
                        surr1 = ratio * A_b
                        surr2 = ratio.clamp(1.0 - a.clip, 1.0 + a.clip) * A_b
                        nm = m_b.sum().clamp(min=1.0)
                        l_pi = -(torch.minimum(surr1, surr2) * m_b).sum() / nm

                        ent_c = torch.where(mp_b > 0.5, a.ent_coef_map, a.ent_coef_macro)
                        ent_bonus = (ent_c * ent_all * m_b).sum() / nm

                        v_f = v.float()
                        if net.cfg.value_hl and "v_scalar" in ex and "hl_logits" in ex:
                            l_vs = F.mse_loss(ex["v_scalar"].float(), R_b)
                            l_hl = -(hl_target(R_b, net.hl_edges, a.hl_sigma) * F.log_softmax(ex["hl_logits"].float(), -1)).sum(-1).mean()
                            l_v = l_vs + a.hl_coef * l_hl
                        else:
                            l_v = F.mse_loss(v_f, R_b)

                        loss = l_pi + a.vf_coef * l_v - ent_bonus
                        loss.backward()
                        gnorm = torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm)
                        if torch.isfinite(gnorm) and all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable):
                            opt.step()
                        else:
                            print(f"[warn] U{upd} ep={ep} non-finite grad skipped (gnorm={float(gnorm)})", flush=True)
                            opt.zero_grad(set_to_none=True)

                        with torch.no_grad():
                            kl = (((ratio - 1.0) - (lp - olp_b)) * m_b).sum() / nm
                            cf = (((ratio - 1.0).abs() > a.clip).float() * m_b).sum() / nm
                            em = (ent_all * m_b).sum() / nm
                            nmp = mp_b.sum().clamp(min=1.0)
                            emp = (ent_all * mp_b).sum() / nmp if mp_b.sum() > 0 else em
                            stats["loss_pi"] += float(l_pi.item())
                            stats["loss_v"] += float(l_v.item())
                            stats["entropy_macro"] += float(em.item())
                            stats["entropy_map"] += float(emp.item())
                            stats["approx_kl"] += float(kl.item())
                            stats["clip_frac"] += float(cf.item())
                            ep_kl += float(kl.item())
                            ep_nb += 1
                            nb += 1
                    epochs_done += 1
                    if ep_nb > 0 and (ep_kl / ep_nb) > 1.5 * a.target_kl:
                        break

                # Update EMA (only macro heads actually changed, trunk stays identical)
                with torch.no_grad():
                    for k, v in net.state_dict().items():
                        if k in ema:
                            ema[k].lerp_(v.detach().float(), 1.0 - a.ema_decay)
                net.eval().infer_ready()

            t_lrn = time.time() - t_lrn0
            for k in stats:
                stats[k] = round(stats[k] / max(nb, 1), 5)

            total_macro_steps += n_m
            total_runs += len(collected)

            # ---- 4. Summarize & log
            floors = [tr["floor"] for tr in collected]
            n_r = len(collected)
            a2_rate = sum(1 for tr in collected if tr["act"] >= 1 or tr["win"]) / n_r
            a3_rate = sum(1 for tr in collected if tr["act"] >= 2 or tr["win"]) / n_r
            win_rate = sum(1 for tr in collected if tr["win"]) / n_r
            elites_mean = sum(tr["elites"] for tr in collected) / n_r
            rh = sum(tr["rest_heal"] for tr in collected)
            rs = sum(tr["rest_smith"] for tr in collected)
            cp = sum(tr["card_pick"] for tr in collected)
            cs = sum(tr["card_skip"] for tr in collected)
            a2e = [tr["a2_boss_entry"] for tr in collected if tr["a2_boss_entry"] is not None]
            a2_hp = round(100.0 * sum(x["hp_pct"] for x in a2e) / len(a2e), 1) if a2e else None
            a2_pot = round(sum(x["pots"] for x in a2e) / len(a2e), 2) if a2e else None

            rec = {
                "time": round(time.time(), 1),
                "update": upd,
                "runs": n_r,
                "total_runs": total_runs,
                "macro_steps": n_m,
                "total_macro_steps": total_macro_steps,
                "t_collect_s": round(t_col, 1),
                "t_learn_s": round(t_lrn, 2),
                "epochs": epochs_done,
                "mean_floor": round(statistics.mean(floors), 2),
                "max_floor": max(floors),
                "reach_act2": round(a2_rate, 4),
                "reach_act3": round(a3_rate, 4),
                "win_rate": round(win_rate, 4),
                "mean_return": round(sum(tr["total_return"] for tr in collected) / n_r, 4),
                "elites_per_run": round(elites_mean, 2),
                "rest_smith_frac": round(rs / max(rh + rs, 1), 3),
                "card_skip_frac": round(cs / max(cp + cs, 1), 3),
                "a2_boss_n": len(a2e),
                "a2_boss_hp_pct": a2_hp,
                "a2_boss_pots": a2_pot,
                "ev_macro": round(ev_macro, 4),
                **stats,
            }
            with log_path.open("a", encoding="utf-8") as lfh:
                lfh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(
                f"[U{upd:02d}/{a.updates:02d}] runs={total_runs:4d} msteps={n_m:4d} ({t_col:.0f}s+{t_lrn:.1f}s) | "
                f"fl={rec['mean_floor']:5.2f}(max {rec['max_floor']}) A2={a2_rate*100:4.1f}% A3={a3_rate*100:4.1f}% Win={win_rate*100:4.1f}% | "
                f"elites={elites_mean:.2f} smith={rec['rest_smith_frac']*100:4.1f}% skip={rec['card_skip_frac']*100:4.1f}% A2B_hp={a2_hp}% pot={a2_pot} | "
                f"ev={ev_macro:.3f} kl={stats['approx_kl']:.4f} ent={stats['entropy_macro']:.3f}",
                flush=True,
            )

            save_compatible_ckpt(
                run_dir / "latest.pt", net, ema, base_blob, upd, total_macro_steps, total_runs, a, opt_state=opt.state_dict()
            )
            if a.milestone_every > 0 and upd % a.milestone_every == 0:
                save_compatible_ckpt(
                    run_dir / "milestones" / f"u{upd:04d}.pt", net, ema, base_blob, upd, total_macro_steps, total_runs, a
                )

        print(f"\nDONE {a.updates} macro-only updates ({total_runs} full SL runs, {total_macro_steps:,} macro steps) in {(time.time()-t_global)/3600:.2f}h", flush=True)
    finally:
        inferer.close()


if __name__ == "__main__":
    main()
