#!/usr/bin/env python3
"""Joint In-Combat SL Expert Iteration + Out-of-Combat Macro PPO Trainer.

Architecture (from v4879 EMA):
  1. In every combat (A1F1 Monster through A3 Boss 2):
       - Run model's own greedy pass (candidate 0).
       - If not already a perfect win (zero net HP & potion loss), run fixed-budget
         decoupled-RNG SL search from combat-start snapshot(0).
       - If SL finds a STRICTLY BETTER verified winning trajectory (`rescued` or
         `improved_win`), replay that verified action sequence to advance the live
         run AND harvest `(combat_obs, verified_action)` pairs (zero extra GPU evals!).
       - Train combat policy (`cand_mlp` + shared encoder) via KL-regularized weighted
         cross-entropy on those verified better trajectories (ExIt distillation).
  2. Out-of-combat (MAP, CARD_REWARD, REST_SITE, SHOP, EVENT, REWARDS):
       - The live run carries the ACTUAL HP, potions, rewards, and deck from the single
         committed combat route (discarded SL branches never emit rewards or macro steps).
       - Sample out-of-combat decisions on-policy and train with PPO + value loss
         (scalar MSE + HL-Gauss) along the realized macro trajectory.
  3. Sampling inference defaults to FP32 (allow_tf32=False) matching `--no-bf16` eval;
     joint learner updates use CUDA BF16 autocast with guarded gradient clipping.
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
from train_macro_sl import (  # noqa: E402
    DualInferer,
    rollout_combat_with_inferer,
    save_compatible_ckpt,
)


def collect_one_exit_run(
    env: Sts2Env,
    seed: str,
    rng_seed: int,
    inferer: DualInferer,
    cfg: argparse.Namespace,
) -> dict:
    """Run one full A10 game collecting BOTH realized macro PPO data and SL-verified combat expert pairs."""
    obs = reset_full_run(env, cfg.character, cfg.asc, seed, cfg.stall_turns)
    rng_mac = np.random.default_rng(rng_seed ^ 0x5A5A5A5A)
    temps = (0.4, 0.7, 1.0, 0.5, 0.85, 1.15)
    budgets = {
        "Monster": (cfg.monster_trials, cfg.monster_timeout),
        "Elite": (cfg.elite_trials, cfg.elite_timeout),
        "Boss": (cfg.boss_trials, cfg.boss_timeout),
    }
    floor_combat_counts: dict[tuple[int, int], int] = {}

    # Realized out-of-combat macro trajectory buffers
    m_obs: list = []
    m_act: list[int] = []
    m_logp: list[float] = []
    m_val: list[float] = []
    m_rew: list[float] = []
    m_multi: list[bool] = []
    m_is_map: list[bool] = []

    # Verified better in-combat SL expert trajectory buffers
    c_obs: list = []
    c_act: list[int] = []
    c_weight: list[float] = []
    c_is_rescue: list[bool] = []

    # Telemetry
    elites_fought = 0
    rest_heal = 0
    rest_smith = 0
    card_pick = 0
    card_skip = 0
    a2_boss_entry: dict | None = None
    a3_boss1_rewarded = False
    combats_total = 0
    combats_rescued = 0
    combats_improved = 0
    hp_saved_total = 0
    pots_saved_total = 0
    extra_trials_total = 0
    anomaly_timeouts = 0

    prev_floor = int(obs.floor)
    prev_act = int(obs.act)
    run_steps = 0
    stuck = False

    def add_reward(val: float) -> None:
        if m_rew and val != 0.0:
            m_rew[-1] += val

    while not obs.done and not obs.stuck and obs.num_actions > 0 and run_steps < cfg.max_steps_run:
        if not obs.in_combat:
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

            dec = int(obs.decision)
            if n_act > 1 and a < obs.cand_ints.shape[0]:
                ak = int(obs.cand_ints[a, 0])
                if dec == int(DecisionKind.CARD_REWARD):
                    if ak == int(ActionKind.PICK_CARD_REWARD):
                        card_pick += 1
                    elif ak == int(ActionKind.CARD_REWARD_ALT):
                        card_skip += 1

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

            if dec == int(DecisionKind.REST_SITE) and ak_chosen == int(ActionKind.REST_OPTION):
                if int(obs.hp) > hp_before:
                    rest_heal += 1
                else:
                    rest_smith += 1

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

        # Combat: greedy pass + fixed-budget decoupled-RNG SL search
        combats_total += 1
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
                    anomaly_timeouts += 1
                    break
                obs_r = env.restore(0)
                t_cur = temps[(ti - 1) % len(temps)]
                rng_t = np.random.default_rng(trial_rng_seed(seed, act, floor, fc_idx, ti))
                cand_res, obs_end = rollout_combat_with_inferer(
                    env, obs_r, inferer, temp=t_cur, rng=rng_t, start_cw=start_cw, max_steps_fight=cfg.max_steps_fight
                )
                extra_trials_total += 1
                last_idx = ti
                last_obs = obs_end
                k = rank_key(cand_res, potion_hp_val, is_greedy=False)
                if k > best_key:
                    best_key = k
                    best_res = cand_res
                    best_idx = ti
                    if is_perfect_win(cand_res, start_hp, start_pots):
                        break

        rescued = (not greedy_res["won"]) and best_res["won"]
        improved_win = greedy_res["won"] and best_res["won"] and (best_idx != 0)
        # Only distill strictly better verified routes (rescued loss, OR improved HP/potion resource score)
        score_g = (greedy_res["end_hp"] + potion_hp_val * greedy_res["end_pots"]) if greedy_res["won"] else -1e9
        score_b = (best_res["end_hp"] + potion_hp_val * best_res["end_pots"]) if best_res["won"] else -1e9
        strictly_better_resource = improved_win and (score_b > score_g + 1e-6)
        harvest_expert = rescued or strictly_better_resource

        if harvest_expert:
            if rescued:
                combats_rescued += 1
                w_step = cfg.rescue_weight
            else:
                combats_improved += 1
                hp_saved_total += best_res["end_hp"] - greedy_res["end_hp"]
                pots_saved_total += best_res["end_pots"] - greedy_res["end_pots"]
                w_step = cfg.improve_weight
            # Replay verified winning action sequence from snapshot(0) — advances live engine AND records (obs, a*)
            obs_re = env.restore(0)
            for a_exp in best_res["actions"]:
                a_i = int(a_exp)
                if obs_re.num_actions > 1:
                    c_obs.append(obs_re)
                    c_act.append(a_i)
                    c_weight.append(w_step)
                    c_is_rescue.append(rescued)
                obs_re = env.step(a_i)
            obs = obs_re
        elif best_idx == last_idx:
            obs = last_obs
        else:
            obs_re = env.restore(0)
            for a_exp in best_res["actions"]:
                obs_re = env.step(int(a_exp))
            obs = obs_re

        run_steps += best_res["steps"]
        if best_res["stuck"]:
            stuck = True

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
        if room == "Boss" and act == 2 and best_res["won"]:
            if obs.done and obs.victory:
                add_reward(cfg.victory_reward + cfg.act3_boss2_reward)
            elif not a3_boss1_rewarded:
                add_reward(cfg.act3_boss_reward)
                a3_boss1_rewarded = True
        if stuck:
            break

    # Realized macro GAE
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
        "combats_total": combats_total,
        "combats_rescued": combats_rescued,
        "combats_improved": combats_improved,
        "hp_saved": hp_saved_total,
        "pots_saved": pots_saved_total,
        "extra_trials": extra_trials_total,
        "anomaly_timeouts": anomaly_timeouts,
        "m_obs": m_obs,
        "m_act": np.array(m_act, dtype=np.int64),
        "m_logp": np.array(m_logp, dtype=np.float32),
        "m_val": np.array(m_val, dtype=np.float32),
        "m_adv": adv,
        "m_ret": ret,
        "m_multi": np.array(m_multi, dtype=bool),
        "m_is_map": np.array(m_is_map, dtype=bool),
        "c_obs": c_obs,
        "c_act": np.array(c_act, dtype=np.int64),
        "c_weight": np.array(c_weight, dtype=np.float32),
        "c_is_rescue": np.array(c_is_rescue, dtype=bool),
        "total_return": float(sum(m_rew)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint In-Combat SL Expert Iteration + Out-of-Combat Macro PPO.")
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--init-weights", choices=("ema", "raw"), default="ema")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--character", default="NECROBINDER")
    ap.add_argument("--asc", type=int, default=10)
    ap.add_argument("--updates", type=int, default=8)
    ap.add_argument("--batch-runs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed-prefix", default="SLEXIT")
    ap.add_argument("--seed", type=int, default=20261128)
    # Fixed combat SL budget
    ap.add_argument("--monster-trials", type=int, default=4)
    ap.add_argument("--elite-trials", type=int, default=12)
    ap.add_argument("--boss-trials", type=int, default=20)
    ap.add_argument("--monster-timeout", type=float, default=60.0)
    ap.add_argument("--elite-timeout", type=float, default=120.0)
    ap.add_argument("--boss-timeout", type=float, default=180.0)
    ap.add_argument("--potion-hp-frac", type=float, default=0.20)
    # ExIt combat imitation + Macro PPO hyperparameters
    ap.add_argument("--lr-trunk", type=float, default=3e-5, help="LR for shared trunk & combat policy head")
    ap.add_argument("--lr-macro", type=float, default=1e-4, help="LR for out-of-combat macro heads")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--ent-coef-macro", type=float, default=0.02)
    ap.add_argument("--ent-coef-map", type=float, default=0.04)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--hl-coef", type=float, default=0.1)
    ap.add_argument("--hl-sigma", type=float, default=0.75)
    ap.add_argument("--exit-coef", type=float, default=1.0, help="Weight on combat SL expert CE loss")
    ap.add_argument("--combat-kl-coef", type=float, default=0.10, help="KL regularization to pre-update combat policy")
    ap.add_argument("--rescue-weight", type=float, default=2.0, help="Per-step weight for rescued-loss trajectories")
    ap.add_argument("--improve-weight", type=float, default=1.0, help="Per-step weight for improved-win trajectories")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--macro-lam", type=float, default=0.95)
    ap.add_argument("--macro-temp", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.85)
    ap.add_argument("--milestone-every", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--infer-bf16", action="store_true", help="Default FP32 rollout inference matching --no-bf16 eval")
    # Rewards
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
        if "base_version" in base_blob:
            base_blob["version"] = int(base_blob["base_version"])
        if "base_total_steps" in base_blob:
            base_blob["total_steps"] = int(base_blob["base_total_steps"])
    else:
        net, base_blob, src_key = load_net_weights(Path(a.init_ckpt), a.init_weights)
    net = net.to(device)
    net.fast = device.type == "cuda"

    macro_prefixes = ("cand_mlp_m.", "value_mlp_m.", "value_hl_m.")
    macro_params, trunk_params = [], []
    for name, p in net.named_parameters():
        p.requires_grad_(True)
        if name.startswith(macro_prefixes):
            macro_params.append(p)
        else:
            trunk_params.append(p)
    all_params = trunk_params + macro_params

    opt = torch.optim.AdamW(
        [
            {"params": trunk_params, "lr": a.lr_trunk, "weight_decay": 0.0},
            {"params": macro_params, "lr": a.lr_macro, "weight_decay": 0.0},
        ],
        eps=1e-5,
    )
    start_upd = 0
    total_steps_trained = 0
    total_runs = 0
    if resuming:
        start_upd = int(base_blob.get("macro_sl_update", 0))
        total_steps_trained = int(base_blob.get("macro_sl_steps", 0))
        total_runs = int(base_blob.get("macro_sl_runs", 0))
        if "opt_state" in base_blob:
            opt.load_state_dict(base_blob["opt_state"])
            opt.param_groups[0]["lr"] = a.lr_trunk
            opt.param_groups[1]["lr"] = a.lr_macro
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
        f"Joint Trainable Params: trunk+combat={sum(p.numel() for p in trunk_params):,} (lr={a.lr_trunk}) | "
        f"macro={sum(p.numel() for p in macro_params):,} (lr={a.lr_macro})\n"
        f"Plan: updates {start_upd + 1}..{a.updates} x {a.batch_runs} runs | "
        f"SL M={a.monster_trials} E={a.elite_trials} B={a.boss_trials} guards={a.monster_timeout}/{a.elite_timeout}/{a.boss_timeout}s | "
        f"infer_bf16={a.infer_bf16} learner_bf16={device.type == 'cuda' and torch.cuda.is_bf16_supported()}",
        flush=True,
    )

    net.eval().infer_ready()
    inferer = DualInferer(net, a.device, max_batch=max(1, a.workers), wait_ms=1.5, use_bf16=a.infer_bf16)
    log_path = run_dir / "log.jsonl"
    use_bf16_lrn = device.type == "cuda" and torch.cuda.is_bf16_supported()
    t_global = time.time()

    try:
        for upd in range(start_upd + 1, a.updates + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t_u0 = time.time()

            # ---- 1. Parallel collection
            work_q: queue.Queue = queue.Queue()
            for ri in range(a.batch_runs):
                work_q.put((ri, f"{a.seed_prefix}_U{upd:03d}_R{ri:03d}"))

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
                            tr = collect_one_exit_run(env, s_str, a.seed + upd * 10000 + ri * 31, inferer, a)
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

            # ---- 2. Assemble Macro PPO dataset + Combat SL Expert dataset
            all_m_obs = [o for tr in collected for o in tr["m_obs"]]
            n_m = len(all_m_obs)
            m_act = np.concatenate([tr["m_act"] for tr in collected])
            m_olp = np.concatenate([tr["m_logp"] for tr in collected])
            m_val = np.concatenate([tr["m_val"] for tr in collected])
            m_adv = np.concatenate([tr["m_adv"] for tr in collected])
            m_ret = np.concatenate([tr["m_ret"] for tr in collected])
            m_multi = np.concatenate([tr["m_multi"] for tr in collected])
            m_is_map = np.concatenate([tr["m_is_map"] for tr in collected])

            adv_use = m_adv[m_multi]
            adv_mean = float(adv_use.mean()) if len(adv_use) else 0.0
            adv_std = float(adv_use.std()) if len(adv_use) > 1 else 1.0
            adv_norm = (m_adv - adv_mean) / max(adv_std, 1e-6)
            var_y = float(np.var(m_ret)) if n_m > 1 else 0.0
            ev_macro = float(1.0 - np.var(m_ret - m_val) / var_y) if var_y > 1e-8 else 0.0

            rg_m = Ragged.from_obs(all_m_obs, np.full(n_m, MODE_RUN, dtype=np.int64))
            m_act_t = torch.from_numpy(m_act).to(device)
            m_olp_t = torch.from_numpy(m_olp).to(device)
            m_adv_t = torch.from_numpy(adv_norm.astype(np.float32)).to(device)
            m_ret_t = torch.from_numpy(m_ret.astype(np.float32)).to(device)
            m_multi_t = torch.from_numpy(m_multi).to(device)
            m_map_t = torch.from_numpy(m_is_map).to(device)

            all_c_obs = [o for tr in collected for o in tr["c_obs"]]
            n_c = len(all_c_obs)
            if n_c > 0:
                c_act = np.concatenate([tr["c_act"] for tr in collected])
                c_w = np.concatenate([tr["c_weight"] for tr in collected])
                c_resc = np.concatenate([tr["c_is_rescue"] for tr in collected])
                rg_c = Ragged.from_obs(all_c_obs, np.full(n_c, MODE_RUN, dtype=np.int64))
                c_act_t = torch.from_numpy(c_act).to(device)
                c_w_t = torch.from_numpy(c_w).to(device)

            # ---- 3. Joint Learning Update (Combat ExIt CE + KL anchor, Macro PPO + Value)
            t_lrn0 = time.time()
            stats = {
                "loss_exit_ce": 0.0, "combat_kl": 0.0, "exit_acc_pre": 0.0, "exit_acc_post": 0.0,
                "loss_pi": 0.0, "loss_v": 0.0, "entropy_macro": 0.0, "approx_kl_macro": 0.0, "clip_frac": 0.0,
                "grad_skipped": 0,
            }

            with inferer.pause_context():
                # Snapshot pre-update combat distributions (for exact KL regularization + pre-update agreement metric)
                old_c_logp_list: list[torch.Tensor] = []
                old_c_valid_list: list[torch.Tensor] = []
                c_chunks: list[np.ndarray] = []
                if n_c > 0:
                    net.eval()
                    pre_hit = 0
                    with torch.no_grad():
                        for start in range(0, n_c, a.minibatch):
                            idx_c = np.arange(start, min(start + a.minibatch, n_c), dtype=np.int64)
                            c_chunks.append(idx_c)
                            bc = collate_ragged(rg_c.select(idx_c), attend=bool(net.cfg.macro_attend)).to(device)
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16_lrn):
                                lg_c, _ = net.forward_infer(bc)
                            lg_c = lg_c.float()
                            v_c = torch.isfinite(lg_c)
                            lg_fin = torch.where(v_c, lg_c, torch.full_like(lg_c, -1e9))
                            lp_c = F.log_softmax(lg_fin, dim=-1)
                            pred = lg_fin.argmax(dim=-1)
                            pre_hit += int((pred == c_act_t[torch.from_numpy(idx_c).to(device)]).sum().item())
                            old_c_logp_list.append(lp_c.cpu())
                            old_c_valid_list.append(v_c.cpu())
                    stats["exit_acc_pre"] = round(pre_hit / n_c, 4)

                net.train()
                nb_m = 0
                nb_c = 0
                epochs_done = 0
                for ep in range(a.epochs):
                    # A. Combat SL Expert Iteration epoch (shuffled chunk order)
                    if n_c > 0:
                        chunk_order = np.random.permutation(len(c_chunks))
                        for ci in chunk_order:
                            idx_c = c_chunks[int(ci)]
                            if len(idx_c) < 2:
                                continue
                            bc = collate_ragged(rg_c.select(idx_c), attend=bool(net.cfg.macro_attend)).to(device)
                            idx_ct = torch.from_numpy(idx_c).to(device)
                            ac_b = c_act_t[idx_ct]
                            wc_b = c_w_t[idx_ct]
                            olp_c = old_c_logp_list[int(ci)].to(device)
                            val_c = old_c_valid_list[int(ci)].to(device)

                            opt.zero_grad(set_to_none=True)
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16_lrn):
                                lg_c, _, _, _ = net.forward_all(bc)
                            lg_c = lg_c.float()
                            lg_fin = torch.where(val_c, lg_c, torch.full_like(lg_c, -1e9))
                            lp_c = F.log_softmax(lg_fin, dim=-1)
                            nll = -lp_c.gather(1, ac_b[:, None]).squeeze(1)
                            l_ce = (nll * wc_b).sum() / wc_b.sum().clamp(min=1e-6)

                            p_old = torch.where(val_c, olp_c.exp(), torch.zeros_like(olp_c))
                            diff_lp = torch.where(val_c, olp_c - lp_c, torch.zeros_like(olp_c))
                            kl_c = (p_old * diff_lp).sum(-1).mean()

                            loss_c = a.exit_coef * l_ce + a.combat_kl_coef * kl_c
                            loss_c.backward()
                            gn = torch.nn.utils.clip_grad_norm_(all_params, a.max_grad_norm)
                            if torch.isfinite(gn):
                                opt.step()
                            else:
                                stats["grad_skipped"] += 1
                                opt.zero_grad(set_to_none=True)
                            with torch.no_grad():
                                stats["loss_exit_ce"] += float(l_ce.item())
                                stats["combat_kl"] += float(kl_c.item())
                                nb_c += 1

                    # B. Macro PPO epoch
                    perm = np.random.permutation(n_m)
                    ep_kl, ep_nb = 0.0, 0
                    for start in range(0, n_m, a.minibatch):
                        idx = np.sort(perm[start : start + a.minibatch])
                        if len(idx) < 4:
                            continue
                        b = collate_ragged(rg_m.select(idx), attend=bool(net.cfg.macro_attend)).to(device)
                        idx_t = torch.from_numpy(idx).to(device)
                        A_b = m_adv_t[idx_t]
                        R_b = m_ret_t[idx_t]
                        a_b = m_act_t[idx_t]
                        olp_b = m_olp_t[idx_t]
                        m_b = m_multi_t[idx_t].float()
                        mp_b = (m_multi_t[idx_t] & m_map_t[idx_t]).float()

                        opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16_lrn):
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

                        loss_m = l_pi + a.vf_coef * l_v - ent_bonus
                        loss_m.backward()
                        gn = torch.nn.utils.clip_grad_norm_(all_params, a.max_grad_norm)
                        if torch.isfinite(gn):
                            opt.step()
                        else:
                            stats["grad_skipped"] += 1
                            opt.zero_grad(set_to_none=True)

                        with torch.no_grad():
                            kl = (((ratio - 1.0) - (lp - olp_b)) * m_b).sum() / nm
                            cf = (((ratio - 1.0).abs() > a.clip).float() * m_b).sum() / nm
                            em = (ent_all * m_b).sum() / nm
                            stats["loss_pi"] += float(l_pi.item())
                            stats["loss_v"] += float(l_v.item())
                            stats["entropy_macro"] += float(em.item())
                            stats["approx_kl_macro"] += float(kl.item())
                            stats["clip_frac"] += float(cf.item())
                            ep_kl += float(kl.item())
                            ep_nb += 1
                            nb_m += 1
                    epochs_done += 1
                    if ep_nb > 0 and (ep_kl / ep_nb) > 1.5 * a.target_kl:
                        break

                # Measure post-update combat expert top-1 accuracy
                if n_c > 0:
                    net.eval()
                    post_hit = 0
                    with torch.no_grad():
                        for idx_c in c_chunks:
                            bc = collate_ragged(rg_c.select(idx_c), attend=bool(net.cfg.macro_attend)).to(device)
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16_lrn):
                                lg_c, _ = net.forward_infer(bc)
                            lg_fin = torch.where(torch.isfinite(lg_c.float()), lg_c.float(), torch.full_like(lg_c.float(), -1e9))
                            post_hit += int((lg_fin.argmax(dim=-1) == c_act_t[torch.from_numpy(idx_c).to(device)]).sum().item())
                    stats["exit_acc_post"] = round(post_hit / n_c, 4)

                with torch.no_grad():
                    for k, v in net.state_dict().items():
                        if k in ema:
                            ema[k].lerp_(v.detach().float(), 1.0 - a.ema_decay)
                net.eval().infer_ready()

            t_lrn = time.time() - t_lrn0
            vram_peak_gb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2) if device.type == "cuda" else 0.0

            for k in ("loss_exit_ce", "combat_kl"):
                stats[k] = round(stats[k] / max(nb_c, 1), 5)
            for k in ("loss_pi", "loss_v", "entropy_macro", "approx_kl_macro", "clip_frac"):
                stats[k] = round(stats[k] / max(nb_m, 1), 5)

            total_steps_trained += n_m + n_c
            total_runs += len(collected)

            n_r = len(collected)
            floors = [tr["floor"] for tr in collected]
            a2_rate = sum(1 for tr in collected if tr["act"] >= 1 or tr["win"]) / n_r
            a3_rate = sum(1 for tr in collected if tr["act"] >= 2 or tr["win"]) / n_r
            win_rate = sum(1 for tr in collected if tr["win"]) / n_r
            tot_comb = sum(tr["combats_total"] for tr in collected)
            tot_resc = sum(tr["combats_rescued"] for tr in collected)
            tot_imp = sum(tr["combats_improved"] for tr in collected)
            tot_hp = sum(tr["hp_saved"] for tr in collected)
            tot_pot = sum(tr["pots_saved"] for tr in collected)
            tot_trials = sum(tr["extra_trials"] for tr in collected)
            tot_anom = sum(tr["anomaly_timeouts"] for tr in collected)
            n_c_resc = int(c_resc.sum()) if n_c > 0 else 0

            t_tot = t_col + t_lrn
            runs_per_hr = round(n_r / max(t_tot, 1e-6) * 3600.0, 1)
            csteps_per_hr = round(n_c / max(t_tot, 1e-6) * 3600.0, 0)

            rec = {
                "time": round(time.time(), 1),
                "update": upd,
                "runs": n_r,
                "total_runs": total_runs,
                "macro_steps": n_m,
                "macro_multi_steps": int(m_multi.sum()),
                "combat_exit_steps": n_c,
                "combat_exit_rescue_steps": n_c_resc,
                "combats_total": tot_comb,
                "combats_rescued": tot_resc,
                "combats_improved": tot_imp,
                "extra_trials": tot_trials,
                "anomaly_timeouts": tot_anom,
                "t_collect_s": round(t_col, 1),
                "t_learn_s": round(t_lrn, 2),
                "learn_frac_pct": round(100.0 * t_lrn / max(t_tot, 1e-6), 2),
                "s_per_run": round(t_tot / n_r, 2),
                "runs_per_hr": runs_per_hr,
                "exit_steps_per_hr": csteps_per_hr,
                "vram_peak_gb": vram_peak_gb,
                "epochs": epochs_done,
                "mean_floor": round(statistics.mean(floors), 2),
                "max_floor": max(floors),
                "reach_act2": round(a2_rate, 4),
                "reach_act3": round(a3_rate, 4),
                "win_rate": round(win_rate, 4),
                "hp_saved_per_run": round(tot_hp / n_r, 1),
                "pots_saved_per_run": round(tot_pot / n_r, 2),
                "ev_macro": round(ev_macro, 4),
                **stats,
            }
            with log_path.open("a", encoding="utf-8") as lfh:
                lfh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(
                f"[U{upd:02d}/{a.updates:02d}] runs={n_r:2d} ({t_col:.1f}s col + {t_lrn:.2f}s lrn = {rec['s_per_run']:.2f}s/run, {runs_per_hr:.0f} run/h, VRAM={vram_peak_gb:.2f}GB) | "
                f"fl={rec['mean_floor']:5.2f} A2={a2_rate*100:4.1f}% A3={a3_rate*100:4.1f}% Win={win_rate*100:4.1f}% | "
                f"ExIt: {tot_resc}resc+{tot_imp}imp/{tot_comb}c -> {n_c} steps ({n_c/n_r:.0f}/run, acc {stats['exit_acc_pre']:.3f}->{stats['exit_acc_post']:.3f}, ce={stats['loss_exit_ce']:.3f}, c_kl={stats['combat_kl']:.4f}) | "
                f"Mac: {n_m} steps (ev={ev_macro:.3f}, m_kl={stats['approx_kl_macro']:.4f}, ent={stats['entropy_macro']:.3f}) | anom={tot_anom} skip={stats['grad_skipped']}",
                flush=True,
            )

            save_compatible_ckpt(
                run_dir / "latest.pt", net, ema, base_blob, upd, total_steps_trained, total_runs, a, opt_state=opt.state_dict()
            )
            if a.milestone_every > 0 and upd % a.milestone_every == 0:
                save_compatible_ckpt(
                    run_dir / "milestones" / f"u{upd:04d}.pt", net, ema, base_blob, upd, total_steps_trained, total_runs, a
                )

        print(f"\nDONE {a.updates} ExIt+MacroSL updates ({total_runs} runs) in {(time.time()-t_global)/60:.1f} min", flush=True)
    finally:
        inferer.close()


if __name__ == "__main__":
    main()
