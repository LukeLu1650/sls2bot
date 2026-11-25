"""Evaluate multiple PPO v2 checkpoints on the same strictly unseen A0/Ascension seed panel."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
from threading import Lock
import time

import torch

from beta_build import CHECKOUT, verify_host
from beta_eval import wilson, write_json
from ppo_v2_policy import InferenceBatcher, ModelConfig, PPOv2Policy
from train_ppo_v2 import play_episode
from train_strategy import panel


def collect_seen_seeds(runs_root: Path) -> set[str]:
    seen: set[str] = set()
    if not runs_root.exists():
        return seen
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        proto = d / "protocol.json"
        if proto.exists():
            p = json.loads(proto.read_text(encoding="utf-8"))
            seen.update(p.get("development_seeds") or [])
            seen.update(p.get("evaluation_seeds") or [])
        for rf in d.glob("round-*.json"):
            rd = json.loads(rf.read_text(encoding="utf-8"))
            for r in rd.get("train_rows") or []:
                if r.get("seed"):
                    seen.add(r["seed"])
    return seen


def load_model(ckpt_path: Path, device: str) -> tuple[PPOv2Policy, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_cfg = dict(ckpt["model_config"])
    raw_cfg.setdefault("value_bins", 1)
    raw_cfg.setdefault("value_symlog_max", 2.6)
    cfg = ModelConfig(**raw_cfg)
    model = PPOv2Policy(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def detailed_summary(rows: list[dict]) -> dict:
    n = len(rows)
    wins = sum(r["victory"] is True for r in rows)
    a1_clears = sum(bool(r["act1_cleared"]) for r in rows)
    a2_clears = sum(bool(r["act2_cleared"]) for r in rows)

    a1_boss = [f for r in rows for f in r.get("fights", []) if f["room"] == "Boss" and f["act"] == 1]
    a2_boss = [f for r in rows for f in r.get("fights", []) if f["room"] == "Boss" and f["act"] == 2]
    a3_boss = [f for r in rows for f in r.get("fights", []) if f["room"] == "Boss" and f["act"] == 3]

    a1_boss_kills = sum(1 for f in a1_boss if f.get("hp_out") and f["hp_out"] > 0)
    a2_boss_kills = sum(1 for f in a2_boss if f.get("hp_out") and f["hp_out"] > 0)

    def avg_boss_dmg(bf):
        vals = [max(0.0, 1.0 - f["enemy_min_hp"] / f["enemy_initial_hp"]) for f in bf if f.get("enemy_initial_hp")]
        return float(statistics.mean(vals)) if vals else 0.0

    a1_norm = [f for r in rows for f in r.get("fights", []) if f["act"] == 1 and f["room"] == "Monster"]
    a2_norm = [f for r in rows for f in r.get("fights", []) if f["act"] == 2 and f["room"] == "Monster"]
    a1_norm_loss = [f["hp_in"] - f["hp_out"] for f in a1_norm if f.get("hp_out") and f["hp_out"] > 0]
    a2_norm_loss = [f["hp_in"] - f["hp_out"] for f in a2_norm if f.get("hp_out") and f["hp_out"] > 0]
    a1_norm_deaths = sum(1 for f in a1_norm if f.get("hp_out") == 0)
    a2_norm_deaths = sum(1 for f in a2_norm if f.get("hp_out") == 0)

    a1_elites = [f for r in rows for f in r.get("fights", []) if f["act"] == 1 and f["room"] == "Elite"]
    a2_elites = [f for r in rows for f in r.get("fights", []) if f["act"] == 2 and f["room"] == "Elite"]
    elites_won = sum(1 for r in rows for f in r.get("fights", []) if f["room"] == "Elite" and f.get("hp_out") and f["hp_out"] > 0)
    a1_elites_won = sum(1 for f in a1_elites if f.get("hp_out") and f["hp_out"] > 0)
    a1_elites_deaths = sum(1 for f in a1_elites if f.get("hp_out") == 0)
    a2_elites_fought = len(a2_elites)
    a2_elites_won = sum(1 for f in a2_elites if f.get("hp_out") and f["hp_out"] > 0)
    a2_elites_deaths = sum(1 for f in a2_elites if f.get("hp_out") == 0)

    a1_boss_deaths = sum(1 for f in a1_boss if f.get("hp_out") == 0)
    a2_boss_deaths = sum(1 for f in a2_boss if f.get("hp_out") == 0)
    a2_total_deaths = sum(1 for r in rows if r.get("status") == "defeat" and r.get("act") == 2)
    a2_other_deaths = max(0, a2_total_deaths - a2_norm_deaths - a2_elites_deaths - a2_boss_deaths)

    def summarize_entries(sub_rows: list[dict]) -> dict | None:
        ents = [r["act2_entry"] for r in sub_rows if r.get("act2_entry")]
        if not ents:
            return None
        return {
            "count": len(ents),
            "med_deck_size": float(statistics.median(e["deck_size"] for e in ents)),
            "mean_deck_size": round(float(statistics.mean(e["deck_size"] for e in ents)), 2),
            "med_upgraded": float(statistics.median(e["upgraded"] for e in ents)),
            "med_basics": float(statistics.median(e["basics"] for e in ents)),
            "med_curses": float(statistics.median(e["curses"] for e in ents)),
            "med_relics": float(statistics.median(len(e["relics"]) for e in ents)),
            "med_potions": float(statistics.median(len(e["potions"]) for e in ents)),
            "med_hp": float(statistics.median(e["hp"] for e in ents if e.get("hp") is not None)),
            "med_gold": float(statistics.median(e["gold"] for e in ents if e.get("gold") is not None)),
        }

    a2_rows = [r for r in rows if r.get("act1_cleared")]
    a2_reached_boss_rows = [r for r in a2_rows if any(f["act"] == 2 and f["room"] == "Boss" for f in r.get("fights", []))]
    a2_died_early_rows = [r for r in a2_rows if r not in a2_reached_boss_rows and not r.get("act2_cleared")]

    prog = [51 if r["victory"] is True else ((r["act"] or 1) - 1) * 17 + min(r["floor"] or 0, 17) for r in rows]

    acts = {
        a: round(float(statistics.mean(r["actions"].get(a, 0) for r in rows)), 3)
        for a in ("select_card_reward", "skip_card_reward", "remove_card", "buy_card", "buy_relic", "buy_potion", "use_potion")
    }
    take, skip = acts["select_card_reward"], acts["skip_card_reward"]
    acts["skip_rate"] = round(skip / max(1e-6, take + skip), 4)

    return {
        "attempts": n,
        "statuses": dict(Counter(r["status"] for r in rows)),
        "wins": wins,
        "win_rate": wins / n,
        "a1_boss_reach": len(a1_boss),
        "a1_boss_reach_rate": len(a1_boss) / n,
        "a1_boss_kills": a1_boss_kills,
        "a1_boss_deaths": a1_boss_deaths,
        "a1_boss_kill_given_reach": a1_boss_kills / max(1, len(a1_boss)),
        "a1_clears": a1_clears,
        "a1_clear_rate": a1_clears / n,
        "a1_clear_wilson95": wilson(a1_clears, n),
        "a1_boss_med_hp_in": float(statistics.median(f["hp_in"] for f in a1_boss)) if a1_boss else None,
        "a1_boss_avg_dmg": avg_boss_dmg(a1_boss),
        "a2_boss_reach": len(a2_boss),
        "a2_boss_reach_given_a2": len(a2_boss) / max(1, a1_clears),
        "a2_boss_kills": a2_boss_kills,
        "a2_boss_deaths": a2_boss_deaths,
        "a2_clears": a2_clears,
        "a2_clear_rate": a2_clears / n,
        "a2_boss_med_hp_in": float(statistics.median(f["hp_in"] for f in a2_boss)) if a2_boss else None,
        "a2_boss_avg_dmg": avg_boss_dmg(a2_boss),
        "a3_boss_reach": len(a3_boss),
        "elites_won": elites_won,
        "a1_elites_won": a1_elites_won,
        "a1_elites_deaths": a1_elites_deaths,
        "a2_elites_fought": a2_elites_fought,
        "a2_elites_won": a2_elites_won,
        "a2_elites_deaths": a2_elites_deaths,
        "a1_normal_fought": len(a1_norm),
        "a1_normal_deaths": a1_norm_deaths,
        "a1_med_normal_hp_loss": float(statistics.median(a1_norm_loss)) if a1_norm_loss else None,
        "a2_normal_fought": len(a2_norm),
        "a2_normal_deaths": a2_norm_deaths,
        "a2_other_deaths": a2_other_deaths,
        "a2_med_normal_hp_loss": float(statistics.median(a2_norm_loss)) if a2_norm_loss else None,
        "a2_entry_all": summarize_entries(a2_rows),
        "a2_entry_reached_boss": summarize_entries(a2_reached_boss_rows),
        "a2_entry_died_early": summarize_entries(a2_died_early_rows),
        "median_progress": float(statistics.median(prog)),
        "mean_progress": float(statistics.mean(prog)),
        "strategic_actions": acts,
    }


def parse_checkpoint_spec(
    spec: str,
    default_combat_temp: float | None,
    default_strategy_temp: float | None,
) -> tuple[str, Path, float | None, float | None]:
    """Parse 'label=path' or 'label=path@combat_temp,strategy_temp'."""
    label, rhs = spec.split("=", 1)
    c_temp = default_combat_temp
    s_temp = default_strategy_temp
    if "@" in rhs:
        path_str, temp_str = rhs.rsplit("@", 1)
        parts = [p.strip() for p in temp_str.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected '@combat_temp,strategy_temp' in {spec!r}")
        c_temp, s_temp = float(parts[0]), float(parts[1])
    else:
        path_str = rhs
    return label, Path(path_str), c_temp, s_temp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seeds-from", type=Path, default=None, help="Lock exact seed list from an existing benchmark JSON")
    parser.add_argument("--combat-temp", type=float, default=None, help="Optional default stochastic combat temperature")
    parser.add_argument("--strategy-temp", type=float, default=None, help="Optional default stochastic strategy temperature")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--namespace", type=str, default="necro-ppo-v2-unseen-bench-v1")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("checkpoints", nargs="+", help="label=path[@c_temp,s_temp], e.g. S5-R16@lowT=runs/.../best-checkpoint.pt@0.32,0.42")
    args = parser.parse_args()

    torch.set_num_threads(1 if args.workers > 1 else 2)
    torch.set_num_interop_threads(1)
    verify_host(steam=True)
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")

    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    if args.seeds_from is not None:
        seed_doc = json.loads(args.seeds_from.read_text(encoding="utf-8"))
        seeds = list(seed_doc["seeds"])[:args.games]
        print(f"Loaded {len(seeds)} locked A{args.ascension} seeds from {args.seeds_from}.", flush=True)
    else:
        seen = collect_seen_seeds(args.runs_root)
        candidates = panel("audit", args.games * 4, args.namespace)
        seeds = [s for s in candidates if s not in seen][:args.games]
        assert len(seeds) == args.games and len(set(seeds) & seen) == 0
        print(f"Prepared {len(seeds)} strictly unseen A{args.ascension} seeds (excluded {len(seen)} historical seeds).", flush=True)

    results = {
        "ascension": args.ascension,
        "games": len(seeds),
        "seeds_from": str(args.seeds_from) if args.seeds_from else None,
        "seeds": seeds,
        "models": {},
    }
    diag_dir = args.out.parent / (args.out.stem + "-diagnostics")
    diag_lock = Lock()

    for spec in args.checkpoints:
        label, ckpt_path, c_temp, s_temp = parse_checkpoint_spec(spec, args.combat_temp, args.strategy_temp)
        stochastic = c_temp is not None and s_temp is not None and (c_temp > 0 or s_temp > 0)
        model, ckpt = load_model(ckpt_path, args.device)
        lock = Lock()
        t0 = time.perf_counter()
        with InferenceBatcher(model, max_batch=max(16, args.workers * 2), max_wait_ms=1.5) as batcher:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rows = list(ex.map(
                    lambda s: play_episode(
                        model, lock, progress, s, ascension=args.ascension, training=False,
                        sample_eval=stochastic,
                        combat_temp=c_temp if stochastic else 0.7,
                        strategy_temp=s_temp if stochastic else 0.9,
                        diag_dir=diag_dir, diag_lock=diag_lock, progress_path=args.progress,
                        batcher=batcher,
                    )[0],
                    seeds,
                ))
        dt = time.perf_counter() - t0
        summary = detailed_summary(rows)
        results["models"][label] = {
            "checkpoint": str(ckpt_path),
            "round": ckpt.get("round"),
            "value_bins": model.cfg.value_bins,
            "sampling": {"combat_temp": c_temp, "strategy_temp": s_temp} if stochastic else "greedy",
            "elapsed_seconds": round(dt, 1),
            "summary": summary,
            "rows": rows,
        }
        print(
            f"[{label}] ({dt:.1f}s) st={summary['statuses']} | "
            f"A1Boss reach={summary['a1_boss_reach']}/{len(seeds)} kill={summary['a1_boss_kills']}/{summary['a1_boss_reach']} ({100*summary['a1_boss_kill_given_reach']:.1f}%) | "
            f"A1clr={summary['a1_clears']}/{len(seeds)} ({100*summary['a1_clear_rate']:.1f}%) | "
            f"A2Deaths(N/E/B/O)={summary['a2_normal_deaths']}/{summary['a2_elites_deaths']}/{summary['a2_boss_deaths']}/{summary['a2_other_deaths']} "
            f"A2Boss={summary['a2_boss_reach']}/{max(1,summary['a1_clears'])} A2clr={summary['a2_clears']} wins={summary['wins']} | "
            f"mProg={summary['mean_progress']:.2f}",
            flush=True,
        )

    write_json(args.out, results)
    print(f"\nSaved full benchmark report to {args.out}", flush=True)


if __name__ == "__main__":
    main()
