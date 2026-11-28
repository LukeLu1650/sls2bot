#!/usr/bin/env python3
"""Paper-grade locked A10 evaluator & milestone batch runner for game-v0.114.0.

Reuses sts2rl.evaluator.play with:
  - Explicit weight selection: --weights ema (default primary) or raw
  - Dynamic checkpoint metadata (reads version & total_steps from .pt; works for both milestones/*.pt and latest.pt)
  - Strict separation of eval seeds (A10EVAL*) vs independent Boss-mining seeds (A10MINE*)
  - Resumable per-checkpoint JSONL output + clean summary table with exact Boss reached/killed counts
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AD_PY = ROOT / "external" / "AlphaDefect-Public" / "game-v0.114.0" / "python"
if str(AD_PY) not in sys.path:
    sys.path.insert(0, str(AD_PY))

import torch  # noqa: E402

from sts2rl.actor import MODE_RUN  # noqa: E402
from sts2rl.catalog import Catalog  # noqa: E402
from sts2rl.evaluator import play, run_pos  # noqa: E402
from sts2rl.model import ModelConfig, PolicyValueNet, load_state, zero_new_vocab  # noqa: E402

VOCAB_N_LEGACY = 1873


def load_net_weights(path: Path, weight_mode: str = "ema") -> tuple[PolicyValueNet, dict, str]:
    """Load model weights cleanly from either milestone (*.pt with ema_state) or latest.pt (with ema)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if weight_mode == "ema":
        state = blob.get("ema_state") or blob.get("ema")
        if state is None:
            raise ValueError(f"{path} has neither 'ema_state' nor 'ema' weights; pass --weights raw if intended.")
        src_key = "ema_state" if "ema_state" in blob else "ema"
    elif weight_mode == "raw":
        state = blob["state"]
        src_key = "state"
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    load_state(net, state, blob.get("content_names"))
    old_n = blob.get("vocab_n") or VOCAB_N_LEGACY
    vocab_n = Catalog().vocab_size
    if old_n < vocab_n:
        with torch.no_grad():
            zero_new_vocab(net, old_n, vocab_n)
    return net, blob, src_key


def boss_outcome(r: dict) -> dict:
    """Extract per-act Boss reach and kill flags from an evaluator.play run result.

    Note: evaluator.play records pre-combat entry dicts in r['combats'] (no per-combat 'win' field),
    so boss kills are determined exactly from run act progression, victory flag, and Act 3 Boss count.
    """
    combats = r.get("combats") or []
    run_act = int(r.get("act", 0))
    run_win = bool(r.get("win"))
    a1_reach = any(c.get("room") == "Boss" and int(c.get("act", -1)) == 0 for c in combats)
    a2_reach = any(c.get("room") == "Boss" and int(c.get("act", -1)) == 1 for c in combats)
    a3_bosses = [c for c in combats if c.get("room") == "Boss" and int(c.get("act", -1)) == 2]
    a3_b1_reach = len(a3_bosses) >= 1
    a3_b2_reach = len(a3_bosses) >= 2
    return {
        "a1_reach": a1_reach,
        "a1_kill": a1_reach and (run_act >= 1 or run_win),
        "a2_reach": a2_reach,
        "a2_kill": a2_reach and (run_act >= 2 or run_win),
        "a3_b1_reach": a3_b1_reach,
        "a3_b1_kill": a3_b1_reach and (a3_b2_reach or run_win),
        "a3_b2_reach": a3_b2_reach,
        "a3_b2_kill": a3_b2_reach and run_win,
    }


def summarize_runs(rows: list[dict], wall_s: float | None = None) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    wins = sum(1 for r in rows if r.get("win"))
    stuck = sum(1 for r in rows if r.get("stuck"))
    floors = [int(r.get("floor", 0)) for r in rows]
    acts = [int(r.get("act", 0)) for r in rows]
    # Act indexing in Sts2Headless: act 0 = Act 1, act 1 = Act 2, act 2 = Act 3
    reach_a2 = sum(1 for r in rows if int(r.get("act", 0)) >= 1 or r.get("win"))
    reach_a3 = sum(1 for r in rows if int(r.get("act", 0)) >= 2 or r.get("win"))

    b = [boss_outcome(r) for r in rows]
    def _cnt(rk: str, kk: str) -> tuple[int, int]:
        return sum(1 for x in b if x[kk]), sum(1 for x in b if x[rk])

    a1k, a1r = _cnt("a1_reach", "a1_kill")
    a2k, a2r = _cnt("a2_reach", "a2_kill")
    a31k, a31r = _cnt("a3_b1_reach", "a3_b1_kill")
    a32k, a32r = _cnt("a3_b2_reach", "a3_b2_kill")

    return {
        "n": n,
        "wins": wins,
        "win_rate": wins / n,
        "stuck": stuck,
        "mean_floor": round(statistics.mean(floors), 2),
        "median_floor": statistics.median(floors),
        "max_floor": max(floors),
        "reach_act2": reach_a2,
        "reach_act2_rate": reach_a2 / n,
        "reach_act3": reach_a3,
        "reach_act3_rate": reach_a3 / n,
        "boss1": [a1k, a1r],
        "boss2": [a2k, a2r],
        "boss3_1": [a31k, a31r],
        "boss3_2": [a32k, a32r],
        "wall_s": round(wall_s, 1) if wall_s is not None else None,
        "games_per_hour": round(n / (wall_s / 3600.0), 1) if wall_s and wall_s > 0 else None,
    }


def extract_boss_snaps(rows: list[dict], version: int, weight_mode: str, character: str, asc: int, failed_only: bool = True) -> list[dict]:
    """Export pre-boss snapshots from Act 1 / 2 / 3 Bosses (either failed-only for mining, or all-reached for Handover Probe)."""
    out = []
    for r in rows:
        if r.get("stuck"):
            continue
        if failed_only and r.get("win"):
            continue
        seed = r["seed"]
        bo = boss_outcome(r)
        candidates = [
            (0, "boss1", bo["a1_reach"], bo["a1_kill"]),
            (1, "boss2", bo["a2_reach"], bo["a2_kill"]),
            (2, "boss3", bo["a3_b1_reach"], bo["a3_b1_kill"]),
        ]
        for act, key, reached, killed in candidates:
            if not reached:
                continue
            if failed_only and killed:
                continue
            bs = r.get(key)
            if not bs or not bs.get("snap"):
                continue
            out.append({
                "id": f"v{version}_{weight_mode}_{seed}_a{act+1}",
                "run_seed": seed,
                "version": version,
                "weights": weight_mode,
                "character": character,
                "ascension": asc,
                "act": act,
                "orig_boss_win": bool(killed),
                "encounter": bs["encounter"],
                "next_encounter": bs.get("next_encounter"),
                "hp": bs.get("hp"),
                "max_hp": bs.get("max_hp"),
                "potions": bs.get("potions"),
                "upgrades": bs.get("upgrades"),
                "deck": bs.get("deck"),
                "snap": bs["snap"],
            })
    return out


def resolve_ckpts(specs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for s in specs:
        matches = sorted(glob.glob(s))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            p = Path(s)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {s}")
            paths.append(p)
    # Deduplicate while preserving order, then sort by version metadata? Keep user order after dedup.
    seen, uniq = set(), []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def load_seeds(runs: int, seed_prefix: str, seeds_file: str | None, seed_start: int = 0) -> list[str]:
    if seeds_file:
        raw = json.loads(Path(seeds_file).read_text(encoding="utf-8"))
        seeds = raw["seeds"] if isinstance(raw, dict) else raw
        return [str(s) for s in seeds[seed_start : seed_start + runs]]
    return [f"{seed_prefix}{i:04d}" for i in range(seed_start, seed_start + runs)]


def eval_one_ckpt(ckpt_path: Path, a: argparse.Namespace, seeds: list[str], out_dir: Path) -> tuple[dict, list[dict]]:
    net, blob, src_key = load_net_weights(ckpt_path, a.weights)
    version = int(blob["version"])
    total_steps = int(blob.get("total_steps", 0))
    tag = f"v{version:06d}_{a.weights}"
    ckpt_dir = out_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_path = ckpt_dir / "runs.jsonl"

    done_rows: dict[str, dict] = {}
    if res_path.exists():
        for ln in res_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                done_rows[r["seed"]] = r

    todo = [s for s in seeds if s not in done_rows]
    t0 = time.time()
    if todo:
        net = net.to(a.device)
        net.infer_ready()
        req_extra = {"loop_cut": True, "stall_turns": a.stall_turns}
        caps = {"run": a.max_steps_run, "fight": a.max_steps_fight}
        it = iter(todo)

        def next_task():
            s = next(it, None)
            if s is None:
                return None
            return (
                {
                    "mode": "run",
                    "character": a.character,
                    "ascension": a.asc,
                    "seed": s,
                    "snapshots": True,
                    "route_summary": True,
                    "tag": s,
                    **req_extra,
                },
                MODE_RUN,
                "run",
            )

        fh = res_path.open("a", encoding="utf-8")
        try:
            def on_result(r: dict) -> None:
                r = {k: v for k, v in r.items() if k != "kind"}
                seed = r.pop("tag")
                rec = {
                    "asc": a.asc,
                    "seed": seed,
                    "character": a.character,
                    "version": version,
                    "total_steps": total_steps,
                    "weights": a.weights,
                    "weight_key": src_key,
                    "pos": run_pos(r),
                    **r,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done_rows[seed] = rec

            policy = {
                "temp": 0.0,
                "explore": None,
                "seed": 1,
                "boss3_snap": True,
                "boss_snaps": bool(a.boss_snaps),
                "merge": bool(a.merge_equiv),
            }
            stop = threading.Event()
            play(
                net,
                a.device,
                next_task,
                a.envs,
                stop,
                None,
                caps,
                str(ckpt_dir / "loops.txt"),
                version,
                replay_all=bool(a.replays),
                on_result=on_result,
                policy=policy,
                time_cap=a.time_cap,
            )
        finally:
            fh.close()
        del net
        if str(a.device).startswith("cuda"):
            torch.cuda.empty_cache()

    wall_s = time.time() - t0 if todo else None
    ordered = [done_rows[s] for s in seeds if s in done_rows]
    summary = {
        "ckpt": str(ckpt_path),
        "version": version,
        "total_steps": total_steps,
        "weights": a.weights,
        "weight_key": src_key,
        "seed_prefix": a.seed_prefix if not a.seeds_file else Path(a.seeds_file).name,
        "newly_run": len(todo),
        **summarize_runs(ordered, wall_s),
    }
    return summary, ordered


def _fmt_ratio(pair: list[int]) -> str:
    k, r = pair
    return f"{k}/{r} ({k/r*100:4.1f}%)" if r > 0 else "0/0 (  - )"


def main() -> None:
    ap = argparse.ArgumentParser(description="Locked A10 evaluator & milestone batch runner for game-v0.114.0.")
    ap.add_argument("--ckpts", nargs="+", required=True, help="Checkpoint file(s) or glob(s), e.g. milestones/*.pt latest.pt")
    ap.add_argument("--out", required=True, help="Output directory for resumable runs.jsonl and summary.json")
    ap.add_argument("--character", default="NECROBINDER")
    ap.add_argument("--asc", type=int, default=10)
    ap.add_argument("--weights", choices=("ema", "raw"), default="ema", help="Primary weight source: ema (default) or raw")
    ap.add_argument("--runs", type=int, default=128)
    ap.add_argument("--seed-start", type=int, default=0, help="Starting index for generated seeds (e.g. 512 -> A10MINE0512..)")
    ap.add_argument("--seed-prefix", default="A10EVAL", help="Use A10EVAL for locked eval; A10MINE for independent Boss-mining")
    ap.add_argument("--seeds-file", default=None, help="Optional JSON seed list file (overrides --seed-prefix)")
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--boss-snaps", action="store_true", help="Record Act 1 & Act 2 pre-boss snapshots (Act 3 is always recorded)")
    ap.add_argument("--export-failed-bosses", default=None, help="Export failed Boss snapshots to JSONL for mining (requires non-EVAL prefix unless --allow-diag-export)")
    ap.add_argument("--export-all-bosses", default=None, help="Export ALL reached Boss-entry snapshots (wins+losses) for read-only Handover Probe diagnostics")
    ap.add_argument("--allow-diag-export", action="store_true", help="Allow exporting failed Boss snapshots from EVAL seeds (diagnostics ONLY, never for training!)")
    ap.add_argument("--merge-equiv", action="store_true", help="Merge identical cards before argmax")
    ap.add_argument("--replays", action="store_true", help="Save full .jsonl.gz replays (off by default for fast milestone sweeps)")
    ap.add_argument("--stall-turns", type=int, default=20)
    ap.add_argument("--max-steps-run", type=int, default=6000)
    ap.add_argument("--max-steps-fight", type=int, default=1000)
    ap.add_argument("--time-cap", type=float, default=600.0)
    a = ap.parse_args()

    if (a.export_failed_bosses or a.export_all_bosses) and not a.boss_snaps:
        a.boss_snaps = True
    if a.export_failed_bosses and not a.allow_diag_export:
        pfx = (a.seed_prefix or "").upper()
        if "EVAL" in pfx or "VAL" in pfx or "HOLD" in pfx or "TEST" in pfx or "CONF" in pfx:
            ap.error(
                f"Refusing to export Boss snapshots from evaluation seed prefix '{a.seed_prefix}' "
                "to prevent eval-set leakage into SolverPool! Use --seed-prefix A10MINE for training-side mining, "
                "or pass --allow-diag-export for read-only diagnostics."
            )

    torch.set_num_threads(2)
    ckpts = resolve_ckpts(a.ckpts)
    seeds = load_seeds(a.runs, a.seed_prefix, a.seeds_file, seed_start=a.seed_start)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    exported_bosses = []
    exported_all_bosses = []
    print(f"Evaluating {len(ckpts)} checkpoint(s) | char={a.character} A{a.asc} | weights={a.weights} | runs={len(seeds)} ({seeds[0]}..{seeds[-1]}) | envs={a.envs} dev={a.device}", flush=True)
    for cp in ckpts:
        sm, rows = eval_one_ckpt(cp, a, seeds, out_dir)
        summaries.append(sm)
        rate_str = f"{sm['games_per_hour']:.0f} g/h" if sm.get("games_per_hour") else "cached"
        print(
            f"v{sm['version']:>5d} ({sm['total_steps']/1e6:5.1f}M) [{sm['weights']}] | "
            f"n={sm['n']:>3d} ({rate_str}) | fl={sm['mean_floor']:5.2f} | "
            f"A2={sm['reach_act2']}/{sm['n']}({sm['reach_act2_rate']*100:4.1f}%) "
            f"A3={sm['reach_act3']}/{sm['n']}({sm['reach_act3_rate']*100:4.1f}%) "
            f"Win={sm['wins']}/{sm['n']}({sm['win_rate']*100:4.1f}%) | "
            f"B1={_fmt_ratio(sm['boss1'])} B2={_fmt_ratio(sm['boss2'])} B3.1={_fmt_ratio(sm['boss3_1'])} | stuck={sm['stuck']}",
            flush=True,
        )
        if a.export_failed_bosses:
            exported_bosses.extend(extract_boss_snaps(rows, sm["version"], a.weights, a.character, a.asc, failed_only=True))
        if a.export_all_bosses:
            exported_all_bosses.extend(extract_boss_snaps(rows, sm["version"], a.weights, a.character, a.asc, failed_only=False))

    summary_path = out_dir / f"summary_{a.weights}.json"
    summary_path.write_text(json.dumps({"config": vars(a), "results": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary -> {summary_path}", flush=True)

    if a.export_failed_bosses:
        exp_path = Path(a.export_failed_bosses)
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        with exp_path.open("w", encoding="utf-8") as fh:
            for it in exported_bosses:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        by_act = {act: sum(1 for x in exported_bosses if x["act"] == act) for act in (0, 1, 2)}
        print(f"Exported {len(exported_bosses)} failed Boss snapshots (A1={by_act[0]}, A2={by_act[1]}, A3={by_act[2]}) -> {exp_path}", flush=True)

    if a.export_all_bosses:
        all_path = Path(a.export_all_bosses)
        all_path.parent.mkdir(parents=True, exist_ok=True)
        with all_path.open("w", encoding="utf-8") as fh:
            for it in exported_all_bosses:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        by_act = {act: sum(1 for x in exported_all_bosses if x["act"] == act) for act in (0, 1, 2)}
        print(f"Exported {len(exported_all_bosses)} all-reached Boss snapshots (A1={by_act[0]}, A2={by_act[1]}, A3={by_act[2]}) -> {all_path}", flush=True)


if __name__ == "__main__":
    main()
