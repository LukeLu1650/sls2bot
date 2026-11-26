#!/usr/bin/env python3
"""Frozen-Fighter Boss Handover Probe: decouples deck/resource preparation from combat skill.

Given pre-Boss snapshots exported by eval_v1140.py --export-all-bosses (from two policies
run on the same unseen A10EVAL seeds), hands every reached Boss-entry state to ONE shared
frozen combat policy under matched deterministic combat seeds (HANDOVER_<run_seed>_a<act>).

Reports per act:
  - Boss reach count & entry preparation stats (HP%, potions, upgrades, deck size)
  - Frozen-fighter Boss win rate (conditional on reach)
  - Unconditional preparation score (frozen-fighter Boss wins / total eval runs)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AD_PY = ROOT / "external" / "AlphaDefect-Public" / "game-v0.114.0" / "python"
if str(AD_PY) not in sys.path:
    sys.path.insert(0, str(AD_PY))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from eval_v1140 import load_net_weights  # noqa: E402
from sts2env.env import EngineError, Sts2Env  # noqa: E402
from verify_boss_solver import _run_episode, reset_combat_with_flags  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen-Fighter Boss Handover Probe.")
    ap.add_argument("--snaps", required=True, help="JSONL of all-reached Boss snapshots (from eval_v1140.py --export-all-bosses)")
    ap.add_argument("--fighter-ckpt", required=True, help="Frozen combat fighter checkpoint (.pt)")
    ap.add_argument("--fighter-weights", choices=("ema", "raw"), default="ema")
    ap.add_argument("--total-runs", type=int, default=128, help="Total eval runs denominator for unconditional win rate")
    ap.add_argument("--acts", default="0,1,2", help="Acts to probe (0=A1, 1=A2, 2=A3)")
    ap.add_argument("--out", required=True, help="Output JSON summary & per-item records")
    ap.add_argument("--stall-turns", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.set_num_threads(2)
    net, blob, src_key = load_net_weights(Path(a.fighter_ckpt), a.fighter_weights)
    net = net.to(a.device)
    net.infer_ready()
    print(f"Frozen fighter: {a.fighter_ckpt} (v{blob['version']}, {a.fighter_weights}/{src_key}) on {a.device}", flush=True)

    act_set = {int(x) for x in a.acts.split(",") if x.strip()}
    items = [json.loads(ln) for ln in Path(a.snaps).read_text(encoding="utf-8").splitlines() if ln.strip()]
    items = [it for it in items if int(it.get("act", -1)) in act_set]

    env = Sts2Env()
    t0 = time.time()
    records = []
    try:
        for idx, it in enumerate(items, 1):
            act = int(it["act"])
            spec = {
                "character": it.get("character", "NECROBINDER"),
                "ascension": int(it.get("ascension", 10)),
                "seed": f"HANDOVER_{it['run_seed']}_a{act+1}",
                "act": act,
                "encounter": it["encounter"],
                "room": "Boss",
                "player_snapshot": it["snap"],
            }
            try:
                obs0 = reset_combat_with_flags(env, spec, a.stall_turns)
                res = _run_episode(env, obs0, net, a.device, None, temp=0.0, max_steps=a.max_steps)
                rec = {
                    "id": it["id"],
                    "run_seed": it["run_seed"],
                    "act": act,
                    "encounter": it["encounter"],
                    "orig_boss_win": bool(it.get("orig_boss_win")),
                    "entry_hp": it.get("hp"),
                    "entry_max_hp": it.get("max_hp"),
                    "entry_hp_frac": (it["hp"] / max(it["max_hp"], 1)) if it.get("hp") is not None and it.get("max_hp") else None,
                    "entry_potions": len(it["potions"]) if isinstance(it.get("potions"), list) else it.get("potions"),
                    "entry_upgrades": it.get("upgrades"),
                    "entry_deck_size": len(it["deck"]) if isinstance(it.get("deck"), list) else it.get("deck"),
                    "fighter_win": bool(res["win"]),
                    "fighter_end_hp": int(res["end_hp"]),
                    "fighter_steps": int(res["steps"]),
                    "stuck": bool(res["stuck"]),
                }
            except EngineError as err:
                rec = {"id": it["id"], "run_seed": it["run_seed"], "act": act, "encounter": it["encounter"], "fighter_win": False, "error": str(err)[:200]}
            records.append(rec)
            if idx % 20 == 0 or idx == len(items):
                print(f"  probed {idx}/{len(items)} ({time.time()-t0:.1f}s)", flush=True)
    finally:
        env.close()

    by_act = {}
    for act in sorted(act_set):
        sub = [r for r in records if r["act"] == act]
        if not sub:
            continue
        fw = sum(1 for r in sub if r.get("fighter_win"))
        ow = sum(1 for r in sub if r.get("orig_boss_win"))
        hp_fracs = [r["entry_hp_frac"] for r in sub if r.get("entry_hp_frac") is not None]
        pots = [r["entry_potions"] for r in sub if r.get("entry_potions") is not None]
        upgs = [r["entry_upgrades"] for r in sub if r.get("entry_upgrades") is not None]
        decks = [r["entry_deck_size"] for r in sub if r.get("entry_deck_size") is not None]
        sm = {
            "reached": len(sub),
            "orig_boss_wins": ow,
            "fighter_wins": fw,
            "fighter_cond_wr": round(fw / len(sub), 4),
            "fighter_uncond_wr": round(fw / max(a.total_runs, 1), 4),
            "mean_hp_frac": round(statistics.mean(hp_fracs), 3) if hp_fracs else None,
            "mean_potions": round(statistics.mean(pots), 2) if pots else None,
            "mean_upgrades": round(statistics.mean(upgs), 2) if upgs else None,
            "mean_deck_size": round(statistics.mean(decks), 2) if decks else None,
        }
        by_act[f"act{act+1}"] = sm
        print(
            f"Act {act+1} Boss | reached={len(sub)}/{a.total_runs} | "
            f"frozen_fighter_win={fw}/{len(sub)} ({sm['fighter_cond_wr']*100:.1f}% cond, {sm['fighter_uncond_wr']*100:.1f}% uncond) | "
            f"HP%={sm['mean_hp_frac']} pots={sm['mean_potions']} upg={sm['mean_upgrades']} deck={sm['mean_deck_size']}",
            flush=True,
        )

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "fighter_ckpt": a.fighter_ckpt,
                "fighter_version": int(blob["version"]),
                "fighter_weights": a.fighter_weights,
                "snaps": a.snaps,
                "total_runs": a.total_runs,
                "wall_s": round(time.time() - t0, 1),
                "by_act": by_act,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved handover probe -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
