#!/usr/bin/env python3
"""Expand independent A10MINE Act-2 Boss pool for Step 1 and build strictly matched Control/Solver pools.

Pipeline (resumable & hash-deduplicated):
  1. Seed master store from existing A10MINE0000..0511 pilot results (if present) without recomputing.
  2. In chunks of --chunk-runs (default 1024) starting from --start-seed (default 512):
       a) Mine independent runs with eval_v1140.py (--seed-prefix A10MINE, --seed-start S).
       b) Run 12-worker batched fast solver (r16 / b0 / 12s) on Act-2 Boss failures (--acts 1).
       c) Merge into deduplicated master_policy_lost.jsonl and master_verified.jsonl (keyed by snap_hash).
     Stop as soon as unique_verified >= --target (default 250).
  3. Build two strictly matched labeled.jsonl pools from the EXACT SAME policy_lost mother pool:
       - Solver arm: N = --target unique verified solvable A2 mistakes (pool_eligible=true, win=true, solver_status="verified_solvable_mistake")
       - Control arm: N = --target unique items sampled WITHOUT replacement from the same policy_lost mother pool
         with EXACTLY MATCHED per-Boss encounter counts, pool_eligible=true, preserving true win & solver_status.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def make_pool_record(res: dict, raw_item: dict) -> dict:
    is_ver = res["status"] == "verified_solvable_mistake"
    spec = res.get("spec") or {
        "character": raw_item.get("character", "NECROBINDER"),
        "ascension": int(raw_item.get("ascension", 10)),
        "seed": res.get("combat_seed") or f"SOLV_{res['id']}",
        "act": int(res["act"]),
        "encounter": res["encounter"],
        "room": "Boss",
        "player_snapshot": raw_item["snap"],
    }
    return {
        "id": res["id"],
        "run_seed": res.get("run_seed") or raw_item.get("run_seed"),
        "snap_hash": res["snap_hash"],
        "act": int(res["act"]),
        "encounter": res["encounter"],
        "pool_eligible": True,
        "win": bool(is_ver),
        "solver_status": res["status"],
        "end_hp": res.get("verified_end_hp") if is_ver else None,
        "entry_hp": raw_item.get("hp"),
        "entry_max_hp": raw_item.get("max_hp"),
        "entry_potions": raw_item.get("potions"),
        "entry_deck": raw_item.get("deck"),
        "entry_upgrades": raw_item.get("upgrades"),
        "spec": spec,
    }


def merge_results(
    mother: dict[str, dict],
    verified: dict[str, dict],
    raw_items_path: Path,
    results_path: Path,
) -> tuple[int, int]:
    raw_map = {it["id"]: it for it in _load_jsonl(raw_items_path)}
    added_lost, added_ver = 0, 0
    for r in _load_jsonl(results_path):
        st = r.get("status")
        if st not in ("verified_solvable_mistake", "unknown"):
            continue
        sh = r.get("snap_hash")
        if not sh or r["id"] not in raw_map:
            continue
        rec = make_pool_record(r, raw_map[r["id"]])
        if sh not in mother:
            mother[sh] = rec
            added_lost += 1
        elif rec["win"] and not mother[sh]["win"]:
            # Upgrade status if verified in another pass (e.g. union of 20s and 12s pilot passes)
            mother[sh] = rec
        if rec["win"] and sh not in verified:
            verified[sh] = rec
            added_ver += 1
    return added_lost, added_ver


def build_matched_pools(
    mother: dict[str, dict],
    verified: dict[str, dict],
    target: int,
    solver_labeled_out: Path,
    control_labeled_out: Path,
    manifest_out: Path,
    rng_seed: int = 20261127,
) -> dict:
    rng = np.random.default_rng(rng_seed)
    ver_list = list(verified.values())
    if len(ver_list) < target:
        raise RuntimeError(f"Only {len(ver_list)} unique verified items available, need {target}")
    # Take first `target` in deterministic mine order (already deduplicated by snap_hash)
    solver_pool = ver_list[:target]
    enc_counts = Counter(r["encounter"] for r in solver_pool)

    # Group entire policy_lost mother pool (verified + unknown) by encounter
    by_enc: dict[str, list[dict]] = {}
    for rec in mother.values():
        by_enc.setdefault(rec["encounter"], []).append(rec)

    control_pool: list[dict] = []
    for enc, need in sorted(enc_counts.items()):
        cands = by_enc.get(enc, [])
        if len(cands) < need:
            raise RuntimeError(f"Encounter {enc} has only {len(cands)} mother items, need {need}")
        idx = rng.choice(len(cands), size=need, replace=False)
        for i in sorted(idx.tolist()):
            control_pool.append(cands[i])

    # Sanity assertions
    assert len(solver_pool) == target and len(control_pool) == target
    assert len({r["snap_hash"] for r in solver_pool}) == target
    assert len({r["snap_hash"] for r in control_pool}) == target
    assert Counter(r["encounter"] for r in solver_pool) == Counter(r["encounter"] for r in control_pool)
    assert all(r["pool_eligible"] is True and r["win"] is True and r["solver_status"] == "verified_solvable_mistake" for r in solver_pool)
    assert all(r["pool_eligible"] is True and r["solver_status"] in ("verified_solvable_mistake", "unknown") for r in control_pool)

    _write_jsonl(solver_labeled_out, solver_pool)
    _write_jsonl(control_labeled_out, control_pool)

    def _stats(arr: list[dict]) -> dict:
        hp = [r["entry_hp"] / max(r["entry_max_hp"] or 1, 1) for r in arr if r.get("entry_hp") is not None]
        pots = [r["entry_potions"] for r in arr if r.get("entry_potions") is not None]
        deck = [r["entry_deck"] for r in arr if r.get("entry_deck") is not None]
        upg = [r["entry_upgrades"] for r in arr if r.get("entry_upgrades") is not None]
        return {
            "n": len(arr),
            "unique_hashes": len({r["snap_hash"] for r in arr}),
            "by_encounter": dict(sorted(Counter(r["encounter"] for r in arr).items())),
            "by_solver_status": dict(sorted(Counter(r["solver_status"] for r in arr).items())),
            "mean_entry_hp_pct": round(100.0 * sum(hp) / max(len(hp), 1), 2),
            "mean_entry_potions": round(sum(pots) / max(len(pots), 1), 2),
            "mean_entry_deck": round(sum(deck) / max(len(deck), 1), 2),
            "mean_entry_upgrades": round(sum(upg) / max(len(upg), 1), 2),
        }

    overlap = len({r["snap_hash"] for r in solver_pool} & {r["snap_hash"] for r in control_pool})
    manifest = {
        "target_size": target,
        "rng_seed": rng_seed,
        "mother_policy_lost_unique": len(mother),
        "all_verified_unique": len(verified),
        "pool_hash_overlap": overlap,
        "solver_pool_path": str(solver_labeled_out),
        "control_pool_path": str(control_labeled_out),
        "solver_pool": _stats(solver_pool),
        "control_pool": _stats(control_pool),
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand A2 Boss verified mistake pool and build matched Step 1 pools.")
    ap.add_argument("--ckpt", default="runs/fork-v3050-rf50-necro-v1140-a10-s0/latest.pt")
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--work-dir", default="runs/step1-a2-pool-v3965")
    ap.add_argument("--seed-pilot-dir", default="runs/solver-pilot-v3965-a2")
    ap.add_argument("--solver-out", default="runs/step1-v3965-solver-a2-s0/solver/labeled.jsonl")
    ap.add_argument("--control-out", default="runs/step1-v3965-randfail-a2-s0/solver/labeled.jsonl")
    ap.add_argument("--target", type=int, default=250)
    ap.add_argument("--start-seed", type=int, default=512)
    ap.add_argument("--chunk-runs", type=int, default=1024)
    ap.add_argument("--max-chunks", type=int, default=12)
    ap.add_argument("--mine-envs", type=int, default=16)
    ap.add_argument("--solve-workers", type=int, default=12)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--branch-points", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=12.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    mother_path = work / "master_policy_lost.jsonl"
    verified_path = work / "master_verified.jsonl"

    mother: dict[str, dict] = {r["snap_hash"]: r for r in _load_jsonl(mother_path)}
    verified: dict[str, dict] = {r["snap_hash"]: r for r in _load_jsonl(verified_path)}

    # Seed from existing 0000..0511 pilot (union of batched 20s and fast 12s)
    pilot = Path(a.seed_pilot_dir)
    if pilot.exists():
        raw512 = pilot / "failed_bosses_512.jsonl"
        for rf in ("a2_batched_results.jsonl", "a2_fast_results.jsonl"):
            rp = pilot / rf
            if raw512.exists() and rp.exists():
                merge_results(mother, verified, raw512, rp)
        _write_jsonl(mother_path, list(mother.values()))
        _write_jsonl(verified_path, list(verified.values()))

    print(
        f"[INIT] mother_policy_lost={len(mother)} | unique_verified={len(verified)}/{a.target} | "
        f"by_enc={dict(sorted(Counter(r['encounter'] for r in verified.values()).items()))}",
        flush=True,
    )

    t_global = time.time()
    start_ver = len(verified)
    cur_seed = a.start_seed

    for chunk_i in range(a.max_chunks):
        if len(verified) >= a.target:
            break
        end_seed = cur_seed + a.chunk_runs - 1
        tag = f"chunk_{cur_seed:04d}_{end_seed:04d}"
        cdir = work / tag
        cdir.mkdir(parents=True, exist_ok=True)
        failed_jsonl = cdir / "failed_bosses.jsonl"
        solv_jsonl = cdir / "a2_verify_results.jsonl"
        solv_summary = cdir / "a2_verify_summary.json"
        solv_labeled = cdir / "a2_verified_labeled.jsonl"

        print(f"\n=== [{tag}] Mining {a.chunk_runs} runs (A10MINE{cur_seed:04d}..{end_seed:04d}) ===", flush=True)
        t_c = time.time()
        if not failed_jsonl.exists():
            subprocess.run(
                [
                    sys.executable, "-u", str(ROOT / "scripts" / "eval_v1140.py"),
                    "--ckpts", a.ckpt,
                    "--out", str(cdir / "mine"),
                    "--weights", a.weights,
                    "--runs", str(a.chunk_runs),
                    "--seed-start", str(cur_seed),
                    "--seed-prefix", "A10MINE",
                    "--envs", str(a.mine_envs),
                    "--device", a.device,
                    "--export-failed-bosses", str(failed_jsonl),
                ],
                check=True,
            )
        else:
            print(f"[{tag}] Reusing existing {failed_jsonl}", flush=True)

        print(f"=== [{tag}] Solving Act-2 failures (r{a.rollouts}/b{a.branch_points}/{a.timeout_s}s, w{a.solve_workers}) ===", flush=True)
        subprocess.run(
            [
                sys.executable, "-u", str(ROOT / "scripts" / "verify_boss_solver.py"),
                "--items", str(failed_jsonl),
                "--ckpt", a.ckpt,
                "--weights", a.weights,
                "--acts", "1",
                "--limit", "0",
                "--workers", str(a.solve_workers),
                "--rollouts", str(a.rollouts),
                "--branch-points", str(a.branch_points),
                "--branch-k", "0",
                "--timeout-s", str(a.timeout_s),
                "--device", a.device,
                "--out", str(solv_jsonl),
                "--summary-out", str(solv_summary),
                "--export-solver-jsonl", str(solv_labeled),
            ],
            check=True,
        )

        al, av = merge_results(mother, verified, failed_jsonl, solv_jsonl)
        _write_jsonl(mother_path, list(mother.values()))
        _write_jsonl(verified_path, list(verified.values()))
        dt_c = time.time() - t_c
        dt_all = time.time() - t_global
        new_tot = len(verified) - start_ver
        rate = new_tot / max(dt_all / 3600.0, 1e-6)
        print(
            f"[{tag} DONE in {dt_c/60:.1f}m] +{al} mother_lost, +{av} unique_verified | "
            f"TOTAL mother={len(mother)}, verified={len(verified)}/{a.target} | "
            f"session_rate={rate:.1f}/hr | by_enc={dict(sorted(Counter(r['encounter'] for r in verified.values()).items()))}",
            flush=True,
        )
        cur_seed += a.chunk_runs

    if len(verified) < a.target:
        raise SystemExit(f"Reached max_chunks={a.max_chunks} with only {len(verified)}/{a.target} verified items.")

    manifest = build_matched_pools(
        mother=mother,
        verified=verified,
        target=a.target,
        solver_labeled_out=Path(a.solver_out),
        control_labeled_out=Path(a.control_out),
        manifest_out=work / "pool_manifest.json",
    )
    print("\n=== MATCHED STEP-1 POOLS BUILT ===", flush=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
