"""Bounded Necrobinder evaluations on the Windows public-beta host."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
from datetime import datetime, timezone

# A single small inference batch should use a single BLAS thread on this laptop.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

from beta_build import ASSEMBLY, BUILD, CHECKOUT, GAME_COMMIT, HOST, MANIFEST, REVISION, ROOT, sha, verify_host
from beta_env import run_episode


# Distinct panels drawn once before evaluating. The first two seeds are reserved for training.
TRAIN = ("RWN5T3VD6N2E", "93T9FBYR1TRZ")
EVAL = ("YESSX3NDTXXZ", "JUHA4BDNP9TV", "MHJA6MWFPM5D", "Q484ZHBQM8VB",
        "6JA7LD1U23BD", "8668D8L9UBCT", "FLJDKK3DH2A4", "BBLKA3Z0NHN3")


class RulePolicy:
    """Play available cards, then end turn. Pick other offered choices by a separate PRNG."""
    def __init__(self, index):
        self.rng = random.Random(6150 + index)

    def __call__(self, state, actions):
        if state["decision"] == "combat_play":
            for kind in ("play_card", "end_turn"):
                for index, action in enumerate(actions):
                    if action["action"] == kind:
                        return index
        return self.rng.randrange(len(actions))


def wilson(wins, n):
    z = 1.959963984540054
    rate = wins / n
    denom = 1 + z*z/n
    center = (rate + z*z/(2*n)) / denom
    margin = z / denom * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))
    return [max(0, center-margin), min(1, center+margin)]


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standard", "blitz"), default="standard")
    parser.add_argument("--ascension", type=int, default=10)
    parser.add_argument("--games", type=int, choices=range(1, len(EVAL)+1), default=8)
    parser.add_argument("--policy", choices=("rule", "initial", "imitation"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--load-from", type=Path, help="Directory containing protocol.json and readout.npz for a fitted head")
    args = parser.parse_args()
    if args.load_from and args.policy != "imitation":
        parser.error("--load-from requires --policy imitation")
    max_level = 5 if args.mode == "blitz" else 10
    if not 0 <= args.ascension <= max_level:
        parser.error(f"{args.mode} ascension must be 0..{max_level}")
    verify_host()
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    out = args.out or ROOT / "runs" / datetime.now(timezone.utc).strftime("necro-%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    meta = {"character": "Necrobinder", "platform": "Windows", "build": BUILD, "game_commit": GAME_COMMIT,
            "host_revision": REVISION, "game_assembly_sha256": ASSEMBLY, "host_manifest_sha256": sha(MANIFEST),
            "mode": args.mode, "ascension": args.ascension, "policy": args.policy, "seeds": EVAL[:args.games],
            "max_steps": 1024, "episode_timeout_seconds": 90,
            "source_files": {name: sha(Path(__file__).with_name(name)) for name in
                             ("beta_build.py", "beta_eval.py", "beta_env.py", "necro_policy.py")}}
    model = None
    if args.policy != "rule":
        import numpy
        from necro_policy import NecroPolicy
        model = NecroPolicy(seed=711072)
        meta.update(parameter_count=model.parameter_count, initialization_seed=711072,
                    initialization_sha256=model.fingerprint(), numpy=numpy.__version__, updates=0)
        if args.load_from:
            old = json.loads((args.load_from / "protocol.json").read_text(encoding="utf-8"))
            for field in ("policy", "initialization_sha256", "build", "mode", "ascension", "host_revision"):
                if old[field] != meta[field]:
                    raise ValueError(f"Incompatible checkpoint: {field}")
            with numpy.load(args.load_from / "readout.npz", allow_pickle=False) as params:
                model.readout[:] = params["readout"]
                model.readout_bias[:] = params["bias"]
            if model.fingerprint() != old["model_sha256"]:
                raise ValueError("Restored model fingerprint mismatch")
            meta.update(updates=old["updates"], training_seeds=old["training_seeds"])
        elif args.policy == "imitation":
            training = []
            for index, seed in enumerate(TRAIN):
                teacher = RulePolicy(index + 100)

                def demonstrate(state, actions):
                    choice = teacher(state, actions)
                    if meta["updates"] < 64 and len(actions) > 1:
                        model.fit_demonstration(state, actions, choice)
                        meta["updates"] += 1
                    return choice

                row = run_episode(HOST, seed, args.mode, args.ascension, demonstrate,
                                  max_steps=meta["max_steps"], episode_timeout=meta["episode_timeout_seconds"])
                training.append(row)
                write_json(out / "training.json", training)
                if row["status"] not in ("defeat", "victory"):
                    raise RuntimeError(f"Demonstration failed on {seed}: {row}")
                if meta["updates"] == 64:
                    break
            if not meta["updates"]:
                raise RuntimeError("No decisions available for imitation")
            meta["training_seeds"] = [row["seed"] for row in training]
        if args.policy == "imitation":
            numpy.savez(out / "readout.npz", readout=model.readout, bias=model.readout_bias)
        meta["model_sha256"] = model.fingerprint()
    write_json(out / "protocol.json", meta)
    rows = []
    for index, seed in enumerate(EVAL[:args.games]):
        policy = model if model is not None else RulePolicy(index)
        calls = model.forward_calls if model is not None else 0
        row = run_episode(HOST, seed, args.mode, args.ascension, policy,
                          max_steps=meta["max_steps"], episode_timeout=meta["episode_timeout_seconds"])
        if model is not None:
            row["forward_calls"] = model.forward_calls - calls
        write_json(out / f"{seed}.json", row)
        rows.append(row)
        print(f"{seed} {row['status']} act={row.get('act')} floor={row.get('floor')} steps={row['steps']}", flush=True)
    statuses = Counter(r["status"] for r in rows)
    completed = [r for r in rows if r["status"] in ("victory", "defeat")]
    summary = {"protocol": meta, "status_counts": dict(statuses), "finished": len(completed),
               "wins": statuses["victory"], "attempted": len(rows),
               "win_rate_all_attempts": statuses["victory"] / len(rows),
               "wilson95_all_attempts": wilson(statuses["victory"], len(rows)),
               "acts_reached": dict(Counter(r["act"] for r in completed)),
               "median_terminal_floor_within_act": statistics.median(r["floor"] for r in completed) if completed else None,
               "rows": rows}
    write_json(out / "summary.json", summary)
    print(out / "summary.json")
    return int(len(completed) != len(rows))


if __name__ == "__main__":
    sys.exit(main())
