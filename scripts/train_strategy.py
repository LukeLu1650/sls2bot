"""Run bounded, resumable strategic-action training in Windows Standard mode."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
from time import perf_counter

from beta_eval import BUILD, CHECKOUT, HOST, MANIFEST, ROOT, sha, verify_host, wilson, write_json
from beta_env import run_episode
from strategic_features import FEATURE_COUNT, VARIANTS, features, probabilities
from train_standard import StandardPolicy, np


BASE = ROOT / "results/windows-v0.113.1/standard-continuation"
ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
SCHEMA = 2
# Earlier schema-2 trainer used the same features and update but always ran all cohorts.
PRE_VARIANT_TRAINER = "e79cb0acf212f08843e9e8857aff9018c39de9c0e9a4a061cf522f65069cba5e"
STRATEGY_FILES = ("train_strategy.py", "strategic_features.py", "beta_env.py", "train_standard.py", "necro_policy.py")


def panel(domain, number, unique_key):
    rng = random.Random(f"standard-strategy-v1/{unique_key}/{domain}")
    seeds = set()
    while len(seeds) < number:
        seeds.add("".join(rng.choices(ALPHABET, k=12)))
    return sorted(seeds)


def load_base():
    model = StandardPolicy()
    with np.load(BASE / "readout.npz", allow_pickle=False) as file:
        model.readout[:] = file["readout"]
        model.readout_bias[:] = file["bias"]
    expected = json.loads((BASE / "report.json").read_bytes())["model_sha256"]
    if model.fingerprint() != expected:
        raise ValueError("Base policy weights differ from the recorded checkpoint")
    return model


class RunPolicy:
    def __init__(self, base, variant, weights, rng=None):
        self.base = base
        self.variant = variant
        self.weights = weights
        self.rng = rng
        self.gradients = []
        self.reason = None
        self.coverage = Counter()

    def __call__(self, state, actions):
        scene = state["decision"]
        if scene == "combat_play":
            self.reason = None
            return self.base(state, actions)
        self.coverage.update(action["action"] for action in actions)
        selected = 0
        if len(actions) > 1:
            logits = self.base.scores(state, actions)
            if self.variant != "anchor":
                matrix = features(state, actions, self.variant, self.reason)
                logits = logits + matrix @ self.weights
            if self.rng is None:
                selected = int(np.argmax(logits))
            else:
                distribution = probabilities(logits)
                selected = int(self.rng.choice(len(actions), p=distribution.astype(np.float64) / distribution.sum(dtype=np.float64)))
                self.gradients.append(matrix[selected] - distribution @ matrix)
        action = actions[selected]
        if action["action"] == "remove_card":
            self.reason = "remove_card"
        elif scene == "rest_site" and action["action"] == "choose_option":
            option = next((item for item in state["options"] if item["index"] == action["args"]["option_index"]), {})
            if any("smith" in str(option.get(key, "")).lower() or "upgrade" in str(option.get(key, "")).lower()
                   for key in ("name", "option_id", "title")):
                self.reason = "smith"
        elif scene != "card_select":
            self.reason = None
        return selected


def reward(row):
    act = row["act"] or 1
    floor = min(row["floor"] or 0, 17)
    # Surviving a floor gives a small signal. A win carries the largest bonus.
    return float(row["victory"]) + 0.16 * (act - 1) + 0.005 * floor


def play(base, variant, weights, seeds, args, training=False):
    rows, samples = [], []
    for seed in seeds:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little")) if training else None
        chooser = RunPolicy(base, variant, weights, rng)
        row = run_episode(HOST, seed, "standard", args.ascension, chooser, max_steps=args.max_steps,
                          episode_timeout=args.episode_timeout, strategic_observations=True)
        row["offered_candidates"] = dict(chooser.coverage)
        row["sampled_strategic_decisions"] = len(chooser.gradients)
        rows.append(row)
        if training and row["status"] in {"victory", "defeat"} and chooser.gradients:
            samples.append((reward(row), np.mean(chooser.gradients, axis=0)))
        print(f"{'train' if training else 'eval'} {variant:<9} {seed}  {row['status']:<18}"
              f" act={row['act']} floor={row['floor']} steps={row['steps']}", flush=True)
    return rows, samples


def summarize(rows):
    total = len(rows)
    wins = sum(row["victory"] is True for row in rows)
    statuses = Counter(row["status"] for row in rows)
    cleared = [sum((row["act"] or 0) > act for row in rows) for act in (1, 2)] + [wins]
    results = []
    for act in (1, 2, 3):
        by_order = {}
        for row in rows:
            entries = [entry for entry in row["boss_entries"] if entry["act"] == act]
            for index, entry in enumerate(entries, start=1):
                by_order.setdefault(index, []).append(entry)
        for order, entries in sorted(by_order.items()):
            before = [x["before_travel"]["hp"] for x in entries if x["before_travel"]]
            results.append({"act": act, "boss_number": order, "samples": len(entries),
                            "median_first_input_hp": statistics.median(x["hp"] for x in entries),
                            "median_first_input_hp_fraction": statistics.median(x["hp"] / x["max_hp"] for x in entries),
                            "before_travel_samples": len(before),
                            "median_before_travel_hp": statistics.median(before) if before else None})
    offered, actions = Counter(), Counter()
    for row in rows:
        offered.update(row["offered_candidates"])
        actions.update(row["actions"])
    return {"attempted": total, "wins": wins, "win_rate": wins / total,
            "win_wilson95": wilson(wins, total), "act_clears": cleared,
            "act_clear_rate": [count / total for count in cleared],
            "statuses": dict(statuses), "boss_hp": results, "offered": dict(offered), "selected": dict(actions)}


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def strategy_sources():
    return {name: sha(Path(__file__).with_name(name)) for name in STRATEGY_FILES}


def normalize_protocol(protocol):
    compatible = dict(protocol)
    compatible.setdefault("variants", list(VARIANTS))
    sources = compatible["sources"]
    if compatible["schema"] == SCHEMA and sources.get("train_strategy.py") == PRE_VARIANT_TRAINER:
        compatible["sources"] = {**sources, "train_strategy.py": sha(Path(__file__))}
    return compatible


def fingerprint(args, variants):
    return {"schema": SCHEMA, "game": BUILD, "host": sha(MANIFEST), "mode": "standard",
            "variants": list(variants),
            "seed_key": args.seed_key, "ascension": args.ascension, "games_per_round": args.games,
            "development_games": args.eval_games, "evaluate_every": args.eval_every,
            "max_steps": args.max_steps, "episode_timeout": args.episode_timeout,
            "base_weights": sha(BASE / "readout.npz"),
            "sources": strategy_sources()}


def checkpoint(path, header, rounds, weights):
    write_json(path, {"protocol": header, "round": rounds,
                      "weights": {name: values.tolist() for name, values in weights.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=positive_int, default=1, help="Total rounds, including completed rounds on resume")
    parser.add_argument("--games", type=positive_int, default=4, help="Training seeds per round and per variant")
    parser.add_argument("--eval-games", type=positive_int, default=4, help="Fixed development seeds")
    parser.add_argument("--eval-every", type=positive_int, default=1, help="Rounds between development evaluations")
    parser.add_argument("--final-games", type=positive_int, default=64, help="Reserved seeds used by --final")
    parser.add_argument("--max-steps", type=positive_int, default=750)
    parser.add_argument("--episode-timeout", type=positive_int, default=65)
    parser.add_argument("--ascension", type=int, choices=range(11), default=0)
    parser.add_argument("--seed-key", default="necro-standard")
    parser.add_argument("--variant", choices=("all", *VARIANTS),
                        help="Train one policy or all three. Resume defaults to the checkpoint's selection.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--final", action="store_true", help="Evaluate saved checkpoints once without updating weights")
    parser.add_argument("--test-ascension", type=int, choices=range(11), help="Final evaluation difficulty")
    parser.add_argument("--start-from", type=Path, help="Initialize a new curriculum stage from another checkpoint.json")
    args = parser.parse_args()
    if args.test_ascension is not None and not args.final:
        parser.error("--test-ascension requires --final")
    if args.start_from and (args.resume or args.final):
        parser.error("--start-from starts a new output directory")
    host = verify_host()
    if host.get("event_patch_sha256") != sha(ROOT / "patches/HeadlessEventPresentation.cs"):
        parser.error("Rebuild the Windows headless host with scripts/beta_build.py before training")
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    state_file = args.out / "checkpoint.json"
    old = json.loads(state_file.read_bytes()) if args.resume or args.final else None
    previous_protocol = normalize_protocol(old["protocol"]) if old else None
    if args.variant is None and previous_protocol:
        variants = tuple(previous_protocol["variants"])
    else:
        variants = VARIANTS if args.variant in (None, "all") else (args.variant,)
    header = fingerprint(args, variants)
    previous = 0
    weights = {name: np.zeros(FEATURE_COUNT, dtype=np.float32) for name in variants}
    if old is not None:
        if "start_from" in previous_protocol:
            header["start_from"] = previous_protocol["start_from"]
        if previous_protocol != header:
            raise ValueError("Checkpoint protocol differs. Reuse the original run arguments and source revision.")
        previous = old["round"]
        weights = {name: np.asarray(old["weights"][name], dtype=np.float32) for name in variants}
    else:
        if args.start_from:
            earlier = json.loads(args.start_from.read_bytes())
            earlier_protocol = normalize_protocol(earlier["protocol"])
            for key in ("schema", "host", "mode", "base_weights", "sources"):
                if earlier_protocol[key] != header[key]:
                    raise ValueError(f"Previous curriculum stage has an incompatible {key}")
            unavailable = set(variants) - earlier["weights"].keys()
            if unavailable:
                raise ValueError(f"Previous checkpoint lacks {', '.join(sorted(unavailable))}. Select an included variant.")
            header["start_from"] = sha(args.start_from)
            weights = {name: np.asarray(earlier["weights"][name], dtype=np.float32) for name in variants}
        args.out.mkdir(parents=True, exist_ok=False)
        checkpoint(state_file, header, 0, weights)
    base = load_base()
    dev = panel("development", args.eval_games, args.seed_key)
    fixed_eval = (panel("final", args.final_games, args.seed_key) if args.final else dev)
    if args.final:
        if previous < 1:
            parser.error("Finish a training round before final evaluation")
        if args.test_ascension is not None:
            args.ascension = args.test_ascension
        destination = args.out / f"final-A{args.ascension}.json"
        if destination.exists():
            parser.error("Final panel already evaluated")
        report = {"round": previous, "panel": fixed_eval, "protocol": header,
                  "test_ascension": args.ascension, "cohorts": {}}
        for name in ("anchor", *variants):
            rows, _ = play(base, name, weights.get(name), fixed_eval, args)
            report["cohorts"][name] = {"summary": summarize(rows), "rows": rows}
        write_json(destination, report)
        print(destination)
        return
    if not (args.out / "anchor.json").exists():
        rows, _ = play(base, "anchor", None, dev, args)
        write_json(args.out / "anchor.json", {"protocol": header, "summary": summarize(rows), "rows": rows})
    for number in range(previous + 1, args.rounds + 1):
        start = perf_counter()
        seeds = panel(f"train/{number}", args.games, args.seed_key)
        assert set(seeds).isdisjoint(dev)
        report = {"round": number, "train_panel": seeds, "development_panel": dev, "cohorts": {}}
        for name, values in weights.items():
            training, samples = play(base, name, values, seeds, args, training=True)
            changed = False
            if samples:
                baseline = statistics.mean(score for score, _ in samples) if len(samples) > 1 else .05
                direction = np.mean([(score - baseline) * gradient for score, gradient in samples], axis=0)
                updated = np.clip(values + 0.6 * direction, -2.0, 2.0)
                changed = not np.array_equal(values, updated)
                values[:] = updated
            cohort = {"valid_training_episodes": len(samples), "optimizer_steps": int(bool(samples)),
                      "weights_changed": changed, "train": summarize(training), "train_rows": training}
            if number % args.eval_every == 0 or number == args.rounds:
                evaluated, _ = play(base, name, values, dev, args)
                cohort.update(eval=summarize(evaluated), eval_rows=evaluated)
            report["cohorts"][name] = cohort
        report["elapsed_seconds"] = perf_counter() - start
        report["sampled_strategic_decisions"] = sum(row["sampled_strategic_decisions"]
            for cohort in report["cohorts"].values() for row in cohort["train_rows"])
        write_json(args.out / f"round-{number:04}.json", report)
        checkpoint(state_file, header, number, weights)
        print(f"Round {number} saved in {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
