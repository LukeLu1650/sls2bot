"""Compare strategic scores and first action changes on frozen anchor trajectories."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from beta_build import CHECKOUT, HOST, MANIFEST, sha, verify_host
from beta_env import run_episode
from beta_eval import write_json
from strategic_features import VARIANTS, features
from train_strategy import RunPolicy, load_base, np


class CachedBase:
    """Reuse the anchor logits when its RunPolicy advances the card-selection context."""
    def __init__(self, model):
        self.model = model
        self.logits = None

    def __call__(self, state, actions):
        return self.model(state, actions)

    def scores(self, state, actions):
        return self.logits


def label(action, state):
    args = action.get("args", {})
    lookup = {
        "select_card_reward": ("cards", "card_index"), "buy_card": ("cards", "card_index"),
        "choose_option": ("options", "option_index"), "buy_relic": ("relics", "relic_index"),
        "buy_potion": ("potions", "potion_index"),
    }
    kind = action["action"]
    if kind in lookup:
        group, key = lookup[kind]
        record = next((item for item in state.get(group) or [] if item.get("index") == args.get(key)), {})
        name = record.get("title") or record.get("name") or record.get("option_id")
        if name:
            return f"{kind}: {name}"
    if kind == "select_map_node":
        record = next((node for node in state.get("choices") or []
                       if node.get("col") == args["col"] and node.get("row") == args["row"]), {})
        return f"{kind}: {record.get('type', '?')} ({args['col']},{args['row']})"
    return f"{kind} {args}" if args else kind


def percentiles(numbers):
    return {name: round(float(value), 8) for name, value in zip(
        ("minimum", "median", "p90", "maximum"), np.percentile(numbers, (0, 50, 90, 100)), strict=True
    )} if numbers else {}


def compare(folder, limit, output):
    host = verify_host()
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    checkpoint = json.loads((folder / "checkpoint.json").read_bytes())
    anchor = json.loads((folder / "anchor.json").read_bytes())
    protocol = checkpoint["protocol"]
    for name, digest in protocol["sources"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Checkpoint feature code differs: {name}")
    same_host = protocol["host"] == sha(MANIFEST)
    model = load_base()
    weights = {name: np.asarray(checkpoint["weights"][name], dtype=np.float32) for name in VARIANTS}
    sampled = anchor["rows"][:limit]
    by_seed = []
    gaps, ranges = [], {name: [] for name in VARIANTS}
    changes = {name: Counter() for name in VARIANTS}
    scenes = Counter()
    for row in sampled:
        base = CachedBase(model)
        policy = RunPolicy(base, "anchor", None)
        first_change = {}
        step = 0

        def inspect(state, actions):
            nonlocal step
            step += 1
            if state["decision"] != "combat_play" and len(actions) > 1:
                base.logits = model.scores(state, actions)
                order = np.argsort(-base.logits, kind="stable")
                selected, runner_up = map(int, order[:2])
                gap = float(base.logits[selected] - base.logits[runner_up])
                gaps.append(gap)
                scene = state["decision"]
                scenes[scene] += 1
                for name, params in weights.items():
                    delta = features(state, actions, name, policy.reason) @ params
                    ranges[name].append(float(np.ptp(delta)))
                    choice = int(np.argmax(base.logits + delta))
                    if choice != selected:
                        changes[name][scene] += 1
                        if name not in first_change:
                            first_change[name] = {
                                "step": step, "scene": scene, "floor": state["context"].get("floor"),
                                "anchor": label(actions[selected], state), "variant": label(actions[choice], state),
                                "base_gap": float(base.logits[selected] - base.logits[choice]),
                                "adjustment_difference": float(delta[choice] - delta[selected]),
                            }
            return policy(state, actions)

        played = run_episode(HOST, row["seed"], "standard", protocol["ascension"], inspect,
                             max_steps=protocol["max_steps"], episode_timeout=protocol["episode_timeout"],
                             strategic_observations=True)
        matches = all(played[key] == row[key] for key in ("status", "floor", "hp", "steps", "actions"))
        by_seed.append({"seed": row["seed"], "matches_saved_anchor": matches,
                        "status": played["status"], "floor": played["floor"], "first_change": first_change})
        print(f"{row['seed']}: {played['status']} floor={played['floor']} "
              f"anchor_match={matches} changed={','.join(first_change) or '-'}", flush=True)
    report = {
        "round": checkpoint["round"], "same_host_as_checkpoint": same_host,
        "host_revision": host["host_revision"], "strategic_decisions": sum(scenes.values()),
        "decisions_by_scene": dict(scenes), "anchor_top_two_margin": percentiles(gaps),
        "cohorts": {name: {"choice_changes": sum(changes[name].values()),
                            "changes_by_scene": dict(changes[name]),
                            "starting_from_anchor_adjustment_range": percentiles(ranges[name])}
                    for name in VARIANTS},
        "seeds": by_seed,
    }
    print(json.dumps({key: value for key, value in report.items() if key != "seeds"}, indent=2), flush=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 1 <= args.games <= 32:
        parser.error("--games must be between 1 and 32")
    compare(args.folder, args.games, args.out)
