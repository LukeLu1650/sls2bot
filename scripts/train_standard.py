"""Continue the A0 Necrobinder readout and record bounded zero-search evaluations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from beta_eval import BUILD, CHECKOUT, HOST, ROOT, RulePolicy, verify_host, write_json
from beta_env import action_cards, run_episode
from necro_policy import NecroPolicy, np


TRAIN = ("N4JBK7A6XJUC", "UFQLS6D9FWN5", "54VBJJXP7UA9", "V4TNAHLSWFA8")
DEV = ("D5R6LJ3YBXP4", "WF5GF6QHXBUP", "96KBYZR5SR28", "YQP8RN5NHMW4",
       "AXTMZZV65F3X", "HJXLBBEKZW7R", "UTX7QGLY5HE2", "UFTFHN52UZ45")
BASE = ROOT / "results/windows-v0.113.1/standard-a0/imitation"


def label(record):
    return str(record.get("name") or record.get("title") or record.get("id") or "—")


def find(items, index):
    return next((item for item in (items or []) if item.get("index") == index), {})


def incoming(state):
    return sum(intent.get("total_damage", intent.get("damage", 0) * intent.get("hits", 1))
               for enemy in state.get("enemies", []) for intent in enemy.get("intents") or [])


def action_features(state, action, threat):
    """Visible card previews and a targeted creature, used by teacher and student."""
    kind = action["action"]
    args = action.get("args", {})
    if kind != "play_card":
        return [0.] * 7
    card = find(state.get("hand"), args.get("card_index"))
    stats = card.get("stats") or {}
    enemy = find(state.get("enemies"), args.get("target_index"))
    target = next((row for row in card.get("damage_by_target") or []
                   if row.get("target_index") == args.get("target_index")), {})
    damage = target.get("total_damage", target.get("damage", stats.get("calculateddamage", stats.get("damage", 0))))
    net = max(0, damage - enemy.get("block", 0))
    kill = bool(enemy and net >= enemy["hp"])
    shield = min(threat, stats.get("block", 0))
    summon = stats.get("summon", 0)
    doom = stats.get("doom", 0)
    return [net, kill, shield, summon, doom, card.get("cost", 0), stats.get("cards", 0)]


class Teacher:
    def __init__(self, index):
        self.other = RulePolicy(index)

    def __call__(self, state, actions):
        if state["decision"] != "combat_play":
            return self.other(state, actions)
        threat = max(0, incoming(state) - state["player"].get("block", 0))

        def priority(action):
            if action["action"] != "play_card":
                return 0 if action["action"] == "end_turn" else -2
            damage, kill, block, summon, doom, cost, draw = action_features(state, action, threat)
            return 1 + (damage + 22 * kill + 1.25 * block + summon + .7 * doom + draw) / max(1, cost)

        return max(range(len(actions)), key=lambda i: priority(actions[i]))


class StandardPolicy(NecroPolicy):
    """Unchanged backbone shape with additional visible card features."""
    @staticmethod
    def tokens(state, actions):
        tokens = NecroPolicy.tokens(state, actions)
        if state["decision"] == "combat_play":
            threat = max(0, incoming(state) - state["player"].get("block", 0))
            for token, action in zip(tokens[-len(actions):], actions):
                names = ("damage", "kill", "useful_block", "summon", "doom", "cost", "draw")
                for name, value in zip(names, action_features(state, action, threat)):
                    if value:
                        token.append((f"preview/{name}", math.copysign(math.log1p(abs(value)), value)))
        return tokens


def load_base(model):
    with np.load(BASE / "readout.npz", allow_pickle=False) as data:
        model.readout[:] = data["readout"]
        model.readout_bias[:] = data["bias"]
    if model.fingerprint() != json.loads((BASE / "protocol.json").read_bytes())["model_sha256"]:
        raise ValueError("A0 baseline fingerprint differs")
    return model


def fit(model, examples, epochs):
    history = []
    rng = np.random.default_rng(6110)
    for _ in range(epochs):
        losses = []
        for index in rng.permutation(len(examples)):
            candidates, chosen = examples[index]
            scores = candidates @ model.readout
            shifted = scores - scores.max()
            weights = np.exp(shifted)
            losses.append(float(np.log(weights.sum()) - shifted[chosen]))
            weights /= weights.sum()
            weights[chosen] -= 1
            model.readout[:] -= np.float32(.002) * (weights @ candidates)
        history.append(sum(losses) / len(losses))
    if not np.isfinite(model.readout).all() or not np.isfinite(history).all():
        raise ValueError("Nonfinite training")
    return history


def brief(state, actions, logits, chosen):
    def describe(action):
        kind, args = action["action"], action.get("args", {})
        if kind == "play_card":
            name = label(find(state.get("hand"), args["card_index"]))
            if "target_index" in args:
                name += " → " + label(find(state.get("enemies"), args["target_index"]))
            return name
        if kind in {"pick_card", "select_cards", "select_bundle", "confirm_selection"}:
            names = ", ".join(label(card) + ("+" if card.get("upgraded") else "")
                              for card in action_cards(state, action))
            if kind == "confirm_selection":
                return f"Confirm · {names}" if names else "Skip selection"
            return f"Select · {names}" if names else "Select cards"
        if kind == "skip_select":
            return "Skip selection"
        if "option_index" in args:
            return label(find(state.get("options"), args["option_index"]))
        if "card_index" in args:
            return label(find(state.get("cards"), args["card_index"]))
        if kind == "select_map_node":
            room = next((n["type"] for n in state.get("choices", [])
                         if n["col"] == args["col"] and n["row"] == args["row"]), "Room")
            return f"{room} [{args['col'] + 1}, {args['row'] + 1}]"
        if kind == "use_potion":
            return label(find(state["player"].get("potions"), args["potion_index"]))
        return kind.replace("_", " ").capitalize()

    top = np.argsort(-logits, kind="stable")[:5]
    player, osty = state["player"], state.get("osty") or {}
    return dict(scene=state["decision"], floor=state["context"].get("floor"),
                hp=player.get("hp"), max_hp=player.get("max_hp"), block=player.get("block"),
                osty=osty.get("hp"), energy=state.get("energy"), round=state.get("round"),
                enemies=[dict(name=label(e), hp=e["hp"], max_hp=e.get("max_hp"), block=e.get("block"),
                              intents=[dict(kind=i["type"], damage=i.get("total_damage", i.get("damage")))
                                       for i in e.get("intents") or []])
                         for e in state.get("enemies") or []],
                hand=[dict(name=label(c), cost=c["cost"], playable=c["can_play"]) for c in state.get("hand") or []],
                selected=describe(actions[chosen]),
                scores=[dict(label=describe(actions[i]), score=round(float(logits[i]), 4), chosen=i == chosen)
                        for i in top])


def evaluate(model, seed, index, ascension, trace=None, first_combat=True):
    if model is None:
        choose = Teacher(index)
    else:
        def choose(state, actions):
            logits = model.scores(state, actions)
            selected = int(np.argmax(logits))
            if trace is not None:
                # JSON bool requires a plain Python scalar.
                frame = brief(state, actions, logits, selected)
                for candidate in frame["scores"]:
                    candidate["chosen"] = bool(candidate["chosen"])
                trace.append(frame)
            return selected
    return run_episode(HOST, seed, "standard", ascension, choose, max_steps=320,
                       episode_timeout=35, first_combat_only=first_combat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    host = verify_host()
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    args.out.mkdir(parents=True, exist_ok=False)
    base = load_base(NecroPolicy())
    model = load_base(StandardPolicy())
    examples, training = [], []
    for index, seed in enumerate(TRAIN):
        teacher = Teacher(200 + index)

        def collect(state, actions):
            picked = teacher(state, actions)
            if len(examples) < 512 and len(actions) > 1:
                examples.append((model._candidate_embeddings(state, actions), picked))
            return picked

        row = run_episode(HOST, seed, "standard", 0, collect, max_steps=160,
                          episode_timeout=30, first_combat_only=True)
        training.append(row)
        print(f"train {seed} {row['status']} / {len(examples)} decisions", flush=True)
        if row["status"] in {"policy_error", "engine_error"}:
            write_json(args.out / "training-error.json", row)
            raise RuntimeError(row["error"])
    if not examples:
        raise RuntimeError("No demonstrations collected")
    losses = fit(model, examples, epochs=10)
    np.savez(args.out / "readout.npz", readout=model.readout, bias=model.readout_bias)
    with np.load(args.out / "readout.npz", allow_pickle=False) as data:
        model = load_base(StandardPolicy())
        model.readout[:] = data["readout"]
        model.readout_bias[:] = data["bias"]
    report = dict(build=BUILD, game_commit=host["game_commit"], host_revision=host["host_revision"],
                  mode="standard", parameter_count=model.parameter_count, model_sha256=model.fingerprint(),
                  encoder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  training=training, demonstrations=len(examples), epochs=len(losses), losses=losses,
                  dev_seeds=DEV, cohorts={})
    for name, policy in (("base", base), ("teacher", None), ("student", model)):
        cohort = [evaluate(policy, seed, index, 0) for index, seed in enumerate(DEV)]
        report["cohorts"][name] = cohort
        print(f"{name}: {sum(r['status'] == 'combat_cleared' for r in cohort)}/{len(DEV)} first fights", flush=True)
        write_json(args.out / "report.json", report)
    # Fixed early development seeds for the published replay and full-run spot check.
    report["showcase_runs"] = []
    episodes = []
    for index, (seed, ascension) in enumerate(((DEV[0], 0), (DEV[1], 0), (DEV[0], 10))):
        frames = []
        result = evaluate(model, seed, index, ascension, frames, first_combat=False)
        report["showcase_runs"].append(result)
        if frames:
            episodes.append(dict(seed=seed, ascension=ascension, result=result["status"], frames=frames))
        print(f"showcase A{ascension} {seed} {result['status']} floor={result['floor']}", flush=True)
    write_json(args.out / "report.json", report)
    write_json(args.out / "replay.json", dict(schema=1, build=BUILD, model_sha256=model.fingerprint(),
                                            demonstrations=len(examples), epochs=len(losses),
                                            parameters=model.parameter_count, episodes=episodes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
