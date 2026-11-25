"""Pipe-based inference for the local game Mod. Stdout is reserved for replies."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

from beta_env import is_combat_decision
from strategic_features import FEATURE_COUNT, features
from train_combat import SIZE, combat_inputs
from train_standard import StandardPolicy, np


def validate_combat_weights(weights):
    return validate_weights(weights, SIZE, "combat")


def validate_weights(weights, count, name):
    array = np.asarray(weights, dtype=np.float32)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name} weights")
    return array


def load_weights(directory, manifest, name, count):
    filename = name + ".dat"
    path = directory / filename
    files = (manifest.get("checkpoint") or {}).get("files")
    if files is not None and filename not in files:
        return None
    if files is None and (name != "combat" or not path.exists()):
        return None
    data = path.read_bytes()
    if files is not None and sha256(data).hexdigest() != files[filename]:
        raise ValueError(f"{name.capitalize()} weight fingerprint mismatch")
    return validate_weights(json.loads(data), count, name)


def strategy_inputs(model, state, actions):
    if state["decision"] == "card_select" and "picked_indices" in state:
        # The trainer scores complete index sets outside combat. Score each live
        # continuation as the set that would result from taking that action.
        selected = state["picked_indices"]
        projected = []
        for action in actions:
            kind = action["action"]
            if kind == "pick_card":
                indices = [*selected, action["args"]["card_index"]]
            elif kind == "confirm_selection":
                indices = selected
            else:
                projected.append(action)
                continue
            projected.append({"action": "select_cards", "args": {"indices": ",".join(map(str, sorted(indices)))}} if indices
                             else {"action": "skip_select"})
        state = {key: value for key, value in state.items() if key != "picked_indices"}
        actions = projected
    return model.scores(state, actions), features(state, actions, "boss_deck", state.get("selection_reason"))


def main():
    directory = Path(__file__).resolve().parent
    model = StandardPolicy()
    with np.load(directory / "readout.npz", allow_pickle=False) as data:
        model.readout[:] = data["readout"]
        model.readout_bias[:] = data["bias"]
    manifest = json.loads((directory / "model.dat").read_bytes())
    if manifest["model_sha256"] != model.fingerprint():
        raise ValueError("Policy checkpoint fingerprint mismatch")
    combat = load_weights(directory, manifest, "combat", SIZE)
    strategy = load_weights(directory, manifest, "strategy", FEATURE_COUNT)
    for line in sys.stdin:
        prompt = json.loads(line)
        start = perf_counter()
        state, actions = prompt["state"], prompt["actions"]
        if is_combat_decision(state):
            if combat is None:
                scores = model.scores(state, actions)
            else:
                scores, matrix = combat_inputs(model, state, actions)
                scores = scores + matrix @ combat
        elif strategy is not None and state["decision"] in {
                "map_select", "card_reward", "card_select", "bundle_select", "shop", "rest_site", "event_choice"}:
            scores, matrix = strategy_inputs(model, state, actions)
            scores = scores + matrix @ strategy
        else:
            scores = model.scores(state, actions)
        order = np.argsort(-scores, kind="stable")[:3]
        print(json.dumps({"Index": int(order[0]), "Ms": (perf_counter() - start) * 1000,
                          "Top": [{"Index": int(i), "Score": float(scores[i])} for i in order]},
                         allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
