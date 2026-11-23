"""Pipe-based inference for the local game Mod. Stdout is reserved for replies."""
import json
from pathlib import Path
import sys
from time import perf_counter

from train_standard import StandardPolicy, np


def main():
    directory = Path(__file__).resolve().parent
    model = StandardPolicy()
    with np.load(directory / "readout.npz", allow_pickle=False) as data:
        model.readout[:] = data["readout"]
        model.readout_bias[:] = data["bias"]
    expected = json.loads((directory / "model.dat").read_bytes())["model_sha256"]
    if expected != model.fingerprint():
        raise ValueError("Policy checkpoint fingerprint mismatch")
    for line in sys.stdin:
        prompt = json.loads(line)
        start = perf_counter()
        scores = model.scores(prompt["state"], prompt["actions"])
        order = np.argsort(-scores, kind="stable")[:3]
        print(json.dumps({"Index": int(order[0]), "Ms": (perf_counter() - start) * 1000,
                          "Top": [{"Index": int(i), "Score": float(scores[i])} for i in order]},
                         allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
