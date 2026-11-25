"""Record watchable F8 replays from verified second-act runs."""
import argparse
import hashlib
import os
from pathlib import Path

from beta_build import BUILD, CHECKOUT, HOST, MANIFEST, sha, verify_host
from beta_env import run_episode
from beta_eval import write_json
from strategic_features import FEATURE_COUNT, features
from strategy_updates import training_rng, update_config
from train_combat import SIZE, combat_inputs
from train_curriculum import Attempt, checkpoint_weights, read_json
from train_joint import JointPolicy
from train_standard import brief
from train_strategy import load_base, np


def record(base, checkpoint, item):
    weights = checkpoint_weights(checkpoint["weights"])
    settings = update_config("warmup", checkpoint["round"] + checkpoint["schedule_offset"], 12)
    sample = item["sample_key"]
    policy = JointPolicy(base, weights, training_rng(sample, 0) if sample else None, settings)
    attempt = Attempt(policy, item["case"])
    prefix_length = len(item["case"]["prefix"]) if item["case"] else 0
    frames = []

    def observe(state, actions):
        source = "recorded_prefix" if len(frames) < prefix_length else "model"
        scores = np.zeros(len(actions), np.float32)
        if source == "model" and len(actions) > 1:
            if policy.combat.handles(state):
                logits, matrix = combat_inputs(base, state, actions)
                scores = logits + matrix @ weights["combat"]
            else:
                matrix = features(state, actions, "boss_deck", policy.strategy.reason)
                scores = base.scores(state, actions) + matrix @ weights["strategy"]
        elif source == "model":
            source = "forced"
        selected = attempt(state, actions)
        frame = brief(state, actions, scores, selected)
        frame.update(act=state["context"].get("act"), source=source)
        if source == "model":
            for scored in frame["scores"]:
                scored["chosen"] = bool(scored["chosen"])
        else:
            frame["scores"] = []
        frames.append(frame)
        return selected

    result = run_episode(HOST, item["seed"], "standard", 0, observe, episode_timeout=100, max_steps=1500,
                         strategic_observations=True, sequential_combat_selection=True)
    expected = item["result"]
    if (result["act"] < 2 or any(result[key] != expected[key] for key in
                                ("act", "floor", "status", "hp", "steps", "actions"))):
        raise ValueError(f"Could not reproduce the verified second-act run for {item['seed']}: {result}")
    act_two = next(i for i, f in enumerate(frames) if f["act"] == 2)
    first_boss = next(e for e in result["boss_entries"] if e["act"] == 1)
    frames.append(dict(scene="game_over", floor=result["floor"], act=result["act"], hp=result["hp"],
                       max_hp=result["max_hp"], source="terminal", selected=result["status"].title(),
                       enemies=[], hand=[], scores=[]))
    return dict(seed=item["seed"], ascension=0, training_round=checkpoint["round"],
                sample_type="sampled" if sample else "greedy", prefix_decisions=prefix_length,
                result=result["status"], finish_act=result["act"], finish_floor=result["floor"],
                act2_frame=act_two, act1_boss=first_boss["boss"], boss_entry_hp=first_boss["hp"],
                frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Curriculum stage directory")
    parser.add_argument("verified", type=Path, help="Verified rollout manifest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify_host()
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    base = load_base()
    checkpoints = {}
    for kind, file in (("best", "best-checkpoint.json"), ("latest", "checkpoint.json")):
        path = args.source / file
        checkpoint = read_json(path)
        if checkpoint["protocol"]["host"] != sha(MANIFEST):
            parser.error("The checkpoint requires a different headless game host")
        checkpoints[kind] = checkpoint, sha(path)
    episodes, provenance = [], []
    for item in read_json(args.verified)["episodes"]:
        checkpoint, digest = checkpoints[item["model"]]
        if item["round"] != checkpoint["round"]:
            parser.error("The recording manifest was generated against a different checkpoint")
        episodes.append(record(base, checkpoint, item))
        provenance.append(dict(seed=item["seed"], round=item["round"], checkpoint_sha256=digest))
        print(f"{item['seed']}: reached Act II, {len(episodes[-1]['frames'])} frames", flush=True)
    if not episodes:
        parser.error("At least one verified run is required")
    fingerprint = hashlib.sha256("".join(item["checkpoint_sha256"] for item in provenance).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "replay.json", dict(schema=1, build=BUILD, model_sha256=fingerprint,
                                              parameters=base.parameter_count + SIZE + FEATURE_COUNT, episodes=episodes))
    write_json(args.out / "report.json", dict(build=BUILD, model_sha256=fingerprint, provenance=provenance,
                                               episodes=[{k: v for k, v in e.items() if k != "frames"} for e in episodes]))


if __name__ == "__main__":
    main()
