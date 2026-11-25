"""Joint on-policy training of Standard A0 combat and run strategy."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from beta_build import BUILD, CHECKOUT, HOST, MANIFEST, sha, verify_host
from beta_env import run_episode
from beta_eval import write_json
from strategic_features import FEATURE_COUNT
from strategy_updates import training_rng, update_config, update_weights
from train_combat import COMBAT_FEATURES_VERSION, CombatPolicy, SIZE, fight_samples, metrics, update
from train_strategy import BASE, STRATEGY_FILES, RunPolicy, load_base, np, panel, positive_int, reward


# These revisions differ only in floor-entry reporting.
REPORTING_COMPATIBLE_TRAINERS = {
    "835e8721469369095670e8274620a8408ecd98e4f86e716b21e05461c46795a5",
    "4e87f54c1fd7d57b751fbc52ccbccbd670c2313c9292ff8adec76b9857eabe04",
}


class JointPolicy:
    def __init__(self, base, weights, rng, settings):
        self.combat = CombatPolicy(base, weights["combat"], rng)
        self.strategy = RunPolicy(base, "boss_deck", weights["strategy"], rng, temperature=settings.temperature)
        self.decisions = Counter()

    def __call__(self, state, actions):
        scene = state["decision"]
        if len(actions) > 1:
            self.decisions[scene] += 1
        if self.combat.handles(state):
            self.strategy.reason = None
            return self.combat(state, actions)
        self.combat.close(state["player"]["hp"])
        return self.strategy(state, actions)


class FloorEntryLog:
    """Capture HP at the first observed decision on each visited floor."""
    def __init__(self, policy):
        self.policy = policy
        self.entries = {}
        self.before_travel = None

    def __call__(self, state, actions):
        context = state["context"]
        room = context.get("act"), context.get("floor")
        player = state["player"]
        health = {key: player.get(key) for key in ("hp", "max_hp")}
        if None not in room and room not in self.entries:
            self.entries[room] = dict(act=room[0], floor=room[1], **health,
                                      room_type=context.get("room_type"), decision=state["decision"],
                                      before_travel=self.before_travel)
        self.before_travel = None
        index = self.policy(state, actions)
        if actions[index]["action"] == "select_map_node":
            self.before_travel = health
        return index


def show_hp(snapshot):
    if not snapshot or snapshot.get("hp") is None:
        return "-"
    return f"{snapshot['hp']}/{snapshot['max_hp']}" if snapshot.get("max_hp") else str(snapshot["hp"])


def play(base, weights, seeds, repeats, settings, training):
    rows, battle_samples, strategy_samples = [], [], []
    for seed in seeds:
        for index in range(repeats):
            rng = training_rng(seed, index) if training else None
            policy = JointPolicy(base, weights, rng, settings)
            entry_log = FloorEntryLog(policy)
            row = run_episode(HOST, seed, "standard", 0, entry_log, episode_timeout=100, max_steps=1500,
                              strategic_observations=True, sequential_combat_selection=True)
            if row["status"] in {"victory", "defeat"}:
                policy.combat.close(row["hp"])
                if training:
                    battle_samples.extend(fight_samples(policy.combat.fights, row))
                    strategy = policy.strategy
                    if strategy.gradients:
                        strategy_samples.append((reward(row), np.mean(strategy.gradients, axis=0),
                                                 strategy.max_feature_pair_bound))
            entry = entry_log.entries.get((row["act"], row["floor"]))
            row.update(rollout=index, choices_by_scene=dict(policy.decisions), last_floor_entry=entry,
                       fights=[f.record() for f in policy.combat.fights])
            rows.append(row)
            print(f"{'train' if training else 'eval'} {seed}/{index+1} {row['status']} "
                  f"act={row['act']} floor={row['floor']} "
                  f"before_travel_hp={show_hp(entry['before_travel'] if entry else None)}", flush=True)
    return rows, battle_samples, strategy_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=positive_int, default=4, help="Total rounds including previous updates")
    parser.add_argument("--games", type=positive_int, default=8)
    parser.add_argument("--rollouts-per-seed", type=positive_int, default=2)
    parser.add_argument("--eval-games", type=positive_int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if verify_host().get("selection_schema") != 1:
        parser.error("Rebuild the host with scripts/beta_build.py for combat selection snapshots")
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    base = load_base()
    protocol = dict(game=BUILD, mode="standard", ascension=0, combat_features=COMBAT_FEATURES_VERSION,
                    host=sha(MANIFEST), base=sha(BASE / "readout.npz"),
                    games=args.games, repeats=args.rollouts_per_seed, eval_games=args.eval_games,
                    sources={p: sha(Path(__file__).with_name(p)) for p in
                             (*STRATEGY_FILES, "train_combat.py", "train_joint.py")})
    checkpoint = args.out / "checkpoint.json"
    weights = dict(combat=np.zeros(SIZE, np.float32), strategy=np.zeros(FEATURE_COUNT, np.float32))
    def save(number):
        write_json(checkpoint, dict(protocol=protocol, round=number,
                                    weights={k: v.tolist() for k, v in weights.items()}))
    previous = 0
    if args.resume:
        saved = json.loads(checkpoint.read_bytes())
        prior = saved["protocol"]
        if prior["sources"].get("train_joint.py") in REPORTING_COMPATIBLE_TRAINERS:
            prior["sources"]["train_joint.py"] = protocol["sources"]["train_joint.py"]
        if prior != protocol:
            parser.error("Resume with the same joint configuration, host and source revision")
        for key, initial in weights.items():
            values = np.asarray(saved["weights"][key], np.float32)
            if values.shape != initial.shape or not np.isfinite(values).all():
                parser.error(f"Invalid {key} checkpoint")
            weights[key] = values
        previous = saved["round"]
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        save(0)
    dev = panel("development", args.eval_games, "necro-standard")
    anchor = args.out / "anchor.json"
    if not anchor.exists():
        initial = {k: np.zeros_like(v) for k, v in weights.items()}
        rows, _, _ = play(base, initial, dev, 1, update_config("warmup", 1, 12), False)
        write_json(anchor, dict(summary=metrics(rows), rows=rows))
    for number in range(previous + 1, args.rounds + 1):
        settings = update_config("warmup", number, 12)
        seeds = panel(f"joint/train/{number}", args.games, "necro-standard")
        assert set(seeds).isdisjoint(dev)
        rows, combat, strategy = play(base, weights, seeds, args.rollouts_per_seed, settings, True)
        updated = update(weights["combat"], combat)
        combat_step = float(np.linalg.norm(updated - weights["combat"]))
        weights["combat"] = updated
        weights["strategy"], strategic_update = update_weights(weights["strategy"], strategy, settings)
        evaluated, _, _ = play(base, weights, dev, 1, settings, False)
        report = dict(round=number, train=metrics(rows), eval=metrics(evaluated), train_rows=rows, eval_rows=evaluated,
                      combat_fights=len(combat), strategy_episodes=len(strategy), combat_step_l2=combat_step,
                      strategy_update=strategic_update)
        write_json(args.out / f"round-{number:04}.json", report)
        save(number)
        print(f"Round {number} saved. Combat step {combat_step:.5f}, strategy step "
              f"{strategic_update['step_l2']:.5f}", flush=True)


if __name__ == "__main__":
    main()
