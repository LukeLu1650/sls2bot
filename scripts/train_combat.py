"""Train a zero-search combat head in the native Standard A0 game."""
import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import statistics

from beta_build import BUILD, CHECKOUT, HOST, MANIFEST, sha, verify_host
from beta_env import action_cards, is_combat_decision, run_episode
from beta_eval import write_json
from strategic_features import MEASURES, bucket, card_values, probabilities
from strategy_updates import training_rng
from train_standard import StandardPolicy, action_features, find, incoming, np
from train_strategy import BASE, load_base, panel, positive_int


# The first block adjusts the existing candidate readout only during combat.
# The second encodes resource interactions and distinguishes individual potions.
SIZE = StandardPolicy.width + 512
COMBAT_FEATURES_VERSION = 1
COMBAT_SCORE_SOURCES = ("train_combat.py", "train_standard.py", "necro_policy.py", "beta_env.py",
                        "strategic_features.py")


def combat_inputs(base, state, actions, *, embeddings=None):
    if embeddings is None:
        embeddings = base._candidate_embeddings(state, actions)
    logits = embeddings @ base.readout + base.readout_bias
    matrix = np.zeros((len(actions), SIZE), dtype=np.float32)
    matrix[:, :base.width] = embeddings
    player, context = state["player"], state["context"]
    potions = player.get("potions") or []
    threat = max(0, incoming(state) - player.get("block", 0))
    exposed = max(0, threat - ((state.get("osty") or {}).get("hp") or 0))
    energy = state.get("energy") or 0
    selection = state["decision"] != "combat_play"
    trigger = state.get("selection_trigger") or {}
    source = trigger.get("id") or trigger.get("name") or trigger.get("action")
    for row, action in zip(matrix, actions, strict=True):
        kind, args = action["action"], action.get("args", {})
        def add(key, value=1):
            row[base.width + bucket(key) % 512] += np.clip(value, -4, 4)

        add(kind)
        add(kind + "/exposure", exposed / 10)
        add(kind + "/low-hp", 1 - player["hp"] / max(1, player["max_hp"]))
        add(kind + "/" + str(context.get("room_type")))
        add(kind + "/energy", energy / 3)
        if selection:
            add(f"{kind}/bounds/{state.get('min_select', 1)}/{state.get('max_select', 1)}")
            add(f"{kind}/already-picked/{len(state.get('picked_indices') or [])}")
            if source:
                add(f"{kind}/trigger/{source}")
            for card in action_cards(state, action):
                identity = card.get("id") or card.get("name")
                zone = card.get("zone") or "offered"
                add(f"{kind}/card/{identity}")
                add(f"{kind}/zone/{zone}")
                for key, value in zip(MEASURES, card_values(card), strict=True):
                    if value:
                        add(f"{kind}/value/{key}", value)
                        add(f"{kind}/{zone}/value/{key}", value)
                        if source:
                            add(f"{kind}/{source}/value/{key}", value)
                        if exposed:
                            add(f"{kind}/threat/value/{key}", min(2, exposed / 10) * value)
        elif kind == "use_potion":
            potion = find(potions, args["potion_index"])
            identity = str(potion.get("id") or potion.get("name"))
            add("potion/" + identity)
            add("potion/" + identity + "/" + str(context.get("room_type")))
            add("potion/" + identity + "/exposure", exposed / 10)
            add("potion/full", len(potions) >= player.get("potion_capacity", 3))
        elif kind == "play_card":
            damage, kill, block, summon, doom, cost, draw = action_features(state, action, threat)
            add("damage", damage / 10)
            add("kill", kill)
            add("useful-shield", min(exposed, block) / 8)
            add("useful-summon", min(exposed, summon) / 8)
            add("doom", doom / 10)
            add("draw", draw / 2)
            if cost >= energy > 0:
                add("last-energy/shield", min(exposed, block) / 8)
                add("last-energy/exposed-nonlethal", exposed / 10 if damage and not kill else 0)
    return logits, matrix


@dataclass
class Fight:
    act: int
    floor: int
    room: str
    hp_in: int
    max_hp: int
    hp_out: int | None = None
    choices: int = 0
    changed: int = 0
    potion_uses: int = 0
    potion_options: int = 0
    selection_choices: int = 0
    selection_changed: int = 0
    gradient: np.ndarray = field(default_factory=lambda: np.zeros(SIZE, np.float32), repr=False)

    def record(self):
        return {key: value for key, value in vars(self).items() if key != "gradient"}


class CombatPolicy:
    def __init__(self, base, weights, rng=None, temperature=0.7):
        self.base, self.weights = base, weights
        self.rng, self.temperature = rng, temperature
        self.fights = []
        self.active = None

    def close(self, hp):
        if self.active is not None:
            self.active.hp_out = hp
            self.active = None

    def handles(self, state):
        room = None if self.active is None else (self.active.act, self.active.floor)
        return is_combat_decision(state, room)

    def __call__(self, state, actions):
        scene, context, player = state["decision"], state["context"], state["player"]
        if not self.handles(state):
            self.close(player["hp"])
            return self.base(state, actions)
        room = context["act"], context["floor"]
        if self.active is None or room != (self.active.act, self.active.floor):
            self.close(player["hp"])
            self.active = Fight(*room, context.get("room_type"), player["hp"], player["max_hp"])
            self.fights.append(self.active)
        if len(actions) == 1:
            return 0
        logits, matrix = combat_inputs(self.base, state, actions)
        scores = logits + matrix @ self.weights
        greedy = int(np.argmax(scores))
        fight = self.active
        changed = int(greedy != np.argmax(logits))
        fight.choices += 1
        fight.changed += changed
        if scene != "combat_play":
            fight.selection_choices += 1
            fight.selection_changed += changed
        fight.potion_options += int(any(a["action"] == "use_potion" for a in actions))
        selected = greedy
        if self.rng is not None:
            probs = probabilities(scores.astype(np.float64) / self.temperature)
            selected = int(self.rng.choice(len(actions), p=probs))
            fight.gradient += (matrix[selected] - probs @ matrix) / self.temperature
        fight.potion_uses += int(actions[selected]["action"] == "use_potion")
        return selected


def fight_samples(fights, result):
    """Return-to-go from this encounter through later fights and the terminal outcome."""
    if result["status"] not in {"victory", "defeat"}:
        return []
    future = 2.0 if result["victory"] else -1.0
    next_act = result["act"]
    samples = []
    for fight in reversed(fights):
        # No reward for individual blocks, attacks, potion uses or unused inventory.
        health_change = (fight.hp_out - fight.hp_in) / max(1, fight.max_hp)
        cleared = fight.hp_out > 0
        local = 0.18 * cleared + 0.65 * health_change + 1.5 * (next_act > fight.act)
        future = local + 0.94 * future
        if fight.choices:
            samples.append((future, fight.gradient, fight.choices))
        next_act = fight.act
    return samples


def play(base, weights, seeds, repeats, training):
    rows, samples = [], []
    for seed in seeds:
        for repeat in range(repeats):
            rng = training_rng(seed, repeat) if training else None
            policy = CombatPolicy(base, weights, rng)
            row = run_episode(HOST, seed, "standard", 0, policy, episode_timeout=100,
                              max_steps=1500, strategic_observations=True, sequential_combat_selection=True)
            if row["status"] in {"victory", "defeat"}:
                policy.close(row["hp"])
                if training:
                    samples.extend(fight_samples(policy.fights, row))
            row["fights"] = [fight.record() for fight in policy.fights]
            row["rollout"] = repeat
            rows.append(row)
            print(f"{'train' if training else 'eval '} {seed}/{repeat+1} {row['status']} "
                  f"act={row['act']} floor={row['floor']} fights={len(policy.fights)}", flush=True)
    return rows, samples


def update(weights, samples):
    if len(samples) < 2:
        return weights.copy()
    returns = np.array([value for value, _, _ in samples])
    advantages = (returns - returns.mean()) / max(returns.std(), 0.01)
    gradient = sum(a * g for a, (_, g, _) in zip(advantages, samples, strict=True))
    gradient /= sum(count for _, _, count in samples)
    norm = np.linalg.norm(gradient)
    changed = weights + gradient * min(0.8, 0.18 / max(norm, 1e-10))
    if not np.isfinite(changed).all():
        raise ValueError("Nonfinite combat weight update")
    return changed.astype(np.float32)


def metrics(rows):
    valid = [r for r in rows if r["status"] in {"defeat", "victory"}]
    fights = [f for r in valid for f in r["fights"] if f["act"] == 1]
    losses = [f["hp_in"] - f["hp_out"] for f in fights if f["hp_out"] > 0]
    entries = [b for r in valid for b in r["boss_entries"] if b["act"] == 1]
    return dict(attempts=len(rows), valid=len(valid), statuses=dict(Counter(r["status"] for r in rows)),
                act1_boss_runs=sum(any(b["act"] == 1 for b in r["boss_entries"]) for r in valid),
                median_boss_entry_hp=statistics.median((b["before_travel"] or b)["hp"] for b in entries) if entries else None,
                act1_clears=sum(r["act"] > 1 or r["victory"] for r in valid), wins=sum(r["victory"] for r in valid),
                act1_fights=len(fights), act1_fights_survived=len(losses),
                median_net_hp_lost_survived=statistics.median(losses) if losses else None,
                decisions=sum(f["choices"] for f in fights), greedy_changes=sum(f["changed"] for f in fights),
                selection_decisions=sum(f["selection_choices"] for f in fights),
                selection_changes=sum(f["selection_changed"] for f in fights),
                potion_decisions=sum(f["potion_options"] for f in fights), potions_used=sum(f["potion_uses"] for f in fights))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rounds", type=positive_int, default=4, help="Total including completed rounds")
    parser.add_argument("--games", type=positive_int, default=8, help="Fresh training seeds per round")
    parser.add_argument("--rollouts-per-seed", type=positive_int, default=2)
    parser.add_argument("--eval-games", type=positive_int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if verify_host().get("selection_schema") != 1:
        parser.error("Rebuild the host with scripts/beta_build.py for combat selection snapshots")
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    base = load_base()
    protocol = dict(build=BUILD, mode="standard", ascension=0, combat_features=COMBAT_FEATURES_VERSION,
                    host=sha(MANIFEST), base=sha(BASE / "readout.npz"),
                    games=args.games, repeats=args.rollouts_per_seed, eval_games=args.eval_games,
                    sources={p: sha(Path(__file__).with_name(p)) for p in (
                        *COMBAT_SCORE_SOURCES, "train_strategy.py", "strategy_updates.py")})
    path = args.out / "checkpoint.json"
    weights = np.zeros(SIZE, np.float32)
    previous = 0
    if args.resume:
        saved = json.loads(path.read_bytes())
        if saved["protocol"] != protocol:
            parser.error("Resume requires the same protocol and training settings")
        weights = np.asarray(saved["weights"], np.float32)
        if weights.shape != (SIZE,) or not np.isfinite(weights).all():
            parser.error("Invalid combat weights")
        previous = saved["round"]
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(path, dict(protocol=protocol, round=0, weights=weights.tolist()))
    dev = panel("development", args.eval_games, "necro-standard")
    if not (args.out / "anchor.json").exists():
        rows, _ = play(base, np.zeros_like(weights), dev, 1, False)
        write_json(args.out / "anchor.json", dict(summary=metrics(rows), rows=rows))
    for number in range(previous + 1, args.rounds + 1):
        seeds = panel(f"combat/train/{number}", args.games, "necro-standard")
        assert set(seeds).isdisjoint(dev)
        rows, samples = play(base, weights, seeds, args.rollouts_per_seed, True)
        updated = update(weights, samples)
        evaluated, _ = play(base, updated, dev, 1, False)
        report = dict(round=number, train=metrics(rows), eval=metrics(evaluated), train_rows=rows,
                      eval_rows=evaluated, training_fights=len(samples),
                      step_l2=float(np.linalg.norm(updated - weights)))
        write_json(args.out / f"round-{number:04}.json", report)
        weights = updated
        write_json(path, dict(protocol=protocol, round=number, weights=weights.tolist()))
        print(f"Round {number} step={report['step_l2']:.4f} eval={report['eval']}", flush=True)


if __name__ == "__main__":
    main()
