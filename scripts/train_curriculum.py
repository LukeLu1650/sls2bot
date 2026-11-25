"""Mix fresh Standard A0 runs with verified starts of difficult fights."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from beta_build import BUILD, CHECKOUT, HOST, MANIFEST, sha, verify_host
from beta_env import is_combat_decision, run_episode
from beta_eval import wilson, write_json
from strategy_updates import training_rng, update_config, update_weights
from train_combat import COMBAT_FEATURES_VERSION, SIZE, fight_samples, metrics, update
from train_joint import FloorEntryLog, JointPolicy, show_hp
from train_standard import action_features, incoming
from train_strategy import BASE, STRATEGY_FILES, load_base, np, panel, positive_int, reward
from strategic_features import FEATURE_COUNT


REGRESSION = ("Q8QQWKP240JZ", "YS2C9ALDRA2Y")
SEED_KEY = "necro-standard-curriculum"
PRIORITIES = ("unused_potion", "high_hp_loss", "boss", "lethal_attacker", "scaling_enemy", "potion_kill_line")
PRE_DEV_SELECTION_TRAINER = "4cd7f97d2ca1093303d2128250f29758b835d54a7b52e3d5e4ee27554d0097f1"
PRE_REPEATED_SAMPLING_TRAINER = "747d47d500aa87db7c8c5502c13b44a0ab0b3b7b951f93d0703631ebb0d29311"
PRE_MULTIPLE_FOCUS_TRAINER = "d7a6b517469d213882e5bfbfe705d934219abeca11c4e37cbb0fc77fe30e36d4"
PRE_STEAM_EXPERIMENT_SOURCES = {
    "train_curriculum.py": "fff4dad89b2f7e66407a9dc6bcd43d5385a5c1d23b449d2f4dceb54d96844fda",
    "beta_env.py": "193d6d1dfa1e7053b21413f969f9acd1d2aa2fbbced8ff885c202f79eb0b3531",
    "train_combat.py": "ae9496cabd29584e396fcee0b8c1b4c44a2976f53170c74d96ff41901fafd3bc",
}


def migrate_legacy_sources(protocol, sources):
    # Aligned-host actions and the PPO embedding argument are opt-in for existing callers.
    for name, previous in PRE_STEAM_EXPERIMENT_SOURCES.items():
        if protocol["sources"].get(name) == previous:
            protocol["sources"][name] = sources[name]


def read_json(path):
    return json.loads(path.read_bytes())


def signature(state):
    payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def potion_changes_kill_line(state, actions, enemies, potions, slots):
    """Flag simple visible estimates for Strength and single-target damage potions."""
    strength = max(((potions[i].get("vars") or {}).get("StrengthPower", 0) for i in slots), default=0)
    hand = {card["index"]: card for card in state.get("hand") or []}
    for action in actions:
        kind, args = action["action"], action.get("args", {})
        target = enemies.get(args.get("target_index"))
        if not target:
            continue
        if kind == "use_potion":
            potion = potions[args["potion_index"]].get("vars") or {}
            if potion.get("Damage", 0) >= target["hp"] + target.get("block", 0):
                return True
        elif strength and kind == "play_card":
            card = hand[args["card_index"]]
            stats = card.get("stats") or {}
            if card.get("type") != "Attack" or stats.get("repeat", 1) != 1:
                continue
            net, kill, *_ = action_features(state, action, 0)
            if net and not kill and net + strength >= target["hp"]:
                return True
    return False


class Attempt:
    """Record a fresh trajectory or replay the prefix before a new policy rollout."""
    def __init__(self, policy, case=None):
        self.policy = FloorEntryLog(policy)
        self.case = case
        self.reached_target = case is None
        self.prefix = []
        self.fights = {}
        self.combat_room = None
        self.enemy_strength = {}

    def __call__(self, state, actions):
        if not self.reached_target:
            index = len(self.prefix)
            if index < len(self.case["prefix"]):
                command = self.case["prefix"][index]
                if command not in actions:
                    raise ValueError(f"Curriculum prefix action {index} is unavailable")
                if command["action"] == "select_map_node":
                    player = state["player"]
                    self.policy.before_travel = {key: player.get(key) for key in ("hp", "max_hp")}
                else:
                    self.policy.before_travel = None
                self.prefix.append(command)
                return actions.index(command)
            if signature(state) != self.case["opening"]:
                raise ValueError("Curriculum opening does not match the recorded fight")
            self.reached_target = True

        context, player = state["context"], state["player"]
        room = context["act"], context["floor"]
        fight = None
        if is_combat_decision(state, self.combat_room):
            self.combat_room = room
            if room not in self.fights:
                self.fights[room] = dict(
                    opening=signature(state), prefix_length=len(self.prefix),
                    act=room[0], floor=room[1], room=context.get("room_type"),
                    hp=player["hp"], max_hp=player["max_hp"],
                    enemies=[e["name"] for e in state.get("enemies") or []],
                    lethal_attacker=0, missed_lethal_attacker=0, scaling_enemy=False,
                    damage_cancelled=0, unused_potions=[], last_potion_round=None, potion_kill_line=False,
                )
            fight = self.fights[room]
        else:
            self.combat_room = None

        selected = self.policy(state, actions)
        if self.case is None:
            self.prefix.append(actions[selected])
        if fight is not None and state["decision"] == "combat_play":
            self.observe_combat(fight, self.enemy_strength.setdefault(room, {}), state, actions, selected)
        return selected

    @staticmethod
    def observe_combat(fight, previous_strength, state, actions, selected):
        enemies = {enemy["index"]: enemy for enemy in state["enemies"]}
        potions = {p["index"]: p for p in state["player"].get("potions") or []}
        incoming_by_enemy = {idx: incoming({"enemies": [e]}) for idx, e in enemies.items()}
        available, cancelled = [], {}
        for index, action in enumerate(actions):
            kind, args = action["action"], action.get("args", {})
            if kind == "use_potion":
                available.append(args["potion_index"])
            if kind == "play_card":
                threat = incoming_by_enemy.get(args.get("target_index"), 0)
                if threat and action_features(state, action, 0)[1]:
                    cancelled[index] = threat
        if cancelled:
            fight["lethal_attacker"] += 1
            saved = cancelled.get(selected, 0)
            fight["missed_lethal_attacker"] += int(saved < max(cancelled.values()))
            fight["damage_cancelled"] += int(saved)
        slots = set(available)
        if slots and not fight["potion_kill_line"]:
            fight["potion_kill_line"] = potion_changes_kill_line(state, actions, enemies, potions, slots)
        if actions[selected]["action"] == "use_potion":
            slots.discard(actions[selected]["args"]["potion_index"])
        fight["unused_potions"] = [potions[i]["name"] for i in sorted(slots)]
        fight["last_potion_round"] = state.get("round")
        strength = {}
        for enemy in enemies.values():
            amount = next((p["amount"] for p in enemy.get("powers") or [] if p["name"] == "Strength"), 0)
            name = enemy["name"]
            strength[name] = max(amount, strength.get(name, amount))
        fight["scaling_enemy"] |= any(
            name in previous_strength and strength[name] > previous_strength[name] for name in strength
        )
        previous_strength.update(strength)

    def cases(self, row, number, *, allowed_acts=(1,)):
        if row["status"] not in {"defeat", "victory"}:
            return
        terminal = row["act"], row["floor"]
        last_fight = row["fights"][-1] if row["fights"] else {}
        fights_by_room = {(f["act"], f["floor"]): f for f in row.get("fights", [])}
        for room, fight in self.fights.items():
            if fight["act"] not in allowed_acts:
                continue
            lost_here = (row["status"] == "defeat" and terminal == room
                         and room == (last_fight.get("act"), last_fight.get("floor"))
                         and last_fight["hp_out"] == 0)
            ratio = fight["hp"] / max(1, fight["max_hp"])
            rec = fights_by_room.get(room) or {}
            hp_lost = (rec.get("hp_in", 0) - rec.get("hp_out", 0)) if rec.get("hp_out") is not None else 0
            tags = []
            if lost_here and ratio >= .5 and fight["unused_potions"]:
                tags.append("unused_potion")
            if lost_here and (ratio >= .85 or (fight["act"] >= 2 and ratio >= .35)):
                tags.append("high_hp_loss")
            elif fight["act"] >= 2 and fight["room"] == "Elite" and (lost_here or hp_lost >= 15):
                tags.append("high_hp_loss")
            if fight["room"] == "Boss":
                tags.append("boss")
            if lost_here and fight["missed_lethal_attacker"]:
                tags.append("lethal_attacker")
            if lost_here and fight["scaling_enemy"]:
                tags.append("scaling_enemy")
            if lost_here and fight["potion_kill_line"]:
                tags.append("potion_kill_line")
            if tags:
                yield dict(**fight, seed=row["seed"], tags=tags, discovered=number, uses=0,
                           prefix=self.prefix[:fight["prefix_length"]])


def run(base, weights, seed, settings, *, sample_key=None, case=None, number=0, rollout=None):
    rng = training_rng(sample_key, rollout - 1 if rollout is not None else 0) if sample_key else None
    policy = JointPolicy(base, weights, rng, settings)
    attempt = Attempt(policy, case)
    row = run_episode(HOST, seed, "standard", 0, attempt, episode_timeout=100, max_steps=1500,
                      strategic_observations=True, sequential_combat_selection=True)
    valid = row["status"] in {"victory", "defeat"}
    if valid:
        policy.combat.close(row["hp"])
    row["fights"] = [f.record() for f in policy.combat.fights]
    terminal = row["act"], row["floor"]
    row["last_floor_entry"] = attempt.policy.entries.get(terminal)
    row["fight_diagnostics"] = [{k: v for k, v in f.items() if k not in {"opening", "prefix_length"}}
                                for f in attempt.fights.values()]
    focus = case is not None
    phase = "focus" if focus else "train" if sample_key else "eval"
    row["phase"] = phase
    if focus:
        row["focus"] = {k: case[k] for k in ("act", "floor", "tags", "prefix_length", "hp", "max_hp")}
        row["focus"]["reached"] = attempt.reached_target
    label = seed
    if rollout is not None:
        row["rollout"] = rollout
        label = f"{seed}/{rollout}"
        if focus:
            row["focus"]["opening"] = case["opening"]
    entry = row["last_floor_entry"] or {}
    detail = f" target={case['floor']}:{'+'.join(case['tags'])}" if focus else ""
    print(f"{phase} {label} {row['status']} act={row['act']} floor={row['floor']} "
          f"before_travel_hp={show_hp(entry.get('before_travel'))}{detail}", flush=True)
    combat, strategy, discovered = [], [], []
    if valid and sample_key and attempt.reached_target:
        combat = fight_samples(policy.combat.fights, row)
        if focus:
            # The replayed prefix has no gradients. Keep the target fight's gradient
            # with return-to-go from the rest of this newly sampled trajectory.
            first = next(iter(policy.combat.fights), None)
            combat = combat[-1:] if first and first.choices else []
        else:
            strategy_policy = policy.strategy
            if strategy_policy.gradients:
                strategy.append((reward(row), np.mean(strategy_policy.gradients, axis=0),
                                 strategy_policy.max_feature_pair_bound))
            discovered = list(attempt.cases(row, number))
    return row, combat, strategy, discovered


def choose_case(pool, tag, used_seeds):
    return min((c for c in pool if c["seed"] not in used_seeds),
               key=lambda c: (tag not in c["tags"], c["uses"],
                              min(PRIORITIES.index(t) for t in c["tags"]), -c["discovered"], c["seed"]),
               default=None)


def train_batch(base, weights, pool, attempts, settings, number, protected, *, sampling=None, offset=0):
    rows, combat, strategy = [], [], []
    case_number = number if sampling is None else offset + number
    domain = f"train/{number}" if sampling is None else f"train/repeated/{case_number}"
    fresh_seeds = iter(seed for seed in panel(domain, attempts + len(protected), SEED_KEY)
                       if seed not in protected)

    def fresh(seed, **options):
        row, battle, run_level, cases = run(base, weights, seed, settings, sample_key=seed, number=case_number,
                                           **options)
        rows.append(row)
        combat.extend(battle)
        strategy.extend(run_level)
        pool.extend(cases)

    focus_failures = []
    if sampling is not None:
        for _ in range(sampling["fresh_seeds"]):
            seed = next(fresh_seeds)
            for rollout in range(1, sampling["rollouts_per_seed"] + 1):
                fresh(seed, rollout=rollout)
        used_seeds = set()
        for index in range(sampling.get("focus_cases", 1)):
            tag = PRIORITIES[(number - 1 + index) % 3]
            case = choose_case(pool, tag, used_seeds)
            if case is None:
                break
            seed = case["seed"]
            used_seeds.add(seed)
            key = f"{seed}/focus/repeated/{case['opening']}/{case_number}"
            missed_opening = False
            for rollout in range(1, sampling["focus_rollouts"] + 1):
                row, battle, _, _ = run(base, weights, seed, settings, sample_key=key, case=case, rollout=rollout)
                if row["focus"]["reached"] and row["status"] in {"victory", "defeat"}:
                    rows.append(row)
                    combat.extend(battle)
                    case["uses"] += 1
                else:
                    focus_failures.append(row)
                    missed_opening |= not row["focus"]["reached"]
            if missed_opening:
                pool.remove(case)
        return rows, combat, strategy, retain_cases(pool), focus_failures

    fresh_count = attempts - attempts // 4
    for _ in range(fresh_count):
        fresh(next(fresh_seeds))
    used_seeds = set()
    focus_count = attempts - fresh_count
    for index in range(focus_count):
        if focus_count >= 4:
            tag = PRIORITIES[index] if index < 3 else PRIORITIES[3 + (number + index - 4) % 3]
        else:
            tag = PRIORITIES[((number - 1) * focus_count + index) % len(PRIORITIES)]
        case = choose_case(pool, tag, used_seeds)
        if case is not None:
            seed = case["seed"]
            used_seeds.add(seed)
            key = f"{seed}/focus/{case['opening']}/{number}/{index}"
            row, battle, _, _ = run(base, weights, seed, settings, sample_key=key, case=case)
            if row["focus"]["reached"] and row["status"] in {"victory", "defeat"}:
                case["uses"] += 1
                rows.append(row)
                combat.extend(battle)
                continue
            focus_failures.append(row)
            if not row["focus"]["reached"]:
                pool.remove(case)
        fresh(next(fresh_seeds))
    return rows, combat, strategy, retain_cases(pool), focus_failures


def retain_cases(pool):
    # Bound repeated sampling and preserve examples from each requested category.
    retained = {}
    for tag in PRIORITIES:
        matching = [c for c in pool if tag in c["tags"]]
        for case in sorted(matching, key=lambda c: (c["uses"], -c["discovered"]))[:24]:
            retained[case["seed"], case["opening"]] = case
    return list(retained.values())


def checkpoint_weights(saved):
    result = {}
    for name, size in (("combat", SIZE), ("strategy", FEATURE_COUNT)):
        values = np.asarray(saved[name], np.float32)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} weights")
        result[name] = values
    return result


def summarize(rows):
    if not rows:
        return None
    summary = metrics(rows)
    cleared, count = summary["act1_clears"], summary["attempts"]
    return {**summary, "act1_clear_rate": cleared / count, "act1_clear_wilson95": wilson(cleared, count)}


def development_rank(rows, seeds):
    """Rank complete development panels apart from the two regression seeds."""
    eligible = set(seeds) - set(REGRESSION)
    entries = {row["seed"]: row for row in rows if row["seed"] in eligible}
    if not eligible or set(entries) != eligible:
        return None
    if any(row["status"] not in {"victory", "defeat"} for row in entries.values()):
        return None
    cleared = sum(row["act"] > 1 or row["victory"] for row in entries.values())
    boss = sum(any(entry["act"] == 1 for entry in row["boss_entries"]) for row in entries.values())
    progress = sum(17 if row["act"] > 1 else min(row["floor"] or 0, 17) for row in entries.values())
    return [cleared, boss, progress]


def choose_best(best, number, weights, rows, seeds):
    rank = development_rank(rows, seeds)
    if best is None or (rank is not None and (best["rank"] is None or rank > best["rank"])):
        return dict(round=number, rank=rank, weights=weights)
    return best


def write_best(path, protocol, schedule_offset, best):
    write_json(path, dict(protocol=protocol, round=best["round"], schedule_offset=schedule_offset,
                          development_rank=best["rank"], weights=best["weights"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-from", type=Path, help="Start a new stage from an existing joint checkpoint")
    parser.add_argument("--rounds", type=positive_int, default=8, help="Total rounds in this stage")
    parser.add_argument("--attempts", type=positive_int, help="Legacy attempts per round, a multiple of 4 (default: 16)")
    parser.add_argument("--fresh-seeds", type=positive_int, help="Distinct new game seeds per round")
    parser.add_argument("--rollouts-per-seed", type=positive_int, help="Independent policy rollouts per new game seed")
    parser.add_argument("--focus-cases", type=positive_int, help="Distinct difficult openings per round (default: 1)")
    parser.add_argument("--focus-rollouts", type=positive_int, help="Independent rollouts per verified difficult opening")
    parser.add_argument("--eval-games", type=positive_int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", type=positive_int, metavar="GAMES", help="Evaluate once on the reserved panel")
    args = parser.parse_args()
    sampling = None
    repeated_flags = (args.fresh_seeds, args.rollouts_per_seed, args.focus_rollouts)
    if any(repeated_flags) or args.focus_cases is not None:
        if not all(repeated_flags) or args.attempts is not None:
            parser.error("Set --fresh-seeds, --rollouts-per-seed and --focus-rollouts together without --attempts")
        sampling = dict(fresh_seeds=args.fresh_seeds, rollouts_per_seed=args.rollouts_per_seed,
                        focus_rollouts=args.focus_rollouts)
        if args.focus_cases is not None:
            sampling["focus_cases"] = args.focus_cases
        focus_attempts = sampling.get("focus_cases", 1) * args.focus_rollouts
        attempts = args.fresh_seeds * args.rollouts_per_seed + focus_attempts
    else:
        attempts = args.attempts or 16
        if attempts % 4:
            parser.error("Legacy --attempts must be divisible by 4")
    if (args.start_from and (args.resume or args.test)) or (args.test and args.test > 64):
        parser.error("Use test size at most 64 and --start-from only for a new stage")
    if verify_host().get("selection_schema") != 1:
        parser.error("Rebuild the host with scripts/beta_build.py for combat selection snapshots")
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    source_names = (*STRATEGY_FILES, "train_combat.py", "train_joint.py", "train_curriculum.py")
    sources = {name: sha(Path(__file__).with_name(name)) for name in source_names}
    identity = dict(game=BUILD, mode="standard", ascension=0, combat_features=COMBAT_FEATURES_VERSION,
                    host=sha(MANIFEST), base=sha(BASE / "readout.npz"))
    protocol = dict(**identity, attempts=attempts, eval_games=args.eval_games, sources=sources)
    if sampling is not None:
        protocol["sampling"] = sampling

    dev = sorted(set(panel("development", args.eval_games, "necro-standard")) | set(REGRESSION))
    holdout = panel("final", 64, SEED_KEY)
    protected = set(dev) | set(holdout)
    assert set(dev).isdisjoint(holdout)
    path = args.out / "checkpoint.json"
    if args.resume or args.test:
        saved = read_json(path)
        migrate_legacy_sources(saved["protocol"], sources)
        previous_trainer = saved["protocol"]["sources"].get("train_curriculum.py")
        if sampling is None and previous_trainer in (PRE_DEV_SELECTION_TRAINER, PRE_REPEATED_SAMPLING_TRAINER):
            saved["protocol"]["sources"]["train_curriculum.py"] = sources["train_curriculum.py"]
        if args.focus_cases is None and previous_trainer == PRE_MULTIPLE_FOCUS_TRAINER:
            saved["protocol"]["sources"]["train_curriculum.py"] = sources["train_curriculum.py"]
        if saved["protocol"] != protocol:
            parser.error("Resume using the same curriculum configuration, host and source revision")
    else:
        offset = 0
        initial = dict(combat=[0.0] * SIZE, strategy=[0.0] * FEATURE_COUNT)
        parent = None
        initial_cases = []
        if args.start_from:
            parent_bytes = args.start_from.read_bytes()
            parent = hashlib.sha256(parent_bytes).hexdigest()
            origin = json.loads(parent_bytes)
            origin_protocol = origin["protocol"]
            migrate_legacy_sources(origin_protocol, sources)
            scoring_files = (*STRATEGY_FILES, "train_combat.py")
            if (any(origin_protocol.get(k) != v for k, v in identity.items()) or
                    any(origin_protocol["sources"].get(k) != sources[k] for k in scoring_files)):
                parser.error("--start-from requires a joint checkpoint with compatible scoring and host")
            initial = origin["weights"]
            checkpoint_weights(initial)
            offset = origin["round"] + origin.get("schedule_offset", 0)
            if sampling is not None:
                initial_cases = [case for case in origin.get("cases", []) if case["seed"] not in protected]
        args.out.mkdir(parents=True, exist_ok=False)
        saved = dict(protocol=protocol, round=0, schedule_offset=offset, parent_checkpoint=parent,
                     initial_weights=initial, weights=initial, cases=initial_cases)
        write_json(path, saved)

    base = load_base()
    weights = checkpoint_weights(saved["weights"])
    if args.test:
        destination = args.out / "holdout.json"
        if not saved["round"] or destination.exists():
            parser.error("Finish training before running the one-time held-out evaluation")

    pool = saved["cases"]
    if any(case["seed"] in protected for case in pool):
        parser.error("A saved curriculum case overlaps the evaluation panels")
    anchor = args.out / "anchor.json"
    if not anchor.exists():
        settings = update_config("warmup", saved["schedule_offset"] + 1, 12)
        baseline = checkpoint_weights(saved["initial_weights"])
        rows = [run(base, baseline, seed, settings)[0] for seed in dev]
        write_json(anchor, dict(summary=summarize(rows), rows=rows))
    best = saved.get("best_dev")
    if best is None:
        best = choose_best(None, 0, saved["initial_weights"], read_json(anchor)["rows"], dev)
        if saved["round"]:
            last_report = read_json(args.out / f"round-{saved['round']:04}.json")
            best = choose_best(best, saved["round"], saved["weights"], last_report["eval_rows"], dev)
    best_path = args.out / "best-checkpoint.json"
    write_best(best_path, protocol, saved["schedule_offset"], best)

    if args.test:
        settings = update_config("warmup", saved["schedule_offset"] + saved["round"], 12)
        result = dict(round=saved["round"], selected_round=best["round"], seeds=holdout[:args.test])
        for label, cohort in [("stage_start", saved["initial_weights"]), ("trained", best["weights"])]:
            policy_weights = checkpoint_weights(cohort)
            rows = [run(base, policy_weights, seed, settings)[0] for seed in result["seeds"]]
            result[label] = dict(summary=summarize(rows), rows=rows)
        write_json(destination, result)
        return

    if "best_dev" not in saved:
        saved["best_dev"] = best
        write_json(path, saved)
    for number in range(saved["round"] + 1, args.rounds + 1):
        settings = update_config("warmup", saved["schedule_offset"] + number, 12)
        rows, battle, strategic, pool, rejected = train_batch(
            base, weights, pool, attempts, settings, number, protected, sampling=sampling,
            offset=saved["schedule_offset"],
        )
        updated = update(weights["combat"], battle)
        step = float(np.linalg.norm(updated - weights["combat"]))
        weights["combat"] = updated
        weights["strategy"], strategic_step = update_weights(weights["strategy"], strategic, settings)
        eval_rows = [run(base, weights, seed, settings)[0] for seed in dev]
        saved_weights = {k: v.tolist() for k, v in weights.items()}
        best = choose_best(best, number, saved_weights, eval_rows, dev)
        fresh = [row for row in rows if row["phase"] == "train"]
        focus = [row for row in rows if row["phase"] == "focus"]
        report = dict(round=number, fresh=summarize(fresh), focus=summarize(focus), eval=summarize(eval_rows),
                      focus_cleared=sum(next((f["hp_out"] > 0 for f in row["fights"]), False) for row in focus),
                      pool_tags=dict(Counter(tag for case in pool for tag in case["tags"])),
                      focus_failures=rejected, train_rows=rows, eval_rows=eval_rows,
                      combat_step_l2=step, strategy_update=strategic_step, selected_round=best["round"])
        if sampling is not None:
            report["sampling"] = sampling
        write_json(args.out / f"round-{number:04}.json", report)
        saved.update(round=number, cases=pool, weights=saved_weights, best_dev=best)
        write_json(path, saved)
        write_best(best_path, protocol, saved["schedule_offset"], best)
        focus_display = str(len(focus)) if sampling is None else f"{len(focus)}/{focus_attempts}"
        print(f"Round {number} saved. Fresh {len(fresh)}, focus {focus_display}, pool {len(pool)}. "
              f"Selected round {best['round']}.", flush=True)


if __name__ == "__main__":
    main()
