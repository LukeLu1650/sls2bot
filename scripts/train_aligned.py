"""Compare the existing scoring updates and PPO in the same Steam-aligned environment."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import statistics

from beta_build import CHECKOUT, STEAM_HOST, STEAM_MANIFEST, sha, verify_host
from beta_env import is_combat_decision, run_episode
from beta_eval import write_json
from strategy_updates import training_rng, update_config, update_weights
from train_combat import update
from train_curriculum import Attempt, PRIORITIES, REGRESSION, checkpoint_weights, choose_case, read_json, retain_cases
from train_joint import JointPolicy, show_hp
from train_strategy import BASE, load_base, np, panel, positive_int, reward


TERMINAL = {"defeat", "victory"}
EXPERIMENT = "standard-a0-aligned-v1"
PRE_CONTINUATION_SOURCE = "6970eb32a6b15a638c717d69574c9cfd3f027d2a1ec164bfc4bfa43fb253b66f"
PRE_TERMINAL_LOG_SOURCE = "9f1316b0783893d622b623859b36844132b74a5565a03abb947f3e3a17d9b3d9"
COMPATIBLE_STEAM_HOSTS = {
    "a291afea7243a4839b10dd39d4297dace6d591aec0fe9b4b7b2a7a35cd3de98c",
    "3f94256154ba293f06af2c195e8958a3f1de1da7cb26fbc4a74be1f06829fd77",
    "5c0acfd4f8027335e180d1496ac0ab4f0cf21d4f0417f6169109e96bdeb92066",
    "330710141478b0dfee06321e585934e623ffd024895bd3a68ad110bedf19cfbb",
}
SETTINGS = update_config("warmup", 12, 12)


def start_command(seed, progress):
    return {"cmd": "start_steam_run", "progress": progress,
            "run": dict(seed=seed, character="Necrobinder", ascension=0, game_mode="standard",
                        is_custom_game=False, modifiers=[])}


class RunTrace:
    def __init__(self):
        self.fights = {}
        self.active = None
        self.confirmed = 0
        self.steps = []
        self.last_room = None
        self.rooms = []

    def observe(self, state):
        self.confirmed = len(self.steps)
        context = state.get("context", {})
        room = context.get("act", state.get("act")), context.get("floor", state.get("floor"))
        hp = state.get("player", {}).get("hp")
        if None in room or hp is None:
            return
        if room != self.last_room:
            self.rooms.append((room, self.confirmed))
            self.last_room = room
        if self.active and (room != self.active or not is_combat_decision(state, self.active)):
            fight = self.fights[self.active]
            fight["hp_out"] = hp if room[0] == self.active[0] else fight["last_hp"]
            if fight["hp_out"] > 0 and not fight["escaped"]:
                fight["kill_order"].extend(sorted(fight["alive"].elements()))
            fight["finished_at"] = self.confirmed
            self.active = None
        if state["decision"] != "combat_play":
            return
        enemies = Counter(e["name"] for e in state["enemies"] if e["hp"] > 0)
        self.active = room
        fight = self.fights.setdefault(room, dict(act=room[0], floor=room[1], room=context.get("room_type"),
                                                 hp_in=hp, max_hp=state["player"]["max_hp"], hp_out=None,
                                                 alive=enemies, last_hp=hp, kill_order=[], potion_uses=0,
                                                 escaped=False))
        fight["kill_order"].extend(sorted((fight["alive"] - enemies).elements()))
        fight.update(alive=enemies, last_hp=hp)

    def record(self):
        return [{k: v for k, v in fight.items() if k not in {"alive", "last_hp", "finished_at", "escaped"}}
                for fight in self.fights.values()]


def fight_return(fight):
    return .18 * (fight["hp_out"] > 0) + .65 * (fight["hp_out"] - fight["hp_in"]) / max(1, fight["max_hp"])


def play(base, weights, model, progress, seed, *, training=False, repeat=0, case=None, number=0):
    opening = (case["act"], case["floor"]) if case else None
    key = f"{seed}/focus/{case['opening']}/{number}" if case else seed
    rng = training_rng(key, repeat) if training else None
    trace = RunTrace()
    old_policy = JointPolicy(base, weights, rng, SETTINGS) if model is None else None
    reason = None
    combat_room = None
    act1_strategy = []

    def choose(state, actions):
        nonlocal reason, combat_room
        context = state["context"]
        room = context["act"], context["floor"]
        combat = is_combat_decision(state, combat_room)
        combat_room = room if combat else None
        if old_policy is not None:
            previous = len(old_policy.strategy.gradients)
            chosen = old_policy(state, actions)
            if room[0] == 1 and len(old_policy.strategy.gradients) > previous:
                act1_strategy.append(old_policy.strategy.gradients[-1])
        else:
            if combat:
                reason = None
            if len(actions) > 1:
                chosen, sample = model.decide(state, actions, combat, reason, rng)
                if sample is not None:
                    trace.steps.append(sample)
            else:
                chosen = 0
            action = actions[chosen]
            if action["action"] == "remove_card":
                reason = "remove_card"
            elif state["decision"] == "rest_site" and action["action"] == "choose_option":
                option = next(o for o in state["options"] if o["index"] == action["args"]["option_index"])
                reason = "smith" if any("smith" in str(option.get(k, "")).lower() or
                                         "upgrade" in str(option.get(k, "")).lower()
                                         for k in ("name", "option_id", "title")) else None
            elif state["decision"] != "card_select":
                reason = None
        if actions[chosen]["action"] == "use_potion" and room in trace.fights:
            trace.fights[room]["potion_uses"] += 1
        return chosen

    attempt = Attempt(choose, case)
    row = run_episode(STEAM_HOST, seed, "standard", 0, attempt, episode_timeout=110, max_steps=1500,
                      strategic_observations=True, sequential_combat_selection=True,
                      start_command=start_command(seed, progress), observe=trace.observe)
    terminal = row["status"] in TERMINAL
    row.update(fights=trace.record(), rollout=repeat + 1, act1_cleared=(row.get("act") or 0) > 1 or row["victory"] is True)
    row["last_floor_entry"] = attempt.policy.entries.get((row["act"], row["floor"]))
    target = trace.fights.get(opening)
    if case:
        row["focus"] = dict(seed=seed, opening=case["opening"], act=opening[0], floor=opening[1],
                            reached=attempt.reached_target)
    if target and target["hp_out"] is not None and attempt.reached_target:
        row["focus"].update(survived=target["hp_out"] > 0, net_hp_lost=target["hp_in"] - target["hp_out"],
                            potion_uses=target["potion_uses"], kill_order=target["kill_order"],
                            return_=fight_return(target))
    if not training or not attempt.reached_target:
        return row, [], [], []
    cases = list(attempt.cases(row, number)) if terminal and not case else []
    if old_policy is None:
        return row, ppo_returns(trace, row, opening), [], cases

    combat, strategy = [], []
    future = (2.0 if row["victory"] else -1.0) if terminal else 0.0
    next_act = row.get("act") or 1
    for fight in reversed(old_policy.combat.fights):
        room = fight.act, fight.floor
        confirmed = trace.fights.get(room)
        if not confirmed or confirmed["hp_out"] is None:
            continue
        local = fight_return(confirmed) + 1.5 * (next_act > fight.act)
        future = local + .94 * future
        if fight.choices and (opening is None or opening == room):
            combat.append((future, fight.gradient, fight.choices))
        next_act = fight.act
    gradients = old_policy.strategy.gradients if terminal else act1_strategy if row["act1_cleared"] else []
    if not case and gradients:
        score = reward(row) if terminal else .16
        strategy.append((score, np.mean(gradients, axis=0), old_policy.strategy.max_feature_pair_bound))
    return row, combat, strategy, cases


def ppo_returns(trace, row, opening):
    samples = trace.steps[:trace.confirmed]
    if not samples or (opening and (opening not in trace.fights or trace.fights[opening]["hp_out"] is None)):
        return []
    rewards = np.zeros(len(samples), dtype=np.float32)
    for fight in trace.fights.values():
        end = fight.get("finished_at", 0)
        if 0 < end <= len(samples):
            rewards[end - 1] += fight_return(fight)
    previous = None
    for room, end in trace.rooms:
        if previous and 0 < end <= len(samples):
            rewards[end - 1] += 1.5 if room[0] > previous[0] else .005
        previous = room
    terminal = row["status"] in TERMINAL
    if terminal:
        rewards[-1] += 2.0 if row["victory"] else -1.0
    # The first still-unconfirmed action was sampled at the last valid observation.
    next_value = 0.0 if terminal else trace.steps[len(samples)].value if len(trace.steps) > len(samples) else samples[-1].value
    advantage = 0.0
    decisions = []
    for decision, immediate in reversed(list(zip(samples, rewards, strict=True))):
        delta = float(immediate) + .995 * next_value - decision.value
        advantage = delta + .995 * .97 * advantage
        if opening is None or (decision.combat and decision.room == opening):
            decisions.append((decision, advantage, advantage + decision.value))
        next_value = decision.value
    return decisions


def focus_advantages(group):
    """Use the other attempts at this opening as an action-independent baseline."""
    completed = [(row["focus"]["return_"], samples) for row, samples in group if "return_" in row["focus"]]
    total = sum(score for score, _ in completed)
    for score, samples in completed:
        relative = score - (total - score) / (len(completed) - 1) if len(completed) > 1 else 0.0
        for decision, advantage, target in samples:
            yield decision, advantage + relative, target


def summary(rows):
    boss = [b["hp"] for row in rows for b in row["boss_entries"] if b["act"] == 1]
    losses = [f["hp_in"] - f["hp_out"] for row in rows for f in row["fights"]
              if f["act"] == 1 and f["room"] == "Monster" and f["hp_out"] and f["hp_out"] > 0]
    clears = sum(r["act1_cleared"] for r in rows)
    unresolved = sum(r["status"] not in TERMINAL and not r["act1_cleared"] for r in rows)
    return dict(attempts=len(rows), confirmed_act1_clears=clears,
                unresolved_before_act2=unresolved, act1_rate_lower=clears / len(rows),
                act1_rate_upper=(clears + unresolved) / len(rows),
                boss_entries=len(boss), median_boss_hp=statistics.median(boss) if boss else None,
                median_normal_fight_hp_loss=statistics.median(losses) if losses else None,
                wins=sum(r["victory"] is True for r in rows), statuses=dict(Counter(r["status"] for r in rows)))


def encode_weights(weights):
    return {k: v.tolist() for k, v in weights.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-ppo", type=Path, help="Continue from a PPO checkpoint and compare against that fixed starting policy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=positive_int, default=32)
    parser.add_argument("--fresh-seeds", type=positive_int, default=12)
    parser.add_argument("--focus-rollouts", type=positive_int, default=3)
    parser.add_argument("--eval-games", type=positive_int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=positive_int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    attempt_count = 2 * args.fresh_seeds + 3 * args.focus_rollouts
    import torch
    from ppo_policy import PPOPolicy

    torch.set_num_threads(1 if args.workers > 1 else 2)
    torch.set_num_interop_threads(1)
    verify_host(steam=True)
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    progress = read_json(args.progress)
    source = read_json(args.checkpoint)
    if source["protocol"]["base"] != sha(BASE / "readout.npz"):
        raise ValueError("Checkpoint base differs from the installed policy")
    initial = checkpoint_weights(source["weights"])
    baseline = initial.copy()
    base = load_base()
    model = PPOPolicy(base, initial, args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    dev = sorted(set(panel("development", 8, "necro-standard")) | set(REGRESSION))
    work = dict(host=sha(STEAM_MANIFEST), progress=sha(args.progress),
                start=sha(args.checkpoint), development_seeds=dev,
                samples_per_arm=dict(fresh_seeds=args.fresh_seeds, repeats=2, focus_cases=3,
                                     focus_repeats=args.focus_rollouts),
                sources={name: sha(Path(__file__).with_name(name)) for name in
                         ("train_aligned.py", "ppo_policy.py", "beta_env.py", "beta_build.py", "train_curriculum.py",
                          "train_combat.py", "train_joint.py", "train_strategy.py", "strategy_updates.py",
                          "strategic_features.py", "necro_policy.py", "train_standard.py")})
    progress_path = args.out / "protocol.json"
    checkpoint_path = args.out / "ppo.pt"
    last_round = 0
    pool = []
    parent = None
    namespace = EXPERIMENT
    previously_evaluated = set()
    if args.init_ppo:
        provenance = read_json(args.init_ppo.with_name("protocol.json"))
        host_ok = provenance["host"] == work["host"] or provenance["host"] in COMPATIBLE_STEAM_HOSTS
        if not host_ok or any(provenance[key] != work[key] for key in ("progress", "start")):
            raise ValueError("PPO warm start requires the same Steam host, progress and base checkpoint")
        parent = torch.load(args.init_ppo, map_location="cpu", weights_only=True)
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(parent["model"]) != trainable_names:
            raise ValueError("PPO warm start requires a compatible learned parameter set")
        model.load_state_dict(parent["model"], strict=False)
        optimizer.load_state_dict(parent["optimizer"])
        work["ppo_parent"] = sha(args.init_ppo)
        namespace = f"{EXPERIMENT}/continuation/{work['ppo_parent'][:16]}"
        previously_evaluated.update(provenance["development_seeds"])
        previously_evaluated.update(provenance["evaluation_seeds"])
    evaluation_pool = panel("final", args.eval_games + len(previously_evaluated), namespace)
    test = [seed for seed in evaluation_pool if seed not in previously_evaluated][:args.eval_games]
    protected = set(dev) | set(test) | previously_evaluated
    work["evaluation_seeds"] = test
    if parent is not None:
        pool = [case for case in parent["pool"] if case["seed"] not in protected]
    if args.resume:
        recorded = read_json(progress_path)
        if recorded["sources"]["train_aligned.py"] in {PRE_CONTINUATION_SOURCE, PRE_TERMINAL_LOG_SOURCE}:
            recorded["sources"]["train_aligned.py"] = work["sources"]["train_aligned.py"]
        if recorded != work:
            raise ValueError("Host, source, progress or evaluation panel changed since the experiment began")
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
            last_round, pool = saved["round"], saved["pool"]
            baseline = checkpoint_weights(saved["baseline"])
            model.load_state_dict(saved["model"], strict=False)
            optimizer.load_state_dict(saved["optimizer"])
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(progress_path, work)

    agents = {"baseline": None, "ppo": model}
    arms = ("ppo",) if parent is not None else ("baseline", "ppo")

    def collect(jobs, training=False):
        def one(job):
            arm, seed, repeat, case = job
            outcome = play(base, baseline, agents[arm], progress, seed, training=training,
                           repeat=repeat, case=case, number=number)
            row = outcome[0]
            entry = row["last_floor_entry"] or {}
            phase = "focus" if case else "train" if training else "eval"
            label = f"{seed}/{repeat + 1}" if training else seed
            hp = show_hp(entry.get("before_travel"))
            print(f"{phase} {arm} {label} {row['status']} act={row['act']} floor={row['floor']} "
                  f"hp={hp}", flush=True)
            return outcome

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            return list(executor.map(one, jobs))

    number = last_round
    for number in range(last_round + 1, args.rounds + 1):
        fresh = [s for s in panel(f"train/{number}", args.fresh_seeds + len(protected), namespace)
                 if s not in protected][:args.fresh_seeds]
        logs = {arm: [] for arm in arms}
        battle, strategy, samples = [], [], []
        jobs = [(arm, seed, repeat, None) for seed in fresh for repeat in range(2) for arm in arms]
        for (arm, _, _, _), (row, batch, strategic, cases) in zip(jobs, collect(jobs, training=True), strict=True):
            logs[arm].append(row)
            if arm == "baseline":
                battle.extend(batch)
                strategy.extend(strategic)
                pool.extend(cases)
            else:
                samples.extend(batch)
                if parent is not None:
                    pool.extend(cases)
        used = set()
        focus_cases = []
        for tag in PRIORITIES[:3]:
            case = choose_case(pool, tag, used)
            if case is None:
                raise RuntimeError("Fewer than three distinct Steam-aligned hard openings are available")
            used.add(case["seed"])
            focus_cases.append(case)
        jobs = [(arm, case["seed"], repeat, case) for case in focus_cases
                for repeat in range(args.focus_rollouts) for arm in arms]
        results = iter(collect(jobs, training=True))
        for case in focus_cases:
            group = []
            stale = False
            for _ in range(args.focus_rollouts):
                for arm in arms:
                    row, batch, _, _ = next(results)
                    logs[arm].append(row)
                    stale |= not row["focus"]["reached"]
                    if arm == "baseline":
                        battle.extend(batch)
                    else:
                        group.append((row, batch))
            samples.extend(focus_advantages(group))
            case["uses"] += args.focus_rollouts
            if stale:
                pool.remove(case)
        pool = retain_cases(pool)
        strategic_step = None
        if parent is None:
            baseline["combat"] = update(baseline["combat"], battle)
            baseline["strategy"], strategic_step = update_weights(baseline["strategy"], strategy, SETTINGS)
        minibatch_key = f"minibatches/{number}" if parent is None else f"{namespace}/minibatches/{number}"
        change = model.update(samples, optimizer, training_rng(minibatch_key, 0))
        development = {arm: summary([row for row, *_ in collect([(arm, s, 0, None) for s in dev])]) for arm in arms}
        write_json(args.out / f"round-{number:04}.json", dict(training=logs, development=development,
                                                             ppo_update=change, baseline_update=strategic_step))
        save = args.out / "ppo.tmp"
        learned = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
        torch.save(dict(round=number, model=learned, optimizer=optimizer.state_dict(),
                        baseline=encode_weights(baseline), pool=pool), save)
        save.replace(checkpoint_path)
        print(f"Round {number}.", flush=True)

    comparison = {}
    if parent is not None:
        agents["start"] = PPOPolicy(base, initial, args.device)
        agents["start"].load_state_dict(parent["model"], strict=False)
    for arm in ("start", "ppo") if parent is not None else arms:
        rows = [row for row, *_ in collect([(arm, s, 0, None) for s in test])]
        comparison[arm] = dict(summary=summary(rows), rows=rows)
        print(arm, comparison[arm]["summary"], flush=True)
    write_json(args.out / "comparison.json", comparison)


if __name__ == "__main__":
    main()
