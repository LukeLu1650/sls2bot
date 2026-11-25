from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_curriculum import (Attempt, REGRESSION, choose_best, choose_case, development_rank,
                              potion_changes_kill_line, signature, train_batch)


class CurriculumTests(unittest.TestCase):
    def test_auto_selection_prioritizes_act_clear_and_ignores_regression(self):
        def row(seed, *, act=1, floor=6, boss=False, status="defeat"):
            return dict(seed=seed, status=status, act=act, floor=floor, victory=False,
                        boss_entries=[{"act": 1}] if boss else [])

        seeds = {"A", "B", *REGRESSION}
        start = [row("A", floor=17, boss=True), row("B", floor=16)]
        best = choose_best(None, 0, "initial", start, seeds)
        self.assertEqual(best["rank"], [0, 1, 33])
        high_floor = [row("A", floor=16), row("B", floor=17), row(REGRESSION[0], act=2)]
        self.assertIs(choose_best(best, 1, "high_floor", high_floor, seeds), best)
        reached_act_two = [row("A", act=2, boss=True), row("B", floor=4)]
        selected = choose_best(best, 2, "cleared", reached_act_two, seeds)
        self.assertEqual(selected["round"], 2)
        timed_out = [row("A", act=2, boss=True), row("B", status="timeout")]
        self.assertIsNone(development_rank(timed_out, seeds))
        self.assertIs(choose_best(selected, 3, "incomplete", timed_out, seeds), selected)

    def test_prefix_cannot_reach_policy_until_verified_opening(self):
        travel = {"action": "select_map_node", "args": {"col": 0, "row": 1}}
        map_state = dict(decision="map_select", context=dict(act=1, floor=1),
                         player=dict(hp=58, max_hp=66))
        combat = dict(decision="combat_play", context=dict(act=1, floor=2, room_type="Elite"),
                      player=dict(hp=57, max_hp=66), round=1,
                      enemies=[dict(index=0, name="Test enemy", intents=[])])
        calls = []
        attempt = Attempt(lambda state, actions: calls.append(state) or 0,
                          dict(prefix=[travel], opening=signature(combat)))
        self.assertEqual(attempt(map_state, [travel]), 0)
        self.assertFalse(calls)
        with self.assertRaisesRegex(ValueError, "does not match"):
            attempt({**combat, "round": 2}, [dict(action="end_turn")])
        self.assertFalse(calls)
        self.assertEqual(attempt(combat, [dict(action="end_turn")]), 0)
        self.assertEqual(calls, [combat])
        self.assertEqual(attempt.policy.entries[1, 2]["before_travel"]["hp"], 58)

    def test_preferred_tag_and_distinct_seeds(self):
        def case(seed, tag):
            return dict(seed=seed, tags=[tag], uses=0, discovered=1)

        potion = case("A", "unused_potion")
        boss = case("B", "boss")
        high_hp = case("A", "high_hp_loss")
        pool = [potion, boss, high_hp]
        self.assertEqual(choose_case(pool, "unused_potion", set()), potion)
        self.assertEqual(choose_case(pool, "high_hp_loss", {"A"}), boss)

    def test_repeated_batch_shares_seeds_and_focus_opening_but_varies_policy_sampling(self):
        focus_case = dict(seed="HARD", opening="first", tags=["unused_potion"], uses=0, discovered=1)
        other_case = dict(seed="OTHER", opening="second", tags=["high_hp_loss"], uses=0, discovered=1)
        seeds = ["PROTECTED", *(f"NEW-{i:02}" for i in range(12))]
        sampling = dict(fresh_seeds=12, rollouts_per_seed=2, focus_rollouts=3)

        def play(base, weights, seed, settings, *, sample_key, case=None, number=0, rollout=None):
            row = dict(seed=seed, status="defeat", rollout=rollout)
            if case is not None:
                row["focus"] = {"reached": rollout != 2}
                return row, [f"focus-{rollout}"], [], []
            return row, [seed], [seed], []

        with patch("train_curriculum.panel", return_value=seeds) as panel, \
             patch("train_curriculum.run", side_effect=play) as run:
            rows, combat, strategy, pool, rejected = train_batch(
                None, None, [focus_case, other_case], 27, None, 1, {"PROTECTED"}, sampling=sampling, offset=292)

        self.assertEqual(panel.call_args.args[0], "train/repeated/293")
        fresh, difficult = run.call_args_list[:24], run.call_args_list[24:]
        self.assertEqual([(call.args[2], call.kwargs["rollout"]) for call in fresh],
                         [(seed, rollout) for seed in seeds[1:] for rollout in (1, 2)])
        self.assertEqual([call.kwargs["rollout"] for call in difficult], [1, 2, 3])
        self.assertTrue(all(call.kwargs["case"] is focus_case for call in difficult))
        self.assertEqual(len({call.kwargs["sample_key"] for call in difficult}), 1)
        self.assertEqual(len(strategy), 24)
        self.assertEqual(len(rows), 26)
        self.assertEqual(combat[-2:], ["focus-1", "focus-3"])
        self.assertEqual([row["rollout"] for row in rejected], [2])
        self.assertEqual(pool, [other_case])

    def test_half_health_losses_require_an_unused_potion_for_priority(self):
        attempt = Attempt(None)
        diagnostics = dict(act=1, floor=6, room="Monster", hp=33, max_hp=66, unused_potions=["Strength Potion"],
                           missed_lethal_attacker=0, scaling_enemy=False, potion_kill_line=False, prefix_length=0)
        attempt.fights[1, 6] = diagnostics
        result = dict(seed="TRAIN", status="defeat", act=1, floor=6, fights=[dict(act=1, floor=6, hp_out=0)])
        self.assertEqual([case["tags"] for case in attempt.cases(result, 1)], [["unused_potion"]])
        diagnostics["unused_potions"] = []
        self.assertEqual(list(attempt.cases(result, 1)), [])

    def test_strength_potion_can_make_a_basic_attack_lethal(self):
        enemy = {0: dict(index=0, hp=8, block=0)}
        potion = {0: dict(vars=dict(StrengthPower=2))}
        state = dict(enemies=list(enemy.values()), hand=[dict(index=0, type="Attack", stats=dict(damage=6))])
        actions = [dict(action="play_card", args=dict(card_index=0, target_index=0))]
        self.assertTrue(potion_changes_kill_line(state, actions, enemy, potion, {0}))
        self.assertFalse(potion_changes_kill_line(state, actions, enemy, potion, set()))
