from pathlib import Path
import sys
from time import monotonic
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from beta_env import _choose_card_selection, action_cards, public_observation
from necro_policy import NecroPolicy
from train_combat import CombatPolicy, Fight, SIZE, combat_inputs, fight_samples, np, update
from train_joint import FloorEntryLog
from train_standard import brief


class Base:
    width = 256
    readout = np.zeros(width, np.float32)
    readout_bias = 0.0

    def _candidate_embeddings(self, state, actions):
        return np.zeros((len(actions), self.width), np.float32)

    def __call__(self, state, actions):
        return 0


def battle():
    state = dict(decision="combat_play", context=dict(act=1, floor=2, room_type="Monster"), energy=1,
                 player=dict(hp=40, max_hp=66, block=0, potions=[dict(index=0, name="Block Potion")]),
                 enemies=[dict(index=0, hp=9, intents=[dict(type="Attack", damage=8)])],
                 hand=[dict(index=0, stats=dict(block=5), cost=1, can_play=True),
                       dict(index=1, stats=dict(damage=6), cost=1, can_play=True)])
    actions = [dict(action="play_card", args=dict(card_index=0)),
               dict(action="play_card", args=dict(card_index=1, target_index=0)),
               dict(action="use_potion", args=dict(potion_index=0)), dict(action="end_turn")]
    return state, actions


def prompt(zone="discard", trigger="DREDGE", action="play_card"):
    state, _ = battle()
    cards = [dict(index=7, id="CARD.STRIKE", name="Strike", cost=1, stats=dict(damage=6),
                  description="Deal damage.", zone=zone),
             dict(index=0, id="CARD.DEFEND", name="Defend", cost=1, stats=dict(block=5),
                  description="Gain Block.", zone=zone)]
    state.update(decision="card_select", in_combat=True, cards=cards, min_select=1, max_select=2,
                 selection_trigger=dict(action=action, id=trigger), picked_indices=[])
    actions = [dict(action="pick_card", args=dict(card_index=card["index"])) for card in cards]
    return state, actions


class CombatTrainingTests(unittest.TestCase):
    def test_last_floor_entry_keeps_pre_travel_and_first_decision_hp(self):
        logger = FloorEntryLog(Base())
        state, actions = battle()
        logger({**state, "decision": "map_select"}, [dict(action="select_map_node")])
        arrival = {**state, "context": {**state["context"], "floor": 3},
                   "player": {**state["player"], "hp": 35}}
        logger(arrival, actions)
        logger({**arrival, "player": {"hp": 9, "max_hp": 66}}, actions)
        self.assertEqual(logger.entries[1, 3]["hp"], 35)
        self.assertEqual(logger.entries[1, 3]["before_travel"], {"hp": 40, "max_hp": 66})

    def test_all_four_candidates_receive_different_gradients(self):
        state, actions = battle()
        _, matrix = combat_inputs(Base(), state, actions)
        for i in range(4):
            gradient = matrix[i] - matrix.mean(axis=0)
            self.assertGreater(np.linalg.norm(gradient), 0)
            for j in range(i):
                self.assertFalse(np.array_equal(matrix[i], matrix[j]))
        policy = CombatPolicy(Base(), np.zeros(SIZE), np.random.default_rng(2))
        policy(state, actions)
        self.assertGreater(np.linalg.norm(policy.active.gradient), 0)

    def test_selection_keeps_encounter_open_and_shop_heal_excluded(self):
        state, actions = battle()
        policy = CombatPolicy(Base(), np.zeros(SIZE))
        policy(state, actions)
        policy({**state, "decision": "card_select"}, [{}])
        policy({**state, "decision": "card_reward", "from_event": True}, [{}])
        self.assertIsNotNone(policy.active)
        policy({**state, "decision": "card_reward", "player": {"hp": 33}}, [{}])
        policy({**state, "decision": "map_select", "player": {"hp": 50}}, [{}])
        self.assertEqual(policy.fights[0].hp_out, 33)

    def test_future_fight_changes_earlier_potion_credit(self):
        fight = Fight(1, 2, "Monster", 40, 66, 35, choices=2, potion_uses=1)
        elite = Fight(1, 6, "Elite", 35, 66, 22, choices=1)
        result = dict(status="defeat", victory=False, act=1)
        credit = fight_samples([fight, elite], result)[-1][0]
        fight.potion_uses = 0
        self.assertEqual(fight_samples([fight, elite], result)[-1][0], credit)
        elite.hp_out = 30
        self.assertGreater(fight_samples([fight, elite], result)[-1][0], credit)
        self.assertEqual(fight_samples([fight], dict(status="timeout")), [])

    def test_observed_combat_choices_can_change_greedy_decision(self):
        state, actions = battle()
        _, matrix = combat_inputs(Base(), state, actions)
        zero = np.zeros(SIZE, np.float32)
        gradient = matrix[2] - matrix.mean(axis=0)
        weights = update(zero, [(1., gradient, 1), (-1., -gradient, 1)])
        self.assertEqual(CombatPolicy(Base(), weights)(state, actions), 2)
        self.assertEqual(CombatPolicy(Base(), zero)(state, actions), 0)

    def test_selection_features_follow_card_contents_when_indices_are_swapped(self):
        state, actions = prompt()
        swapped = {**state, "cards": [{**card, "index": index}
                                     for card, index in zip(reversed(state["cards"]), (7, 0))]}
        encoded = NecroPolicy.tokens(state, actions)[-2:]
        self.assertEqual(encoded[::-1], NecroPolicy.tokens(swapped, actions)[-2:])
        self.assertIn(("selection/action/card/effect/block", 1.0), encoded[1])
        _, matrix = combat_inputs(Base(), state, actions)
        _, exchanged = combat_inputs(Base(), swapped, actions)
        np.testing.assert_array_equal(matrix[::-1], exchanged)
        draw, _ = prompt("draw", "SECRET_TECHNIQUE")
        self.assertFalse(np.array_equal(matrix, combat_inputs(Base(), draw, actions)[1]))

    def test_choices_contribute_to_fight_before_first_playable_turn(self):
        for zone, trigger, kind in [("discard", "DREDGE", "play_card"),
                                    ("draw", "SECRET_TECHNIQUE", "play_card"),
                                    ("hand", "BURNING_PACT", "play_card"),
                                    ("hand", "GAMBLERS_BREW", "use_potion")]:
            with self.subTest(trigger=trigger):
                state, actions = prompt(zone, trigger, kind)
                policy = CombatPolicy(Base(), np.zeros(SIZE), np.random.default_rng(2))
                policy(state, actions)
                fight = policy.active
                self.assertEqual(fight.selection_choices, 1)
                self.assertGreater(np.linalg.norm(fight.gradient), 0)
                policy({**state, "in_combat": False, "player": {"hp": 31}}, actions)
                self.assertEqual(fight.hp_out, 31)
                [(credit, gradient, choices)] = fight_samples(policy.fights, dict(status="defeat", victory=False, act=1))
                self.assertEqual(choices, 1)
                np.testing.assert_array_equal(gradient, fight.gradient)
                self.assertAlmostEqual(credit, .18 + .65 * (31 - 40) / 66 - .94)

    def test_sequential_selection_respects_limits_and_records_card_names(self):
        state, _ = prompt()
        state.update(min_select=2, draw_pile=[dict(id="hidden")], type="decision")
        public = public_observation(state)
        self.assertNotIn("draw_pile", public)
        frames = []

        def choose(observation, actions):
            self.assertEqual(len(observation["picked_indices"]), len(frames))
            chosen = len(actions) - 1
            frames.append(brief(observation, actions, np.arange(len(actions)), chosen))
            return chosen

        command = _choose_card_selection(choose, public, monotonic() + 5)
        self.assertEqual(command["args"]["indices"], "0,7")
        self.assertEqual([card["name"] for card in action_cards(public, command)], ["Defend", "Strike"])
        self.assertEqual([frame["selected"] for frame in frames],
                         ["Select · Defend", "Select · Strike", "Confirm · Defend, Strike"])
        state["min_select"] = 0
        skipped = _choose_card_selection(lambda state, actions: len(actions) - 1, state, monotonic() + 5)
        self.assertEqual(skipped["action"], "skip_select")
