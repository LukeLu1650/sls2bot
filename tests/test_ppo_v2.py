from dataclasses import replace
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
from threading import Lock
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from beta_build import patch_shop_removal
from beta_env import legal_actions
from ppo_v2_policy import (
    InferenceBatcher, PRESETS, PPOv2Policy, StepRecord,
    encode_tokens, pack_tokens, symexp, symlog, two_hot_encode,
)
from solver_bc import match_solver_step
from train_ppo_v2 import (
    RunTraceV2, aggregate_engine_errors, compute_ppo_returns, cosine_lr, dev_rank,
    error_signature, fight_return, focus_advantages, save_error_diagnostic, schedule_value,
    validate_resume_protocol,
)


def sample_combat_state(**overrides):
    state = {
        "decision": "combat_play",
        "round": 2,
        "energy": 2,
        "max_energy": 3,
        "draw_pile_count": 7,
        "discard_pile_count": 3,
        "context": {"act": 1, "floor": 6, "room_type": "Elite", "ascension": 0, "boss": {"id": "WATERFALL_GIANT"}},
        "player": {
            "hp": 48, "max_hp": 66, "block": 5, "gold": 120, "deck_size": 12,
            "potion_capacity": 3, "has_open_potion_slots": True,
            "relics": [{"id": "RELIC.BOUND_PHYLACTERY", "name": "Bound Phylactery", "counter": 1}],
            "potions": [{"index": 0, "id": "POTION.FIRE_POTION", "name": "Fire Potion", "vars": {"Damage": 20}}],
            "deck": [
                {"id": "CARD.STRIKE_NECROBINDER", "cost": 1, "stats": {"damage": 6}},
                {"id": "CARD.DEFEND_NECROBINDER", "cost": 1, "stats": {"block": 5}, "upgraded": True},
            ],
        },
        "osty": {"alive": True, "hp": 8, "max_hp": 8, "block": 0},
        "enemies": [
            {"index": 0, "name": "Terror Eel", "hp": 45, "max_hp": 60, "block": 0,
             "intends_attack": True, "intents": [{"type": "Attack", "damage": 14, "hits": 1}],
             "powers": [{"name": "Strength", "amount": 2}]},
        ],
        "hand": [
            {"index": 0, "id": "CARD.STRIKE_NECROBINDER", "name": "Strike", "type": "Attack",
             "target_type": "AnyEnemy", "cost": 1, "can_play": True, "upgraded": False,
             "keywords": ["Retain"], "enchantment": "Sharp", "enchantment_amount": 3,
             "affliction": "Bound", "stats": {"damage": 9}},
            {"index": 1, "id": "CARD.DEFEND_NECROBINDER", "name": "Defend", "type": "Skill",
             "target_type": "Self", "cost": 1, "can_play": True, "stats": {"block": 5}},
        ],
        "player_powers": [{"name": "Feral", "amount": 1}],
        "visible_map": {
            "current_coord": {"col": 2, "row": 5},
            "rows": [
                [{"col": 2, "row": 5, "type": "Elite", "children": [{"col": 2, "row": 6}]}],
                [{"col": 2, "row": 6, "type": "RestSite", "children": []}],
            ],
        },
    }
    state.update(overrides)
    return state


def sample_actions():
    return [
        {"action": "play_card", "args": {"card_index": 0, "target_index": 0}},
        {"action": "play_card", "args": {"card_index": 1}},
        {"action": "use_potion", "args": {"potion_index": 0, "target_index": 0}},
        {"action": "end_turn"},
    ]


class PPOv2Tests(unittest.TestCase):
    def test_all_presets_are_compact_and_100_percent_trainable(self):
        for name, cfg in PRESETS.items():
            model = PPOv2Policy(cfg, "cpu")
            summary = model.parameter_summary()
            self.assertEqual(summary["frozen"], 0, f"{name} must have zero frozen params")
            self.assertEqual(summary["trainable"], summary["total"])
            # Must be much lighter than the legacy 18.1M 5024-FFN architecture
            self.assertLess(summary["total"], 10_000_000)

    def test_tokens_include_relics_potions_deck_map_and_hand_metadata_without_leaks(self):
        state = sample_combat_state()
        actions = sample_actions()
        tokens = encode_tokens(state, actions)
        labels = {label for tok in tokens for label, _ in tok}
        self.assertIn("token/deck_summary", labels)
        self.assertIn("token/relic", labels)
        self.assertIn("relic/name/Bound Phylactery", labels)
        self.assertIn("token/potion", labels)
        self.assertIn("potion/name/Fire Potion", labels)
        self.assertIn("hand/keyword/Retain", labels)
        self.assertIn("hand/enchantment/Sharp", labels)
        self.assertIn("hand/affliction/Bound", labels)
        self.assertIn("map/ahead/1/RestSite", labels)
        self.assertIn("context/boss/WATERFALL_GIANT", labels)
        # Verify hidden fields (seed / draw_pile order) do not affect tokenization
        noisy = {**state, "seed": "SECRET_SEED", "draw_pile": [{"id": "HIDDEN_CARD"}]}
        self.assertEqual(tokens, encode_tokens(noisy, actions))

    def test_value_head_sees_relic_and_deck_changes_and_gradients_reach_all_layers(self):
        model = PPOv2Policy(PRESETS["compact"], "cpu")
        s1 = sample_combat_state()
        s2 = sample_combat_state()
        s2["player"] = {**s1["player"], "relics": [{"id": "RELIC.SNECKO_EYE", "name": "Snecko Eye"}]}
        acts = sample_actions()
        e1 = pack_tokens(encode_tokens(s1, acts), len(acts), model.cfg.vocab_size)
        e2 = pack_tokens(encode_tokens(s2, acts), len(acts), model.cfg.vocab_size)
        out = model.forward_batch([e1, e2])
        self.assertEqual(out["logits"].shape, (2, len(acts)))
        # Non-zero value head weights after one PPO update should react to relic/deck changes
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(7)
        samples = []
        # Mix variable action counts (4 vs 2 actions) to verify padded act_mask never causes NaN grads
        short_acts = acts[:2]
        for idx, (st, a_list, combat) in enumerate([
            (s1, acts, True), (s2, short_acts, True), (s1, acts, False), (s2, short_acts, False),
        ]):
            choice, rec = model.decide(st, a_list, combat, None, rng)
            samples.append({
                "step": rec, "advantage": 1.0 if idx % 2 == 0 else -1.0,
                "target": 0.5 if idx % 2 == 0 else -0.5,
                "final_floor": 17, "combat_end_hp": 0.7 if combat else None,
            })
        bc = [(e1, 0), (e2, 1)]
        stats = model.update(samples, opt, rng, epochs=1, minibatch=4, bc_samples=bc, bc_coef=0.2)
        self.assertGreater(stats["updates"], 0)
        self.assertIsNotNone(stats["bc_loss"])
        # Every single parameter (including embedding and block 0) must receive gradients
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Nonfinite gradient for {name}")
        out_after = model.forward_batch([e1, e2])
        self.assertNotAlmostEqual(float(out_after["v_return"][0].detach()), float(out_after["v_return"][1].detach()), places=5)

    def test_room_boundary_discounting_preserves_long_horizon_credit_and_boss_shaping(self):
        # Simulate 16 rooms x 15 decisions = 240 steps ending in Act 1 Boss win
        trace = RunTraceV2()
        dummy_enc = pack_tokens([[(f"k/{i}", 1.0)] for i in range(3)], 2, 4096)
        step_idx = 0
        for fl in range(1, 18):
            trace.rooms.append(((1, fl), step_idx))
            for k in range(14):
                trace.steps.append(StepRecord(dummy_enc, combat=(k > 0), choice=0, log_prob=-0.69, value=0.0, room=(1, fl), floor_total=fl))
                step_idx += 1
        trace.rooms.append(((2, 1), step_idx))
        trace.confirmed = len(trace.steps)
        row = {"status": "victory", "victory": True, "act": 2, "floor": 1}
        items = compute_ppo_returns(trace, row)
        # First strategic decision on floor 1 must retain substantial credit from Boss/Act clear
        # (vs ~0.01 under legacy per-step gamma=0.995, lam=0.97 across 238 steps)
        self.assertGreater(items[0]["target"], 1.5)

        # Smooth Boss damage shaping: 90% boss damage dealt >> 10% boss damage dealt on defeat
        close_boss = {"room": "Boss", "hp_in": 50, "hp_out": 0, "max_hp": 66, "enemy_initial_hp": 200, "enemy_min_hp": 20}
        far_boss = {"room": "Boss", "hp_in": 50, "hp_out": 0, "max_hp": 66, "enemy_initial_hp": 200, "enemy_min_hp": 180}
        self.assertGreater(fight_return(close_boss) - fight_return(far_boss), 0.30)

        # Elite kill bonus exceeds normal Monster fight reward with identical HP loss (and scales in Act 2/3)
        elite_win = {"room": "Elite", "hp_in": 60, "hp_out": 50, "max_hp": 66}
        normal_win = {"room": "Monster", "hp_in": 60, "hp_out": 50, "max_hp": 66}
        self.assertAlmostEqual(fight_return(elite_win) - fight_return(normal_win), 0.35, places=5)
        elite_a2 = {"act": 2, "room": "Elite", "hp_in": 60, "hp_out": 50, "max_hp": 66}
        self.assertAlmostEqual(fight_return(elite_a2) - fight_return(normal_win), 0.25, places=5)
        boss_a2 = {"act": 2, "room": "Boss", "hp_in": 60, "hp_out": 0, "max_hp": 66, "enemy_initial_hp": 300, "enemy_min_hp": 150}
        boss_a1 = {"act": 1, "room": "Boss", "hp_in": 60, "hp_out": 0, "max_hp": 66, "enemy_initial_hp": 300, "enemy_min_hp": 150}
        self.assertGreater(fight_return(boss_a2), fight_return(boss_a1))

    def test_schedules_focus_advantages_dev_rank_and_solver_matcher(self):
        self.assertAlmostEqual(schedule_value(1.0, 0.35, 1, 10), 1.0)
        self.assertAlmostEqual(schedule_value(1.0, 0.35, 10, 10), 0.35)
        self.assertLess(cosine_lr(3e-4, 64, 64), cosine_lr(3e-4, 4, 64))

        dummy = {"step": None, "advantage": 0.0, "target": 0.0}
        group = [
            ({"focus": {"return_": 0.2}}, [dict(dummy)]),
            ({"focus": {"return_": -0.4}}, [dict(dummy)]),
            ({"focus": {"return_": -0.4}}, [dict(dummy)]),
        ]
        merged = focus_advantages(group)
        self.assertGreater(merged[0]["advantage"], 0.0)
        self.assertLess(merged[1]["advantage"], 0.0)

        r1 = dev_rank({"attempts": 24, "wins": 0, "act2_clears": 0, "act1_clears": 3, "boss_entries": 10, "mean_progress": 12.0, "statuses": {"defeat": 24}})
        r2 = dev_rank({"attempts": 24, "wins": 0, "act2_clears": 0, "act1_clears": 4, "boss_entries": 9, "mean_progress": 11.0, "statuses": {"defeat": 24}})
        self.assertGreater(r2, r1)
        # Broadly stronger Act 1/progress checkpoint must beat a single noisy Act 2 outlier
        r_strong = dev_rank({"attempts": 24, "wins": 0, "act2_clears": 0, "act1_clears": 11, "boss_entries": 18, "mean_progress": 19.88, "statuses": {"defeat": 24}})
        r_noisy = dev_rank({"attempts": 24, "wins": 0, "act2_clears": 1, "act1_clears": 5, "boss_entries": 15, "mean_progress": 16.79, "statuses": {"defeat": 24}})
        self.assertGreater(r_strong, r_noisy)
        self.assertIsNone(dev_rank({"attempts": 24, "wins": 1, "act2_clears": 1, "act1_clears": 1, "boss_entries": 1, "mean_progress": 10.0, "statuses": {"engine_error": 1}}))

        state = sample_combat_state()
        acts = sample_actions()
        self.assertEqual(match_solver_step({"kind": "PlayCard", "cardId": "STRIKE_NECROBINDER", "cardOccurrence": 0, "targetIndex": 0}, state, acts), 0)
        self.assertEqual(match_solver_step({"kind": "UsePotion", "potionSlot": 0, "targetIndex": 0}, state, acts), 2)
        self.assertEqual(match_solver_step({"kind": "EndTurn"}, state, acts), 3)

    def test_diagnostic_bundle_deduplication_and_error_aggregation(self):
        err_row_a = {
            "seed": "SEED_A", "status": "engine_error", "act": 1, "floor": 8, "steps": 42,
            "error_code": "evt_nre", "error": "NullReferenceException",
            "stack_trace": "   at Event.Foo()\n   at Sim.Bar()\n   at Other.Baz()",
            "stderr_tail": "[SIM] Choosing option 0",
        }
        err_row_b = {**err_row_a, "seed": "SEED_B", "floor": 14}
        err_row_c = {**err_row_a, "seed": "SEED_C", "error": "DifferentError", "stack_trace": "   at Card.Play()"}
        ok_row = {"seed": "SEED_OK", "status": "defeat"}

        self.assertEqual(error_signature(err_row_a), error_signature(err_row_b))
        self.assertNotEqual(error_signature(err_row_a), error_signature(err_row_c))

        groups = aggregate_engine_errors([err_row_a, err_row_b, err_row_c, ok_row])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["count"], 2)
        self.assertEqual(groups[1]["count"], 1)

        with TemporaryDirectory() as tmp:
            diag_dir = Path(tmp) / "diagnostics"
            lock = Lock()
            p1 = save_error_diagnostic(
                diag_dir, lock, err_row_a,
                actions_prefix=[{"action": "choose_option", "args": {"option_index": 0}}],
                last_state=sample_combat_state(), last_actions=sample_actions(),
                progress_path=Path(tmp) / "progress.save", ascension=0, round_idx=3, phase="train",
            )
            p2 = save_error_diagnostic(
                diag_dir, lock, err_row_b,
                actions_prefix=[{"action": "end_turn"}],
                last_state=sample_combat_state(), last_actions=sample_actions(),
                progress_path=Path(tmp) / "progress.save", ascension=0, round_idx=3, phase="train",
            )
            # Same signature deduplicates to the exact same file without overwriting
            self.assertEqual(p1, p2)
            saved = json.loads(p1.read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], "SEED_A")
            self.assertEqual(len(saved["actions_prefix"]), 1)
            self.assertIn("last_state", saved)
            self.assertIn("last_actions", saved)
            self.assertIsNone(save_error_diagnostic(
                diag_dir, lock, ok_row, actions_prefix=[], last_state=None, last_actions=None,
                progress_path=Path(tmp) / "progress.save", ascension=0, round_idx=3, phase="train",
            ))
            # Non-engine abnormal episodes (step_limit / timeout) also save deduplicated bundles
            sl_pot = {
                "seed": "SEED_SL1", "status": "step_limit", "act": 2, "floor": 6, "steps": 1500,
                "error_code": "max_steps", "error": "Action step limit reached",
                "actions": {"buy_potion": 1200, "play_card": 100}, "decisions": {"shop": 1201},
            }
            sl_rem = {**sl_pot, "seed": "SEED_SL2", "actions": {"remove_card": 1200, "play_card": 100}}
            to_row = {
                "seed": "SEED_TO", "status": "timeout", "act": 1, "floor": 10, "steps": 90,
                "error_code": "host_timeout", "error": "Host did not respond",
                "actions": {"play_card": 60}, "decisions": {"combat_play": 60},
            }
            self.assertNotEqual(error_signature(sl_pot), error_signature(sl_rem))
            p_sl = save_error_diagnostic(
                diag_dir, lock, sl_pot, actions_prefix=[{"action": "buy_potion"}],
                last_state=sample_combat_state(), last_actions=sample_actions(),
                progress_path=Path(tmp) / "progress.save", ascension=0, round_idx=4, phase="eval",
            )
            self.assertTrue(p_sl.name.startswith("step_limit-"))
            sl_saved = json.loads(p_sl.read_text(encoding="utf-8"))
            self.assertEqual(sl_saved["status"], "step_limit")
            self.assertEqual(sl_saved["actions"]["buy_potion"], 1200)
            abn_groups = aggregate_engine_errors([sl_pot, sl_rem, to_row, ok_row])
            self.assertEqual(len(abn_groups), 3)
            self.assertEqual({g["status"] for g in abn_groups}, {"step_limit", "timeout"})

    def test_shop_card_removal_stock_guard_and_host_patch(self):
        base_shop = {
            "decision": "shop",
            "is_blitz_shop": False,
            "cards": [], "relics": [], "potions": [],
            "card_removal_cost": 100,
            "player": {"gold": 150, "potions": [], "potion_capacity": 3, "has_open_potion_slots": True},
        }
        acts_available = [a["action"] for a in legal_actions({**base_shop, "card_removal_stocked": True})]
        self.assertIn("remove_card", acts_available)
        acts_used = [a["action"] for a in legal_actions({**base_shop, "card_removal_stocked": False})]
        self.assertNotIn("remove_card", acts_used)
        acts_none = [a["action"] for a in legal_actions({**base_shop, "card_removal_cost": None})]
        self.assertNotIn("remove_card", acts_none)

        sample_cs = (
            b'        if (removal == null) return Error("No card removal available");\r\n'
            b'            ["card_removal_cost"] = removal?.Cost,\r\n'
        )
        patched = patch_shop_removal(sample_cs, b"\r\n")
        self.assertIn(b"!removal.IsStocked || removal.Used", patched)
        self.assertIn(b"removal is { IsStocked: true, Used: false } ? removal.Cost : null", patched)
        self.assertIn(b'["card_removal_stocked"]', patched)

    def test_rich_deck_composition_roster_archetypes_and_skip_bloat(self):
        deck = [
            {"id": "CARD.STRIKE_NECROBINDER", "name": "Strike", "type": "Attack", "cost": 1, "upgraded": False, "stats": {"damage": 6}},
            {"id": "CARD.STRIKE_NECROBINDER", "name": "Strike", "type": "Attack", "cost": 1, "upgraded": False, "stats": {"damage": 6}},
            {"id": "CARD.DEFEND_NECROBINDER", "name": "Defend", "type": "Skill", "cost": 1, "upgraded": True, "stats": {"block": 8}},
            {"id": "CARD.INJURY", "name": "Injury", "type": "Curse", "cost": -1, "upgraded": False},
            {"id": "CARD.DOOM_SLASH", "name": "Doom Slash", "type": "Attack", "cost": 1, "description": "Apply 12 Doom and Summon 3 Osty.", "keywords": ["Sly", "Exhaust"], "stats": {"doom": 12}},
        ] + [
            {"id": f"CARD.FILLER_{i}", "name": f"Filler {i}", "type": "Skill", "cost": 1, "upgraded": False}
            for i in range(12)
        ]
        state = {
            "decision": "card_reward",
            "context": {"act": 1, "floor": 10, "room_type": "Monster", "ascension": 0, "boss": {"id": "THE_KIN_BOSS"}},
            "player": {"hp": 50, "max_hp": 66, "gold": 120, "deck": deck, "deck_size": len(deck), "relics": [], "potions": []},
            "cards": [
                {"index": 0, "id": "CARD.DOOM_SLASH", "name": "Doom Slash", "type": "Attack", "cost": 1, "description": "Apply 12 Doom."},
            ],
            "can_skip": True,
        }
        actions = [
            {"action": "select_card_reward", "args": {"card_index": 0}},
            {"action": "skip_card_reward"},
        ]
        tokens = encode_tokens(state, actions)
        labels = {label for tok in tokens for label, _ in tok}
        for expected in (
            "token/deck_cards",
            "deck/count/strike",
            "deck/frac/basic",
            "deck/count/curse_status",
            "deck/arch_frac/doom",
            "deck/arch_frac/osty",
            "deck/arch_frac/sly",
            "deck/kw_frac/Sly",
            "deck/card/CARD.DOOM_SLASH",
            "deck/card_up/CARD.DEFEND_NECROBINDER",
            "action/offer/deck_copies",
            "action/skip/bloat",
        ):
            self.assertIn(expected, labels, f"Missing expected deck/strategic feature: {expected}")

    def test_inference_batcher_matches_single_decide_concurrently(self):
        from concurrent.futures import ThreadPoolExecutor

        model = PPOv2Policy(PRESETS["compact"], "cpu")
        state = sample_combat_state()
        actions = sample_actions()
        enc = pack_tokens(encode_tokens(state, actions), len(actions), model.cfg.vocab_size)
        model.eval()
        with torch.inference_mode():
            ref = model.forward_batch([enc])
            ref_logits = ref["logits"][0, :len(actions)].numpy()
            ref_val = float(ref["v_return"][0].item())

        with InferenceBatcher(model, max_batch=8, max_wait_ms=2.0) as batcher:
            with ThreadPoolExecutor(max_workers=6) as ex:
                results = list(ex.map(lambda _: batcher.predict(enc), range(12)))
            self.assertEqual(batcher.total_requests, 12)
            self.assertGreaterEqual(batcher.total_batches, 1)
            for logits_np, val in results:
                np.testing.assert_allclose(logits_np, ref_logits, atol=1e-4)
                self.assertAlmostEqual(val, ref_val, places=4)

    def test_event_option_text_and_lethal_attacker_kill_preview_features(self):
        # Event choice at full HP: Dense Vegetation "Rest" should expose combat & near-full-HP heal flags
        ev_state = {
            "decision": "event_choice",
            "event_name": "Dense Vegetation",
            "context": {"act": 1, "floor": 3, "room_type": "Event", "ascension": 0},
            "player": {"hp": 66, "max_hp": 66, "gold": 99, "deck": [], "relics": [], "potions": []},
            "options": [
                {"index": 0, "title": "Trudge On", "description": "Gain 50 Gold. Lose 11 HP."},
                {"index": 1, "title": "Rest", "description": "Heal 19 HP. Fight some enemies."},
            ],
        }
        ev_actions = [
            {"action": "choose_option", "args": {"option_index": 0}},
            {"action": "choose_option", "args": {"option_index": 1}},
        ]
        ev_tokens = encode_tokens(ev_state, ev_actions)
        rest_labels = {label for label, _ in ev_tokens[-1]}
        self.assertIn("action/option/text/fight", rest_labels)
        self.assertIn("action/option/text/enemies", rest_labels)
        self.assertIn("action/option/starts_combat", rest_labels)
        self.assertIn("action/option/heal_near_full_hp", rest_labels)

        # Combat: lethal Strike against a 4-HP attacking enemy exposes kill_attacker_dmg & kill_saves_hp
        cb_state = sample_combat_state(
            enemies=[
                {"index": 0, "name": "Wriggler", "hp": 4, "max_hp": 20, "block": 0,
                 "intends_attack": True, "intents": [{"type": "Attack", "damage": 18, "hits": 1}], "powers": []},
                {"index": 1, "name": "Wriggler", "hp": 20, "max_hp": 20, "block": 0,
                 "intends_attack": False, "intents": [{"type": "Buff"}], "powers": []},
            ]
        )
        cb_actions = [
            {"action": "play_card", "args": {"card_index": 0, "target_index": 0}},
            {"action": "play_card", "args": {"card_index": 0, "target_index": 1}},
            {"action": "end_turn"},
        ]
        cb_tokens = encode_tokens(cb_state, cb_actions)
        kill_atk_labels = {label for label, _ in cb_tokens[-3]}
        hit_buf_labels = {label for label, _ in cb_tokens[-2]}
        self.assertIn("action/target/incoming_damage", kill_atk_labels)
        self.assertIn("preview/kill_attacker_dmg", kill_atk_labels)
        self.assertIn("preview/kill_saves_hp", kill_atk_labels)
        self.assertNotIn("preview/kill_attacker_dmg", hit_buf_labels)

    def test_relic_exact_counter_description_and_shop_card_features(self):
        shop_state = {
            "decision": "shop",
            "is_blitz_shop": False,
            "context": {"act": 1, "floor": 7, "room_type": "Shop", "ascension": 0},
            "player": {
                "hp": 55, "max_hp": 66, "gold": 220, "deck_size": 1,
                "deck": [{"id": "CARD.DEFILE", "name": "Defile", "type": "Attack", "cost": 1}],
                "relics": [{
                    "name": "Pocketwatch",
                    "description": "Whenever you play 3 or fewer cards during your turn, draw 3 additional cards at the start of your next turn.",
                    "counter": 3,
                    "vars": {"Cards": 3},
                }],
                "potions": [], "potion_capacity": 3, "has_open_potion_slots": True,
            },
            "cards": [{
                "index": 0, "name": "Defile", "type": "Attack", "rarity": "Uncommon",
                "card_cost": 1, "cost": 75, "is_stocked": True, "on_sale": True,
                "description": "Deal 13 damage.",
            }],
            "relics": [], "potions": [], "card_removal_cost": 75,
        }
        actions = [
            {"action": "buy_card", "args": {"card_index": 0}},
            {"action": "leave_room"},
        ]
        tokens = encode_tokens(shop_state, actions)
        all_labels = {label for tok in tokens for label, _ in tok}
        buy_labels = {label for label, _ in tokens[-2]}
        self.assertIn("relic/counter_exact/Pocketwatch/3", all_labels)
        self.assertIn("relic/text/draw", all_labels)
        self.assertIn("relic/var/Cards", all_labels)
        self.assertIn("action/offer/deck_copies", buy_labels)
        self.assertIn("action/offer_card/rarity/Uncommon", buy_labels)
        self.assertIn("action/offer_card/on_sale", buy_labels)
        self.assertIn("action/shop/cost_frac_of_gold", buy_labels)

    def test_distributional_two_hot_critic_transfer_and_critic_warmup(self):
        # 1. Symlog/symexp invertibility and two-hot expected value accuracy across Act 1..Victory range
        bins = torch.linspace(-2.6, 2.6, 41)
        support = symexp(bins)
        targets = torch.tensor([-1.5, -1.0, 0.0, 1.5, 3.0, 7.5])
        probs = two_hot_encode(targets, bins)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones_like(targets), atol=1e-6))
        reconstructed = (probs * support).sum(dim=-1)
        self.assertTrue(torch.allclose(reconstructed, targets, atol=0.05))

        # 2. Transfer scalar checkpoint -> 41-bin distributional critic keeps >99.5% weights
        scalar_model = PPOv2Policy(PRESETS["compact"], "cpu")
        dist_cfg = replace(PRESETS["compact"], value_bins=41, value_symlog_max=2.6)
        dist_model = PPOv2Policy(dist_cfg, "cpu")
        loaded, total, skipped = dist_model.load_compatible_state_dict(scalar_model.state_dict())
        self.assertEqual(set(skipped), {"v_return.weight", "v_return.bias"})
        self.assertGreater(loaded / total, 0.995)
        # Preset mismatch must fail loudly
        wrong_model = PPOv2Policy(replace(PRESETS["base"], value_bins=41), "cpu")
        with self.assertRaises(ValueError):
            wrong_model.load_compatible_state_dict(scalar_model.state_dict())

        # 3. Initial expected value is 0.0, and critic_only=True updates critic without touching policy/backbone
        state = sample_combat_state()
        actions = sample_actions()
        enc = pack_tokens(encode_tokens(state, actions), len(actions), dist_cfg.vocab_size)
        out0 = dist_model.forward_batch([enc])
        self.assertAlmostEqual(float(out0["v_return"][0].item()), 0.0, places=5)
        self.assertEqual(out0["v_logits"].shape, (1, 41))

        opt = torch.optim.AdamW(dist_model.parameters(), lr=1e-3)
        rng = np.random.default_rng(11)
        samples = []
        for idx in range(8):
            ch, rec = dist_model.decide(state, actions, True, None, rng)
            samples.append({
                "step": rec, "advantage": 1.0 if idx % 2 == 0 else -1.0,
                "target": 7.0 if idx % 2 == 0 else -1.0,
                "final_floor": 51 if idx % 2 == 0 else 12, "combat_end_hp": 0.7,
            })
        before_policy = dist_model.policy_head[0].weight.detach().clone()
        before_embed = dist_model.embedding.weight.detach().clone()
        before_v = dist_model.v_return.weight.detach().clone()

        stats_warm = dist_model.update(samples, opt, rng, epochs=2, minibatch=4, critic_only=True)
        self.assertEqual(stats_warm["mean_kl"], 0.0)
        # Normalized two-hot CE must start near ~1.0 (divided by ln(K)), not ~3.7
        self.assertLessEqual(stats_warm["value_loss"], 1.15)
        self.assertTrue(torch.equal(before_policy, dist_model.policy_head[0].weight))
        self.assertTrue(torch.equal(before_embed, dist_model.embedding.weight))
        self.assertFalse(torch.equal(before_v, dist_model.v_return.weight))

        # 4. Normal PPO update updates both actor and two-hot critic with finite gradients
        stats_full = dist_model.update(samples, opt, rng, epochs=1, minibatch=4, critic_only=False)
        self.assertGreater(stats_full["updates"], 0)
        self.assertFalse(torch.equal(before_policy, dist_model.policy_head[0].weight))

    def test_nonterminal_floor_mask_strict_dev_rank_and_resume_validation(self):
        # 1. Non-terminal episode must set final_floor=None and mask v_floor loss cleanly
        trace = RunTraceV2()
        state = sample_combat_state()
        actions = sample_actions()
        model = PPOv2Policy(PRESETS["compact"], "cpu")
        rng = np.random.default_rng(0)
        _, rec0 = model.decide(state, actions, False, None, rng)
        _, rec1 = model.decide(state, actions, True, None, rng)
        rec0.room = (1, 1)
        rec1.room = (1, 2)
        trace.rooms = [((1, 1), 0), ((1, 2), 1)]
        trace.steps = [rec0, rec1]
        trace.confirmed = 2
        items = compute_ppo_returns(trace, {"status": "timeout", "victory": None, "act": 1, "floor": 12})
        self.assertTrue(all(item["final_floor"] is None for item in items))
        # Ensure update() handles an all-masked minibatch cleanly with finite aux_loss=0
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        stats = model.update(items, opt, rng, epochs=1, minibatch=2)
        self.assertEqual(stats["updates"], 1)
        self.assertEqual(stats["aux_loss"], 0.0)

        # 2. dev_rank rejects timeout & step_limit just like engine_error
        base = {"attempts": 32, "wins": 1, "act2_clears": 2, "act1_clears": 10, "boss_entries": 24, "mean_progress": 20.0}
        self.assertIsNotNone(dev_rank({**base, "statuses": {"defeat": 31, "victory": 1}}))
        for bad in ("engine_error", "timeout", "step_limit", "policy_error"):
            self.assertIsNone(dev_rank({**base, "statuses": {"defeat": 31, bad: 1}}))

        # 3. validate_resume_protocol catches progress, dev panel, and observation schema drift
        proto = {
            "host": "h1", "progress": "p1", "observation_version": "v2.4",
            "model_config": {"vocab_size": 8192, "width": 256, "layers": 6, "heads": 4, "ffn_width": 1024, "n_memory": 4, "value_bins": 41, "value_symlog_max": 2.6},
            "development_seeds": ["S1", "S2"],
        }
        validate_resume_protocol(proto, dict(proto))
        # Legacy protocol without observation_version remains resumable
        legacy = {k: v for k, v in proto.items() if k != "observation_version"}
        validate_resume_protocol(legacy, proto)
        with self.assertRaises(ValueError):
            validate_resume_protocol(proto, {**proto, "progress": "p2"})
        with self.assertRaises(ValueError):
            validate_resume_protocol(proto, {**proto, "development_seeds": ["S1", "S2", "S3"]})
        with self.assertRaises(ValueError):
            validate_resume_protocol(proto, {**proto, "observation_version": "v2.5"})

    def test_top_k_checkpoints_focus_acts_and_act2_entry_snapshot(self):
        import tempfile
        from pathlib import Path
        from train_curriculum import Attempt
        from train_ppo_v2 import RunTraceV2, parse_focus_acts, update_top_k_checkpoints

        # 1. parse_focus_acts validates and deduplicates act lists
        self.assertEqual(parse_focus_acts("1"), (1,))
        self.assertEqual(parse_focus_acts("2, 3, 2"), (2, 3))
        with self.assertRaises(Exception):
            parse_focus_acts("4")

        # 2. Top-K retains exactly 3 highest-ranked checkpoints and prunes evicted files
        with tempfile.TemporaryDirectory() as tmp:
            tk_dir = Path(tmp) / "top-k"
            entries: list[dict] = []
            for r, score in [(8, [5.5, 0, 0, 28, 21.0]), (16, [6.18, 0, 1, 33, 21.7]),
                             (96, [5.74, 0, 2, 27, 21.1]), (152, [5.78, 0, 1, 30, 20.8]),
                             (160, [6.00, 0, 1, 29, 22.2]), (168, [5.19, 0, 0, 25, 20.3])]:
                entries = update_top_k_checkpoints(
                    entries, k=3, round_idx=r, rank=score,
                    payload={"round": r, "rank": score, "model": {}}, top_k_dir=tk_dir,
                )
            self.assertEqual([e["round"] for e in entries], [16, 160, 152])
            pt_files = sorted(p.name for p in tk_dir.glob("*.pt"))
            self.assertEqual(pt_files, ["round-0016.pt", "round-0152.pt", "round-0160.pt"])
            self.assertTrue((tk_dir / "manifest.json").exists())

        # 3. Attempt.cases respects allowed_acts=(2, 3) and captures Act 2 normal/elite/boss fights
        attempt = Attempt(lambda s, a: 0)
        attempt.prefix = [{"action": "play_card"}] * 10
        attempt.fights = {
            (1, 17): {
                "act": 1, "floor": 17, "room": "Boss", "hp": 50, "max_hp": 66,
                "unused_potions": [], "missed_lethal_attacker": False, "scaling_enemy": False,
                "potion_kill_line": False, "opening": "a1b", "prefix_length": 4,
            },
            (2, 6): {
                "act": 2, "floor": 6, "room": "Monster", "hp": 30, "max_hp": 66,
                "unused_potions": [], "missed_lethal_attacker": False, "scaling_enemy": False,
                "potion_kill_line": False, "opening": "a2m", "prefix_length": 8,
            },
        }
        row = {
            "seed": "TESTSEED", "status": "defeat", "act": 2, "floor": 6,
            "fights": [{"act": 1, "floor": 17, "hp_in": 50, "hp_out": 40}, {"act": 2, "floor": 6, "hp_in": 30, "hp_out": 0}],
        }
        act1_only = list(attempt.cases(row, 1))
        self.assertEqual([(c["act"], c["floor"]) for c in act1_only], [(1, 17)])
        act2_only = list(attempt.cases(row, 1, allowed_acts=(2, 3)))
        self.assertEqual([(c["act"], c["floor"]) for c in act2_only], [(2, 6)])
        self.assertIn("high_hp_loss", act2_only[0]["tags"])

        # 4. RunTraceV2 records act2_entry snapshot once on entering Act 2
        trace = RunTraceV2()
        st = {
            "decision": "map_select",
            "context": {"act": 2, "floor": 1},
            "player": {
                "hp": 66, "max_hp": 66, "gold": 210,
                "deck": [
                    {"id": "STRIKE_NECRO", "name": "Strike", "type": "Attack", "upgraded": False},
                    {"id": "DEFEND_NECRO", "name": "Defend", "type": "Skill", "upgraded": True},
                ],
                "relics": [{"name": "Bound Phylactery"}],
                "potions": [{"name": "Fire Potion"}, {}],
            },
        }
        trace.observe(st)
        self.assertIsNotNone(trace.act2_entry)
        self.assertEqual(trace.act2_entry["deck_size"], 2)
        self.assertEqual(trace.act2_entry["upgraded"], 1)
        self.assertEqual(trace.act2_entry["basics"], 2)
        self.assertEqual(trace.act2_entry["potions"], ["Fire Potion"])


if __name__ == "__main__":
    unittest.main()
