from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_strategy import summarize


class StrategyReportTests(unittest.TestCase):
    def test_reports_boss_health_with_and_without_before_travel_snapshot(self):
        def attempt(hp, before):
            return {
                "act": 1, "victory": False, "status": "defeat",
                "offered_candidates": {}, "actions": {},
                "boss_entries": [{"act": 1, "hp": hp, "max_hp": 66, "before_travel": before}],
            }

        report = summarize([attempt(20, {"hp": 24}), attempt(40, None)])
        boss = report["boss_hp"][0]
        self.assertEqual(boss["samples"], 2)
        self.assertEqual(boss["median_first_input_hp"], 30)
        self.assertAlmostEqual(boss["median_first_input_hp_fraction"], 30 / 66)
        self.assertEqual(boss["before_travel_samples"], 1)
        self.assertEqual(boss["median_before_travel_hp"], 24)
