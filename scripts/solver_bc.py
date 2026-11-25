"""Optional CombatSolver offline guidance helpers (zero human data).

Loads CombatSolver route outputs (route.json) or recorded solver demonstration
JSONL files ({state, actions, chosen}) and converts them into EncodedDecision
targets for PPOv2Policy's optional BC regularizer (--solver-demos / --bc-coef).
"""
from __future__ import annotations

import json
from pathlib import Path

from ppo_v2_policy import EncodedDecision, encode_tokens, pack_tokens


def match_solver_step(step: dict, state: dict, actions: list[dict]) -> int | None:
    """Map one CombatSolver route.json action entry to a legal beta_env action index."""
    kind = step.get("kind")
    if kind == "EndTurn":
        return next((i for i, a in enumerate(actions) if a["action"] == "end_turn"), None)

    if kind == "PlayCard":
        card_id = str(step.get("cardId") or "").upper()
        occurrence = int(step.get("cardOccurrence") or 0)
        target_idx = step.get("targetIndex")
        hand = state.get("hand") or []
        matching_cards = [
            c["index"] for c in hand
            if str(c.get("id") or "").upper().endswith("." + card_id)
            or str(c.get("id") or "").upper() == card_id
        ]
        if not matching_cards:
            return None
        chosen_card_index = matching_cards[min(occurrence, len(matching_cards) - 1)]
        for i, a in enumerate(actions):
            if a["action"] != "play_card":
                continue
            args = a.get("args") or {}
            if args.get("card_index") != chosen_card_index:
                continue
            if "target_index" in args and target_idx is not None and int(target_idx) >= 0:
                if args["target_index"] != int(target_idx):
                    continue
            return i
        return None

    if kind == "UsePotion":
        slot = int(step.get("potionSlot", -1))
        target_idx = step.get("targetIndex")
        for i, a in enumerate(actions):
            if a["action"] != "use_potion":
                continue
            args = a.get("args") or {}
            if slot >= 0 and args.get("potion_index") != slot:
                continue
            if "target_index" in args and target_idx is not None and int(target_idx) >= 0:
                if args["target_index"] != int(target_idx):
                    continue
            return i
        return None

    return None


def load_bc_demos(path: Path, vocab_size: int) -> list[tuple[EncodedDecision, int]]:
    """Load recorded solver demonstrations from a .jsonl file or directory of .jsonl files.

    Each JSON line must contain: {"state": <decision dict>, "actions": <legal actions>, "chosen": <int>}.
    """
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    demos: list[tuple[EncodedDecision, int]] = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            state = row["state"]
            actions = row["actions"]
            chosen = int(row["chosen"])
            if len(actions) < 2 or not 0 <= chosen < len(actions):
                continue
            tokens = encode_tokens(state, actions, row.get("reason"))
            demos.append((pack_tokens(tokens, len(actions), vocab_size), chosen))
    return demos
