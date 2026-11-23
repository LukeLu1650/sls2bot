"""Player-visible features for the three standard-mode policy variants."""
from collections import Counter
from functools import lru_cache
import hashlib
import math

import numpy as np


VARIANTS = ("current", "route", "boss_deck")
FEATURE_COUNT = 4096


@lru_cache(maxsize=32768)
def bucket(key):
    return int.from_bytes(hashlib.blake2s(key.encode("utf-8"), digest_size=4).digest(), "little") % FEATURE_COUNT


def coord(node):
    return node["col"], node["row"]


def routes(view, starts, depth=4):
    """Average the visible reachable room types by depth, without reading encounters."""
    board = {coord(node): node for row in (view or {}).get("rows", []) for node in row}
    layer = set(starts)
    totals = Counter()
    for distance in range(depth):
        nodes = [board[key] for key in sorted(layer) if key in board]
        if not nodes:
            break
        factor = 1 / len(nodes) / (1 + distance)
        for node in nodes:
            totals[node.get("type", "?")] += factor
        totals["width"] += min(len(nodes), 5) / 5 * (1 / (1 + distance))
        layer = {coord(child) for node in nodes for child in node.get("children") or []}
    return totals


def card_values(card):
    stats = card.get("stats") or {}
    damage = stats.get("calculateddamage", stats.get("damage", 0)) * max(1, stats.get("repeat", 1))
    area = card.get("target_type") == "AllEnemies"
    if "target_type" not in card:
        description = card.get("description") or ""
        description = description if isinstance(description, str) else description.get("en", "")
        area = "all enemies" in description.lower()
    return np.array([
        damage / 12, damage / 12 if area else 0, stats.get("block", 0) / 10,
        stats.get("cards", stats.get("draw", 0)) / 2, stats.get("energy", 0) / 2,
        max(0, card.get("card_cost", card.get("cost", 0)) or 0) / 2,
        stats.get("summon", 0) / 8, stats.get("doom", 0) / 10,
        float(card.get("type") in {"Curse", "Status"}), float(card.get("upgraded", False)),
    ], dtype=np.float32)


def indexed(records, index):
    return next((item for item in records or [] if item.get("index") == index), {})


def action_details(state, action, reason):
    name, args = action["action"], action.get("args", {})
    card, sign, upgrade = {}, 0, False
    title = None
    values = {}
    if name in {"select_card_reward", "buy_card"}:
        card = indexed(state.get("cards"), args["card_index"])
        sign = 1
        values["deck_size_change"] = 1
    elif name == "select_cards":
        cards = [indexed(state.get("cards"), int(index)) for index in args["indices"].split(",")]
        cards = [card for card in cards if card]
        if reason in {"remove_card", "smith"} and cards:
            card, sign, upgrade = cards[0], -1 if reason == "remove_card" else 1, reason == "smith"
            deltas = [card_values(item) for item in cards] if not upgrade else [
                card_values({**item, **item["after_upgrade"], "upgraded": True}) - card_values(item)
                for item in cards if isinstance(item.get("after_upgrade"), dict) and item["after_upgrade"].get("stats")
            ]
            if deltas:
                values["deck_change"] = sum(deltas) * sign
                values["deck_size_change"] = -len(cards) if not upgrade else 0
    elif name == "choose_option":
        row = indexed(state.get("options"), args["option_index"])
        title = row.get("text_key") or row.get("option_id") or row.get("title")
        variables = row.get("vars") or {}
        for key in ("Gold", "Heal", "HpLoss", "MaxHp", "Cards"):
            if isinstance(variables.get(key), (int, float)):
                values[key] = variables[key] / 25
    elif name in {"buy_relic", "buy_potion"}:
        kind = name.removeprefix("buy_")
        offer = indexed(state.get(kind + "s"), args[f"{kind}_index"])
        title = offer.get("name")
        values["cost"] = offer.get("cost", 0) / 100
    if card:
        title = card.get("id") or card.get("name")
        values.setdefault("deck_change", card_values(card) * sign)
        if name == "buy_card":
            values["cost"] = card.get("cost", 0) / 100
    if name == "remove_card":
        values["cost"] = (state.get("card_removal_cost") or 0) / 100
    return title, values


MEASURES = ("damage", "area", "block", "draw", "energy", "cost", "osty", "doom", "dead_draw", "upgraded")


def features(state, actions, variant, reason):
    """Action-conditioned feature rows. Every candidate, including skipping, has a row."""
    scene, player = state["decision"], state["player"]
    context = state.get("context") or {}
    hp = player["hp"] / max(1, player["max_hp"])
    act = context.get("act") or 1
    board = state.get("visible_map") if variant != "current" else None
    current = (board or {}).get("current_coord")
    view = routes(board, [coord(current)]) if current else {}
    count = max(1, len(player.get("deck") or []))
    deck = (sum((card_values(card) for card in player.get("deck") or []), np.zeros(len(MEASURES))) / count
            if variant == "boss_deck" else None)
    boss = context.get("boss") or {}
    boss_id = boss.get("id", "unknown") if isinstance(boss, dict) else boss
    matrix = np.zeros((len(actions), FEATURE_COUNT), dtype=np.float32)
    for row, action in zip(matrix, actions, strict=True):
        name, args = action["action"], action.get("args", {})

        def add(key, value=1.0):
            if value:
                row[bucket(key)] += max(-5.0, min(5.0, float(value)))

        prefix = f"{scene}/{name}"
        add(prefix)
        add(prefix + "/hp", hp - 0.5)
        add(prefix + "/act", act / 3)
        for key, value in sorted(args.items()):
            add(f"{prefix}/position/{key}/{value}")
        if variant != "current":
            summary = routes(board, [(args["col"], args["row"])]) if name == "select_map_node" else view
            for kind, value in summary.items():
                add(prefix + "/route/" + kind, value)
                add(prefix + "/route-hp/" + kind, value * (hp - 0.5))
            if name == "select_map_node":
                room = next((item["type"] for item in state.get("choices", []) if coord(item) == coord(args)), "?")
                add(prefix + "/" + room)
                add(prefix + "/" + room + "/hp", hp - 0.5)
        if deck is None:
            continue
        title, detail = action_details(state, action, reason)
        add(f"{prefix}/boss/{boss_id}")
        if title:
            add(f"{prefix}/offer/{title}")
        change = detail.pop("deck_change", np.zeros(len(MEASURES)))
        for key, value in detail.items():
            add(f"{prefix}/offer-value/{key}", value)
        for measure, average, delta in zip(MEASURES, deck, change, strict=True):
            add(f"{prefix}/deck/{measure}", average)
            add(f"{prefix}/marginal/{measure}", delta)
            add(f"{prefix}/gap/{measure}", delta * (0.5 - average))
            add(f"{prefix}/boss/{boss_id}/marginal/{measure}", delta)
    return matrix


def probabilities(logits):
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / math.fsum(values)
