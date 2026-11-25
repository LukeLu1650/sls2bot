"""Lightweight, fully trainable Entity-Transformer PPO policy trained from scratch.

Replaces the frozen random 5-layer + 5024-FFN architecture with a compact,
end-to-end trainable Pre-LN Transformer using fused scaled dot-product attention,
rich player-visible entity tokens (relics, potions, deck summary, visible map
lookahead, hand keywords/enchantments/afflictions/previews), attention-pooled
global state, a 2-layer MLP value head with auxiliary floor/combat-HP heads,
and a 2-layer candidate scoring MLP.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from queue import Empty, Queue
import re
from threading import Event, Thread
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from beta_env import action_cards
from strategic_features import MEASURES, action_details, card_values, coord, routes
from train_standard import action_features, find, incoming


OBSERVATION_VERSION = "v2.4-deck-event-relic-shop"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    width: int = 256
    layers: int = 6
    heads: int = 4
    ffn_width: int = 1024
    n_memory: int = 4
    value_bins: int = 1          # 1 = scalar Huber critic; >1 (e.g. 41) = DreamerV3 symlog two-hot critic
    value_symlog_max: float = 2.6


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def two_hot_encode(targets: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode scalar return targets into DreamerV3 symlog two-hot targets over sorted symlog bins."""
    z = symlog(targets.float()).clamp(float(bins[0]), float(bins[-1]))
    hi = torch.searchsorted(bins, z, right=False).clamp(1, bins.numel() - 1)
    lo = hi - 1
    lo_val = bins[lo]
    hi_val = bins[hi]
    w_hi = ((z - lo_val) / (hi_val - lo_val).clamp_min(1e-6)).clamp(0.0, 1.0)
    w_lo = 1.0 - w_hi
    out = torch.zeros((targets.shape[0], bins.numel()), device=targets.device, dtype=torch.float32)
    out.scatter_add_(1, lo.unsqueeze(1), w_lo.unsqueeze(1))
    out.scatter_add_(1, hi.unsqueeze(1), w_hi.unsqueeze(1))
    return out


PRESETS: dict[str, ModelConfig] = {
    # ~2.9M trainable params: fast CPU/GPU iteration and smoke runs
    "compact": ModelConfig(vocab_size=4096, width=192, layers=4, heads=4, ffn_width=768, n_memory=4),
    # ~7.1M trainable params: 6-layer 4-head 256-width lightweight backbone (default)
    "base": ModelConfig(vocab_size=8192, width=256, layers=6, heads=4, ffn_width=1024, n_memory=4),
    # ~7.1M trainable params: 6-layer 8-head 256-width backbone (matching sts2ai head split)
    "wide": ModelConfig(vocab_size=8192, width=256, layers=6, heads=8, ffn_width=1024, n_memory=4),
}


@lru_cache(maxsize=65536)
def _bucket(label: str, vocab_size: int) -> int:
    digest = hashlib.blake2s(label.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % vocab_size


def _signed_log(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _text_of(value) -> str:
    if isinstance(value, dict):
        return str(value.get("en") or "")
    return str(value or "")


def _visible_features(record: dict, namespace: str) -> list[tuple[str, float]]:
    """Allowlisted player-visible fields only (never seed, RNG, or hidden pile order)."""
    result: list[tuple[str, float]] = []
    for key in ("id", "name", "type", "rarity", "target_type", "option_id", "room_type", "text_key"):
        val = record.get(key)
        if val is not None:
            result.append((f"{namespace}/{key}/{val}", 1.0))
    for key in (
        "hp", "max_hp", "block", "amount", "counter", "damage", "hits", "total_damage",
        "cost", "card_cost", "star_cost", "energy", "max_energy", "round", "gold",
        "deck_size", "floor", "act", "ascension", "alive", "upgraded",
        "draw_pile_count", "discard_pile_count", "can_play", "intends_attack",
        "potion_capacity", "has_open_potion_slots", "card_removal_cost", "on_sale",
    ):
        val = record.get(key)
        if isinstance(val, (int, float)):
            number = float(val)
            if number != 0.0:
                result.append((f"{namespace}/{key}", _signed_log(number)))
    return result


def _card_rich_features(card: dict, namespace: str, *, include_text: bool = False) -> list[tuple[str, float]]:
    """Encode complete visible card attributes: identity, keywords, enchantments, afflictions, stats."""
    feats = _visible_features(card, namespace)
    if card.get("zone"):
        feats.append((f"{namespace}/zone/{card['zone']}", 1.0))
    for kw in card.get("keywords") or []:
        feats.append((f"{namespace}/keyword/{kw}", 1.0))
    ench = card.get("enchantment")
    if ench:
        feats.append((f"{namespace}/enchantment/{_text_of(ench)}", 1.0))
        if isinstance(card.get("enchantment_amount"), (int, float)) and card["enchantment_amount"]:
            feats.append((f"{namespace}/enchantment_amount", _signed_log(float(card["enchantment_amount"]))))
    affl = card.get("affliction")
    if affl:
        feats.append((f"{namespace}/affliction/{_text_of(affl)}", 1.0))
        if isinstance(card.get("affliction_amount"), (int, float)) and card["affliction_amount"]:
            feats.append((f"{namespace}/affliction_amount", _signed_log(float(card["affliction_amount"]))))
    for stat_name, number in (card.get("stats") or {}).items():
        if isinstance(number, (int, float)) and number != 0:
            feats.append((f"{namespace}/stat/{stat_name}", _signed_log(float(number))))
    for measure, val in zip(MEASURES, card_values(card), strict=True):
        if val:
            feats.append((f"{namespace}/measure/{measure}", float(np.clip(val, -4.0, 4.0))))
    if include_text:
        text = _text_of(card.get("description"))
        words = re.findall(r"[a-z0-9_]+", re.sub(r"\[[^\]]*\]", "", text).lower())
        feats.extend((f"{namespace}/effect/{word}", 1.0) for word in dict.fromkeys(words))
    return feats


_DECK_ARCHETYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("doom", ("doom",)),
    ("osty", ("osty", "summon")),
    ("sly", ("sly",)),
    ("soul", ("soul",)),
    ("ethereal", ("ethereal",)),
    ("exhaust", ("exhaust",)),
    ("retain", ("retain",)),
    ("draw", ("draw",)),
    ("energy", ("energy",)),
    ("aoe", ("all enemies",)),
    ("debuff", ("weak", "vulnerable")),
    ("scaling", ("strength", "dexterity")),
)


def _deck_composition_tokens(
    deck_cards: list[dict],
    boss_id: str,
) -> tuple[np.ndarray, list[tuple[str, float]], list[tuple[str, float]] | None, dict[str, int], dict[str, float]]:
    """Build rich deck summary + exact card roster tokens (all player-visible)."""
    n_cards = len(deck_cards)
    denom = max(1, n_cards)
    deck_avg = (
        sum((card_values(card) for card in deck_cards), np.zeros(len(MEASURES), dtype=np.float32)) / denom
        if deck_cards else np.zeros(len(MEASURES), dtype=np.float32)
    )
    if n_cards <= 14:
        size_bucket = "small"
    elif n_cards <= 20:
        size_bucket = "medium"
    elif n_cards <= 26:
        size_bucket = "large"
    else:
        size_bucket = "bloated"

    card_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    up_counts: Counter[str] = Counter()
    kw_counts: Counter[str] = Counter()
    arch_counts: Counter[str] = Counter()
    strikes = defends = curses_status = upgraded = attacks = skills = powers = 0
    cost_sum = 0.0
    cost_n = 0

    for card in deck_cards:
        cid = str(card.get("id") or card.get("name") or "unknown")
        cname = str(card.get("name") or "")
        ctype = str(card.get("type") or "")
        card_counts[cid] += 1
        if cname:
            name_counts[cname.lower()] += 1
        if card.get("upgraded"):
            upgraded += 1
            up_counts[cid] += 1
        cid_upper = cid.upper()
        if "STRIKE" in cid_upper or cname.lower() == "strike":
            strikes += 1
        elif "DEFEND" in cid_upper or cname.lower() == "defend":
            defends += 1
        if ctype in {"Curse", "Status"}:
            curses_status += 1
        elif ctype == "Attack":
            attacks += 1
        elif ctype == "Skill":
            skills += 1
        elif ctype == "Power":
            powers += 1
        cost = card.get("cost")
        if isinstance(cost, (int, float)) and cost >= 0:
            cost_sum += float(cost)
            cost_n += 1

        kws = [str(k) for k in (card.get("keywords") or []) if k]
        for kw in kws:
            kw_counts[kw] += 1
        stats = card.get("stats") or {}
        blob = f"{cid} {cname} {_text_of(card.get('description'))} {' '.join(kws)} {' '.join(stats.keys())}".lower()
        for arch, needles in _DECK_ARCHETYPES:
            if any(needle in blob for needle in needles):
                arch_counts[arch] += 1
        if ctype == "Power" and "scaling" not in arch_counts:
            arch_counts["scaling"] += 1

    basics = strikes + defends
    deck_token: list[tuple[str, float]] = [
        ("token/deck_summary", 1.0),
        ("deck/size", _signed_log(float(n_cards))),
        (f"deck/size_bucket/{size_bucket}", 1.0),
        (f"deck/boss/{boss_id}", 1.0),
    ]
    for measure, avg in zip(MEASURES, deck_avg, strict=True):
        if avg:
            deck_token.append((f"deck/avg/{measure}", float(np.clip(avg, -4.0, 4.0))))

    for name, count in (
        ("strike", strikes),
        ("defend", defends),
        ("basic", basics),
        ("curse_status", curses_status),
        ("upgraded", upgraded),
        ("attack", attacks),
        ("skill", skills),
        ("power", powers),
    ):
        if count:
            deck_token.append((f"deck/count/{name}", _signed_log(float(count))))
            deck_token.append((f"deck/frac/{name}", float(count) / denom))
    if cost_n:
        deck_token.append(("deck/avg/cost", cost_sum / cost_n))
    for kw, count in kw_counts.items():
        deck_token.append((f"deck/kw_count/{kw}", _signed_log(float(count))))
        deck_token.append((f"deck/kw_frac/{kw}", float(count) / denom))
    for arch, count in arch_counts.items():
        deck_token.append((f"deck/arch_count/{arch}", _signed_log(float(count))))
        deck_token.append((f"deck/arch_frac/{arch}", float(count) / denom))

    roster_token: list[tuple[str, float]] | None = None
    if card_counts:
        roster_token = [("token/deck_cards", 1.0)]
        for cid, count in card_counts.items():
            roster_token.append((f"deck/card/{cid}", _signed_log(float(count))))
            roster_token.append((f"deck/card_frac/{cid}", float(count) / denom))
        for cid, up_count in up_counts.items():
            roster_token.append((f"deck/card_up/{cid}", _signed_log(float(up_count))))

    meta = {
        "n_cards": float(n_cards),
        "basics": float(basics),
        "basic_frac": float(basics) / denom,
        "curses": float(curses_status),
        "name_counts": dict(name_counts),
    }
    return deck_avg, deck_token, roster_token, dict(card_counts), meta


def encode_tokens(state: dict, actions: list[dict], reason: str | None = None) -> list[list[tuple[str, float]]]:
    """Build complete player-visible entity + candidate action tokens for every decision scene."""
    scene = state.get("decision", "")
    player = state.get("player") or {}
    context = state.get("context") or {}
    hp = player.get("hp") or 0
    max_hp = max(1, player.get("max_hp") or 1)
    hp_frac = hp / max_hp

    # --- Token 0: Global summary (player, context, boss, Osty, current map lookahead) ---
    summary: list[tuple[str, float]] = [
        ("token/summary", 1.0),
        (f"decision/{scene}", 1.0),
        ("player/hp_fraction", hp_frac),
        ("player/low_hp", 1.0 - hp_frac),
    ]
    for label, record in (
        ("turn", state),
        ("player", player),
        ("context", context),
        ("osty", state.get("osty") or {}),
    ):
        summary.extend(_visible_features(record, label))
    boss = context.get("boss")
    boss_id = boss.get("id", "unknown") if isinstance(boss, dict) else (boss or "unknown")
    summary.append((f"context/boss/{boss_id}", 1.0))
    if state.get("event_name"):
        summary.append((f"event/name/{state['event_name']}", 1.0))

    board = state.get("visible_map")
    current = (board or {}).get("current_coord")
    current_node = next(
        (node for row in (board or {}).get("rows", []) for node in row if current and coord(node) == coord(current)),
        {},
    )
    map_view = routes(board, [coord(child) for child in current_node.get("children") or []])
    for kind, val in map_view.items():
        if val:
            summary.append((f"map/ahead/{kind}", float(val)))
            summary.append((f"map/ahead_hp/{kind}", float(val * (hp_frac - 0.5))))

    in_combat = bool(state.get("in_combat")) or scene == "combat_play"
    threat = 0.0
    exposed = 0.0
    if in_combat:
        threat = float(max(0, incoming(state) - (player.get("block") or 0)))
        exposed = float(max(0.0, threat - ((state.get("osty") or {}).get("hp") or 0)))
        if threat:
            summary.append(("combat/threat", _signed_log(threat)))
        if exposed:
            summary.append(("combat/exposed", _signed_log(exposed)))

    tokens: list[list[tuple[str, float]]] = [summary]

    # --- Token 1 & 2: Deck composition summary + exact card roster (always visible) ---
    deck_cards = player.get("deck") or []
    deck_avg, deck_token, roster_token, deck_card_counts, deck_meta = _deck_composition_tokens(deck_cards, boss_id)
    tokens.append(deck_token)
    if roster_token is not None:
        tokens.append(roster_token)

    # --- Relics tokens (with exact counter buckets & description semantics for Pocketwatch/Pen Nib/etc.) ---
    for relic in player.get("relics") or []:
        if isinstance(relic, dict):
            r_feats = [("token/relic", 1.0)] + _visible_features(relic, "relic")
            r_name = str(relic.get("name") or relic.get("id") or "unknown")
            counter = relic.get("counter")
            if isinstance(counter, (int, float)):
                c_int = int(counter)
                r_feats.append((f"relic/counter_exact/{r_name}/{min(c_int, 20)}", 1.0))
                if c_int == 0:
                    r_feats.append(("relic/counter_zero", 1.0))
            r_text = _text_of(relic.get("description"))
            if r_text:
                r_words = re.findall(r"[a-z0-9_]+", re.sub(r"\[[^\]]*\]", "", r_text).lower())
                r_feats.extend((f"relic/text/{w}", 1.0) for w in dict.fromkeys(r_words))
            for k, v in (relic.get("vars") or {}).items():
                if isinstance(v, (int, float)) and v:
                    r_feats.append((f"relic/var/{k}", _signed_log(float(v))))
            tokens.append(r_feats)
        else:
            tokens.append([("token/relic", 1.0), (f"relic/name/{relic}", 1.0)])

    # --- Potions tokens ---
    potions = player.get("potions") or []
    for potion in potions:
        if isinstance(potion, dict):
            p_feats = [("token/potion", 1.0)] + _visible_features(potion, "potion")
            for k, v in (potion.get("vars") or {}).items():
                if isinstance(v, (int, float)) and v:
                    p_feats.append((f"potion/var/{k}", _signed_log(float(v))))
            tokens.append(p_feats)

    # --- Selection prompt context ---
    selection = scene in {"card_select", "bundle_select"} or (scene == "card_reward" and state.get("in_combat"))
    if selection:
        picked = set(state.get("picked_indices") or [])
        trigger = state.get("selection_trigger") or {}
        summary.extend(_visible_features(trigger, "selection/trigger"))
        if trigger.get("action"):
            summary.append((f"selection/trigger/{trigger['action']}", 1.0))
        for field, count in (
            ("min", state.get("min_select", 1)),
            ("max", state.get("max_select", 1)),
            ("picked", len(picked)),
        ):
            summary.append((f"selection/{field}/{count}", 1.0))
        for card in state.get("cards") or []:
            ns = "selection/picked" if card["index"] in picked else "selection/offered"
            tokens.append([(ns, 1.0)] + _card_rich_features(card, ns, include_text=True))

    # --- Combat entities: enemies, hand (rich!), player_powers ---
    for enemy in state.get("enemies") or []:
        e_feats = [("token/enemy", 1.0)] + _visible_features(enemy, "enemies")
        ehp = enemy.get("hp") or 0
        emax = max(1, enemy.get("max_hp") or 1)
        e_feats.append(("enemies/hp_fraction", ehp / emax))
        for detail in ("intents", "powers"):
            for effect in enemy.get(detail) or []:
                e_feats.extend(_visible_features(effect, f"enemies/{detail}"))
        tokens.append(e_feats)

    for card in state.get("hand") or []:
        tokens.append([("token/hand", 1.0)] + _card_rich_features(card, "hand", include_text=False))

    for power in state.get("player_powers") or []:
        tokens.append([("token/player_power", 1.0)] + _visible_features(power, "player_powers"))

    # --- Candidate action tokens (always the last len(actions) tokens) ---
    energy = state.get("energy") or 0
    pot_cap = player.get("potion_capacity") or 3

    def lookup(group: str, index) -> dict:
        for pos, entity in enumerate(state.get(group) or []):
            if isinstance(entity, dict) and entity.get("index", pos) == index:
                return entity
        return {}

    for cmd in actions:
        kind = cmd["action"]
        args = cmd.get("args") or {}
        a_feats: list[tuple[str, float]] = [
            ("token/action", 1.0),
            (f"action/{kind}", 1.0),
            (f"action/{scene}/{kind}", 1.0),
        ]
        for key, value in sorted(args.items()):
            if not selection or key not in {"indices", "card_index", "bundle_index"}:
                a_feats.append((f"argument/{key}/{value}", 1.0))

        if selection:
            cards = action_cards(state, cmd)
            a_feats.append((f"selection/action/count/{len(cards)}", 1.0))
            for card in cards:
                a_feats.extend(_card_rich_features(card, "selection/action/card", include_text=True))
        elif kind == "play_card":
            card = lookup("hand", args.get("card_index"))
            a_feats.extend(_card_rich_features(card, "action/card", include_text=False))
            target_incoming = 0.0
            if "target_index" in args:
                target = lookup("enemies", args["target_index"])
                a_feats.extend(_visible_features(target, "action/target"))
                target_incoming = float(incoming({"enemies": [target]})) if target else 0.0
                if target_incoming > 0.0:
                    a_feats.append(("action/target/incoming_damage", _signed_log(target_incoming)))
                    a_feats.append(("action/target/is_attacker", 1.0))
            damage, kill, shield, summon, doom, cost, draw = action_features(state, cmd, threat)
            if damage:
                a_feats.append(("preview/damage", _signed_log(float(damage))))
            if kill:
                a_feats.append(("preview/kill", 1.0))
                if target_incoming > 0.0:
                    a_feats.append(("preview/kill_attacker_dmg", _signed_log(target_incoming)))
                    if exposed > 0.0:
                        a_feats.append(("preview/kill_saves_hp", _signed_log(float(min(exposed, target_incoming)))))
            if shield:
                a_feats.append(("preview/useful_block", _signed_log(float(min(exposed, shield)))))
            if summon:
                a_feats.append(("preview/useful_summon", _signed_log(float(min(exposed, summon)))))
            if doom:
                a_feats.append(("preview/doom", _signed_log(float(doom))))
            if draw:
                a_feats.append(("preview/draw", _signed_log(float(draw))))
            if cost >= energy > 0:
                if exposed > 0 and shield > 0:
                    a_feats.append(("preview/last_energy_shield", _signed_log(float(min(exposed, shield)))))
                if exposed > 0 and damage > 0 and not kill:
                    a_feats.append(("preview/last_energy_exposed_nonlethal", _signed_log(float(exposed))))
        elif kind == "use_potion":
            potion = find(potions, args.get("potion_index"))
            a_feats.extend(_visible_features(potion, "action/potion"))
            if "target_index" in args:
                target = lookup("enemies", args["target_index"])
                a_feats.extend(_visible_features(target, "action/target"))
                target_incoming = float(incoming({"enemies": [target]})) if target else 0.0
                if target_incoming > 0.0:
                    a_feats.append(("action/target/incoming_damage", _signed_log(target_incoming)))
                    pot_dmg = float((potion.get("vars") or {}).get("Damage", 0) or 0)
                    t_ehp = float((target.get("hp") or 0) + (target.get("block") or 0))
                    if pot_dmg > 0.0 and pot_dmg >= t_ehp > 0.0:
                        a_feats.append(("preview/kill", 1.0))
                        a_feats.append(("preview/kill_attacker_dmg", _signed_log(target_incoming)))
            if exposed > 0:
                a_feats.append(("action/potion/exposed", _signed_log(exposed)))
            if len(potions) >= pot_cap:
                a_feats.append(("action/potion/slots_full", 1.0))
        elif kind == "select_map_node":
            room_type = next(
                (item.get("type", "?") for item in state.get("choices") or [] if coord(item) == coord(args)),
                "?",
            )
            a_feats.append((f"action/map_room/{room_type}", 1.0))
            a_feats.append((f"action/map_room_hp/{room_type}", hp_frac - 0.5))
            node_routes = routes(board, [(args["col"], args["row"])])
            for rk, rv in node_routes.items():
                if rv:
                    a_feats.append((f"action/route/{rk}", float(rv)))
                    a_feats.append((f"action/route_hp/{rk}", float(rv * (hp_frac - 0.5))))
        else:
            shop_cost = None
            if "card_index" in args:
                card = lookup("cards", args["card_index"])
                a_feats.extend(_card_rich_features(card, "action/offer_card", include_text=True))
                cid = str(card.get("id") or "")
                cname = str(card.get("name") or "").lower()
                existing = deck_card_counts.get(cid, 0) if cid else 0
                if not existing and cname:
                    existing = deck_meta["name_counts"].get(cname, 0)
                if existing:
                    a_feats.append(("action/offer/deck_copies", _signed_log(float(existing))))
                if kind == "buy_card" and isinstance(card.get("cost"), (int, float)):
                    shop_cost = float(card["cost"])
            if "relic_index" in args:
                relic = lookup("relics", args["relic_index"])
                a_feats.extend(_visible_features(relic, "action/offer_relic"))
                r_text = _text_of(relic.get("description"))
                r_words = re.findall(r"[a-z0-9_]+", re.sub(r"\[[^\]]*\]", "", r_text).lower())
                a_feats.extend((f"action/offer_relic/effect/{w}", 1.0) for w in dict.fromkeys(r_words))
                if isinstance(relic.get("cost"), (int, float)):
                    shop_cost = float(relic["cost"])
            if "potion_index" in args:
                shop_pot = lookup("potions", args["potion_index"])
                a_feats.extend(_visible_features(shop_pot, "action/offer_potion"))
                if isinstance(shop_pot.get("cost"), (int, float)):
                    shop_cost = float(shop_pot["cost"])
            if kind == "remove_card" and isinstance(state.get("card_removal_cost"), (int, float)):
                shop_cost = float(state["card_removal_cost"])
            if shop_cost is not None and (player.get("gold") or 0) > 0:
                gold_f = float(player["gold"])
                a_feats.append(("action/shop/cost_frac_of_gold", min(1.0, shop_cost / gold_f)))
                a_feats.append(("action/shop/gold_left_frac", max(0.0, gold_f - shop_cost) / gold_f))
            if "option_index" in args:
                opt = lookup("options", args["option_index"])
                a_feats.extend(_visible_features(opt, "action/option"))
                opt_text = f"{_text_of(opt.get('title'))} {_text_of(opt.get('description'))}"
                opt_words = re.findall(r"[a-z0-9_]+", re.sub(r"\[[^\]]*\]", "", opt_text).lower())
                a_feats.extend((f"action/option/text/{w}", 1.0) for w in dict.fromkeys(opt_words))
                opt_lower = opt_text.lower()
                if "heal" in opt_lower:
                    if hp_frac >= 0.90:
                        a_feats.append(("action/option/heal_near_full_hp", 1.0))
                    else:
                        a_feats.append(("action/option/heal_missing_hp", 1.0 - hp_frac))
                if any(k in opt_lower for k in ("fight", "combat", "enemies", "enemy", "ambush")):
                    a_feats.append(("action/option/starts_combat", 1.0))
                if "curse" in opt_lower:
                    a_feats.append(("action/option/adds_curse", 1.0))
                for vk, vv in (opt.get("vars") or {}).items():
                    if isinstance(vv, (int, float)) and vv:
                        a_feats.append((f"action/option/var/{vk}", _signed_log(float(vv))))
            if kind == "skip_card_reward":
                a_feats.append(("action/skip/deck_size", _signed_log(deck_meta["n_cards"])))
                if deck_meta["n_cards"] > 14.0:
                    a_feats.append(("action/skip/bloat", min(3.0, (deck_meta["n_cards"] - 14.0) / 8.0)))
                if deck_meta["basic_frac"] > 0.0:
                    a_feats.append(("action/skip/basic_frac", deck_meta["basic_frac"]))
            elif kind == "remove_card":
                if deck_meta["basics"] > 0.0:
                    a_feats.append(("action/remove/basic_count", _signed_log(deck_meta["basics"])))
                if deck_meta["curses"] > 0.0:
                    a_feats.append(("action/remove/curse_count", _signed_log(deck_meta["curses"])))
            title, detail = action_details(state, cmd, reason)
            if title:
                a_feats.append((f"action/offer/{title}", 1.0))
            delta = detail.pop("deck_change", None)
            for dk, dv in detail.items():
                if dv:
                    a_feats.append((f"action/detail/{dk}", float(np.clip(dv, -5.0, 5.0))))
            if delta is not None:
                for measure, avg, d_val in zip(MEASURES, deck_avg, delta, strict=True):
                    if d_val:
                        a_feats.append((f"action/marginal/{measure}", float(np.clip(d_val, -5.0, 5.0))))
                        a_feats.append((f"action/gap/{measure}", float(np.clip(d_val * (0.5 - avg), -5.0, 5.0))))
        tokens.append(a_feats)

    return tokens


@dataclass
class EncodedDecision:
    """Compact CPU representation of one decision for end-to-end batched training."""
    indices: np.ndarray     # int32 [nnz]
    rows: np.ndarray        # int16 [nnz]
    values: np.ndarray      # float32 [nnz]
    n_tokens: int
    n_actions: int


@dataclass
class StepRecord:
    encoded: EncodedDecision
    combat: bool
    choice: int
    log_prob: float
    value: float
    room: tuple[int, int]
    floor_total: int


def pack_tokens(tokens: list[list[tuple[str, float]]], n_actions: int, vocab_size: int) -> EncodedDecision:
    indices, rows, values = [], [], []
    for row_idx, feat_list in enumerate(tokens):
        for label, val in feat_list:
            indices.append(_bucket(label, vocab_size))
            rows.append(row_idx)
            values.append(val)
    return EncodedDecision(
        indices=np.asarray(indices, dtype=np.int32),
        rows=np.asarray(rows, dtype=np.int16),
        values=np.asarray(values, dtype=np.float32),
        n_tokens=len(tokens),
        n_actions=n_actions,
    )


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block using fused scaled_dot_product_attention and GELU."""
    def __init__(self, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.ln1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, ffn_width)
        self.ff2 = nn.Linear(ffn_width, width)

    def forward(self, x: torch.Tensor, keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, width = x.shape
        qkv = self.qkv(self.ln1(x)).view(batch, seq, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn_mask = keep_mask[:, None, None, :] if keep_mask is not None else None
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, seq, width))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class PPOv2Policy(nn.Module):
    """Fully trainable Entity-Transformer actor-critic for zero-human-data RL."""

    def __init__(self, cfg: ModelConfig = PRESETS["base"], device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        d = cfg.width

        self.embedding = nn.Embedding(cfg.vocab_size, d)
        self.memory = nn.Parameter(torch.empty(cfg.n_memory, d))
        self.blocks = nn.ModuleList(
            TransformerBlock(d, cfg.heads, cfg.ffn_width) for _ in range(cfg.layers)
        )
        self.final_norm = nn.LayerNorm(d)

        # Attention-pooled global summary: concat(token_0, attn_pool(all live tokens)) -> 2*d
        self.pool_query = nn.Parameter(torch.empty(d))

        # 2-layer MLP Policy Head scoring each candidate against global summary
        self.policy_head = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

        # 2-layer MLP Value Head + auxiliary run-floor / combat-HP prediction heads
        self.value_trunk = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
        )
        self.v_return = nn.Linear(d, max(1, int(cfg.value_bins)))
        self.v_floor = nn.Linear(d, 1)
        self.v_hp = nn.Linear(d, 1)

        if cfg.value_bins > 1:
            bins = torch.linspace(-cfg.value_symlog_max, cfg.value_symlog_max, cfg.value_bins, dtype=torch.float32)
            self.register_buffer("value_bins_symlog", bins, persistent=False)
            self.register_buffer("value_bins_support", symexp(bins), persistent=False)
        else:
            self.register_buffer("value_bins_symlog", torch.empty(0, dtype=torch.float32), persistent=False)
            self.register_buffer("value_bins_support", torch.empty(0, dtype=torch.float32), persistent=False)

        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.normal_(self.memory, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)
        # Small final heads for stable initial entropy and zero-centered value estimates
        nn.init.zeros_(self.v_return.weight)
        nn.init.zeros_(self.v_return.bias)
        nn.init.zeros_(self.v_floor.weight)
        nn.init.zeros_(self.v_floor.bias)
        nn.init.zeros_(self.v_hp.weight)
        nn.init.zeros_(self.v_hp.bias)

    def parameter_summary(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor]) -> tuple[int, int, list[str]]:
        """Load matching weights from a parent checkpoint, allowing v_return head upgrades (scalar <-> two-hot)."""
        current = self.state_dict()
        compatible = {}
        skipped = []
        loaded_params = 0
        total_params = sum(t.numel() for t in current.values())
        for key, target in current.items():
            src = state_dict.get(key)
            if src is not None and src.shape == target.shape:
                compatible[key] = src
                loaded_params += target.numel()
            else:
                skipped.append(key)
        allowed_skip = {"v_return.weight", "v_return.bias"}
        unexpected = [k for k in skipped if k not in allowed_skip]
        if unexpected:
            raise ValueError(f"Incompatible checkpoint weights (mismatched layers: {unexpected[:6]})")
        self.load_state_dict(compatible, strict=False)
        return loaded_params, total_params, skipped

    def _embed_batch(self, batch: list[EncodedDecision]) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = len(batch)
        max_tok = max(item.n_tokens for item in batch)
        d = self.cfg.width
        device = self.device

        # Flatten sparse bag-of-features across the entire minibatch in one vectorized index_add_
        flat_indices = np.concatenate([item.indices for item in batch])
        flat_values = np.concatenate([item.values for item in batch])
        flat_rows = np.concatenate([item.rows.astype(np.int32) + b * max_tok for b, item in enumerate(batch)])

        idx_t = torch.from_numpy(flat_indices).to(device=device, dtype=torch.long)
        val_t = torch.from_numpy(flat_values).to(device=device, dtype=self.embedding.weight.dtype)
        row_t = torch.from_numpy(flat_rows).to(device=device, dtype=torch.long)

        x = self.embedding.weight.new_zeros((bsz * max_tok, d))
        x.index_add_(0, row_t, self.embedding(idx_t) * val_t[:, None])
        x = x.view(bsz, max_tok, d)

        lengths = torch.tensor([item.n_tokens for item in batch], device=device, dtype=torch.long)
        tok_mask = torch.arange(max_tok, device=device)[None, :] < lengths[:, None]
        return x, tok_mask

    def forward_batch(
        self,
        batch: list[EncodedDecision],
        *,
        detach_critic_backbone: bool = False,
    ) -> dict[str, torch.Tensor]:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            x, tok_mask = self._embed_batch(batch)
            bsz = len(batch)
            m = self.cfg.n_memory
            d = self.cfg.width

            mem = self.memory.unsqueeze(0).expand(bsz, -1, -1)
            h = torch.cat([x, mem], dim=1)
            mem_mask = torch.ones((bsz, m), dtype=torch.bool, device=self.device)
            keep_mask = torch.cat([tok_mask, mem_mask], dim=1)

            for block in self.blocks:
                h = block(h, keep_mask)
            h = self.final_norm(h)

            # Global summary: [token_0, attention-pooled live tokens]
            pool_scores = (h.float() @ self.pool_query.float()) / math.sqrt(d)
            pool_scores = pool_scores.masked_fill(~keep_mask, float("-inf"))
            pooled = torch.einsum("bs,bsd->bd", pool_scores.softmax(-1).to(h.dtype), h)
            summary = torch.cat([h[:, 0], pooled], dim=-1)  # [B, 2*d]

            # Candidate action logits (padded to max_actions in batch)
            max_act = max(item.n_actions for item in batch)
            cand = h.new_zeros((bsz, max_act, d))
            act_mask = torch.zeros((bsz, max_act), dtype=torch.bool, device=self.device)
            for b, item in enumerate(batch):
                start = item.n_tokens - item.n_actions
                cand[b, :item.n_actions] = h[b, start:item.n_tokens]
                act_mask[b, :item.n_actions] = True

            sum_exp = summary.unsqueeze(1).expand(-1, max_act, -1)
            logits = self.policy_head(torch.cat([cand, sum_exp], dim=-1)).squeeze(-1).float()
            logits = torch.where(act_mask, logits, torch.full_like(logits, -1e9))

            v_in = summary.detach() if detach_critic_backbone else summary
            v_feat = self.value_trunk(v_in)
            v_raw = self.v_return(v_feat).float()
            if self.cfg.value_bins > 1:
                v_logits = v_raw
                v_return = (F.softmax(v_logits, dim=-1) * self.value_bins_support).sum(dim=-1)
            else:
                v_logits = None
                v_return = v_raw.squeeze(-1)
            v_floor = self.v_floor(v_feat).squeeze(-1).float()
            v_hp = self.v_hp(v_feat).squeeze(-1).float()
        return {
            "logits": logits,
            "act_mask": act_mask,
            "v_return": v_return,
            "v_logits": v_logits,
            "v_floor": v_floor,
            "v_hp": v_hp,
        }

    @staticmethod
    def _sample_from_logits(
        state: dict,
        actions: list[dict],
        encoded: EncodedDecision,
        logits_np: np.ndarray,
        value_scalar: float,
        combat: bool,
        rng: np.random.Generator | None,
        *,
        combat_temp: float = 0.7,
        strategy_temp: float = 0.9,
    ) -> tuple[int, StepRecord | None]:
        logits_1d = np.asarray(logits_np[:len(actions)], dtype=np.float64)
        if rng is None:
            return int(np.argmax(logits_1d)), None
        temp = max(combat_temp if combat else strategy_temp, 1e-4)
        scaled = logits_1d / temp
        scaled = scaled - float(np.max(scaled))
        log_probs = scaled - float(np.log(np.sum(np.exp(scaled))))
        probs = np.exp(log_probs)
        probs = probs / probs.sum()
        choice = int(rng.choice(len(actions), p=probs))
        ctx = state["context"]
        act = ctx.get("act") or 1
        floor = ctx.get("floor") or 0
        record = StepRecord(
            encoded=encoded,
            combat=combat,
            choice=choice,
            log_prob=float(log_probs[choice]),
            value=float(value_scalar),
            room=(act, floor),
            floor_total=(act - 1) * 17 + min(floor, 17),
        )
        return choice, record

    @torch.inference_mode()
    def decide(
        self,
        state: dict,
        actions: list[dict],
        combat: bool,
        reason: str | None,
        rng: np.random.Generator | None,
        *,
        combat_temp: float = 0.7,
        strategy_temp: float = 0.9,
    ) -> tuple[int, StepRecord | None]:
        self.eval()
        tokens = encode_tokens(state, actions, reason)
        encoded = pack_tokens(tokens, len(actions), self.cfg.vocab_size)
        out = self.forward_batch([encoded])
        logits_np = out["logits"][0, :len(actions)].cpu().numpy()
        value_scalar = float(out["v_return"][0].item())
        return self._sample_from_logits(
            state, actions, encoded, logits_np, value_scalar, combat, rng,
            combat_temp=combat_temp, strategy_temp=strategy_temp,
        )

    def update(
        self,
        samples: list[dict],
        optimizer: torch.optim.Optimizer,
        rng: np.random.Generator,
        *,
        epochs: int = 3,
        minibatch: int = 128,
        clip: float = 0.2,
        vclip: float = 0.2,
        target_kl: float = 0.025,
        ent_combat: float = 0.005,
        ent_strategic: float = 0.015,
        vf_coef: float = 0.5,
        aux_coef: float = 0.2,
        combat_temp: float = 0.7,
        strategy_temp: float = 0.9,
        max_grad_norm: float = 0.5,
        bc_samples: list[tuple[EncodedDecision, int]] | None = None,
        bc_coef: float = 0.0,
        critic_only: bool = False,
    ) -> dict:
        """PPO update with separate combat/strategic advantage normalization, scalar Huber or DreamerV3 two-hot value loss, aux heads, and optional BC."""
        if not samples:
            raise ValueError("PPOv2 received no confirmed decisions")
        self.train()

        n = len(samples)
        adv = np.array([s["advantage"] for s in samples], dtype=np.float32)
        is_combat = np.array([s["step"].combat for s in samples], dtype=bool)
        # Normalize combat and strategic advantages separately so ~15% strategic choices aren't drowned out
        for mask in (is_combat, ~is_combat):
            if mask.sum() >= 2:
                vals = adv[mask]
                adv[mask] = (vals - vals.mean()) / max(float(vals.std()), 0.01)
            elif mask.sum() == 1:
                adv[mask] = 0.0
        adv = np.clip(adv, -5.0, 5.0)

        losses, actor_losses, value_losses, aux_losses, kls, ents_c, ents_s, bc_losses = (
            [], [], [], [], [], [], [], []
        )
        update_count = 0
        early_stopped = False

        for _ in range(epochs):
            order = rng.permutation(n)
            epoch_kls = []
            for start in range(0, n, minibatch):
                idx = order[start:start + minibatch]
                batch_items = [samples[i] for i in idx]
                encoded_batch = [item["step"].encoded for item in batch_items]

                out = self.forward_batch(encoded_batch, detach_critic_backbone=critic_only)
                logits = out["logits"]
                act_mask = out["act_mask"]
                temps = torch.tensor(
                    [combat_temp if item["step"].combat else strategy_temp for item in batch_items],
                    device=self.device,
                    dtype=logits.dtype,
                ).unsqueeze(-1)
                scaled_logits = torch.where(act_mask, logits / temps.clamp_min(1e-4), torch.full_like(logits, -1e9))
                log_probs_all = F.log_softmax(scaled_logits, dim=-1)
                choices = torch.tensor([item["step"].choice for item in batch_items], device=self.device, dtype=torch.long)
                new_logp = log_probs_all.gather(1, choices[:, None]).squeeze(1)

                # Mask both factors BEFORE multiplying so backward never sees 0 * (-inf) = NaN
                probs_safe = torch.where(act_mask, log_probs_all.exp(), torch.zeros_like(log_probs_all))
                logp_safe = torch.where(act_mask, log_probs_all, torch.zeros_like(log_probs_all))
                entropy = -(probs_safe * logp_safe).sum(dim=-1)

                old_logp = torch.tensor([item["step"].log_prob for item in batch_items], device=self.device, dtype=torch.float32)
                log_ratio = new_logp - old_logp
                ratio = log_ratio.exp()
                approx_kl = 0.0 if critic_only else float((ratio - 1.0 - log_ratio).mean().detach().item())
                kls.append(approx_kl)
                epoch_kls.append(approx_kl)
                if not critic_only and approx_kl > target_kl * 1.5:
                    continue

                mb_adv = torch.tensor(adv[idx], device=self.device, dtype=torch.float32)
                pg1 = ratio * mb_adv
                pg2 = ratio.clamp(1.0 - clip, 1.0 + clip) * mb_adv
                actor_loss = torch.zeros((), device=self.device) if critic_only else -torch.minimum(pg1, pg2).mean()

                ret_target = torch.tensor([item["target"] for item in batch_items], device=self.device, dtype=torch.float32)
                if self.cfg.value_bins > 1 and out["v_logits"] is not None:
                    target_probs = two_hot_encode(ret_target, self.value_bins_symlog)
                    # Normalize by ln(K) so two-hot CE lives on [0, 1] and never drowns out actor gradients
                    vf_loss = -(target_probs * F.log_softmax(out["v_logits"], dim=-1)).sum(dim=-1).mean() / math.log(self.cfg.value_bins)
                    eff_vf_coef = vf_coef * 0.5
                else:
                    old_val = torch.tensor([item["step"].value for item in batch_items], device=self.device, dtype=torch.float32)
                    v_pred = out["v_return"]
                    v_clipped = old_val + (v_pred - old_val).clamp(-vclip, vclip)
                    vf_unclipped = F.smooth_l1_loss(v_pred, ret_target, beta=1.0, reduction="none")
                    vf_clipped = F.smooth_l1_loss(v_clipped, ret_target, beta=1.0, reduction="none")
                    vf_loss = torch.maximum(vf_unclipped, vf_clipped).mean()
                    eff_vf_coef = vf_coef

                # Auxiliary heads (zero human data: pure self-play targets; mask unknown final floor on non-terminal episodes)
                aux_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                floor_mask = torch.tensor([item.get("final_floor") is not None for item in batch_items], device=self.device, dtype=torch.bool)
                if floor_mask.any():
                    floor_target = torch.tensor(
                        [item["final_floor"] / 51.0 for item in batch_items if item.get("final_floor") is not None],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    aux_loss = aux_loss + F.mse_loss(out["v_floor"][floor_mask], floor_target)
                hp_mask = torch.tensor([item["combat_end_hp"] is not None for item in batch_items], device=self.device, dtype=torch.bool)
                if hp_mask.any():
                    hp_target = torch.tensor(
                        [item["combat_end_hp"] for item in batch_items if item["combat_end_hp"] is not None],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    aux_loss = aux_loss + F.mse_loss(out["v_hp"][hp_mask], hp_target)

                mb_combat = torch.tensor([item["step"].combat for item in batch_items], device=self.device, dtype=torch.bool)
                ent_weights = torch.where(mb_combat, ent_combat, ent_strategic)
                ent_bonus = (entropy * ent_weights).mean()

                if critic_only:
                    loss = vf_loss + aux_coef * aux_loss
                else:
                    loss = actor_loss + eff_vf_coef * vf_loss + aux_coef * aux_loss - ent_bonus

                # Optional P2 offline CombatSolver BC guidance (zero human data)
                if not critic_only and bc_samples and bc_coef > 0.0:
                    bc_n = min(len(bc_samples), max(8, minibatch // 4))
                    bc_idx = rng.integers(0, len(bc_samples), size=bc_n)
                    bc_enc = [bc_samples[int(i)][0] for i in bc_idx]
                    bc_tgt = torch.tensor([bc_samples[int(i)][1] for i in bc_idx], device=self.device, dtype=torch.long)
                    bc_out = self.forward_batch(bc_enc)
                    bc_loss = F.cross_entropy(bc_out["logits"], bc_tgt)
                    loss = loss + bc_coef * bc_loss
                    bc_losses.append(float(bc_loss.detach().item()))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                update_count += 1

                losses.append(float(loss.detach().item()))
                actor_losses.append(float(actor_loss.detach().item()))
                value_losses.append(float(vf_loss.detach().item()))
                aux_losses.append(float(aux_loss.detach().item()))
                if mb_combat.any():
                    ents_c.append(float(entropy[mb_combat].mean().detach().item()))
                if (~mb_combat).any():
                    ents_s.append(float(entropy[~mb_combat].mean().detach().item()))

            if epoch_kls and float(np.mean(epoch_kls)) > target_kl:
                early_stopped = True
                break

        return {
            "decisions": n,
            "combat_decisions": int(is_combat.sum()),
            "strategic_decisions": int((~is_combat).sum()),
            "updates": update_count,
            "early_stopped": early_stopped,
            "loss": float(np.mean(losses)) if losses else None,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else None,
            "value_loss": float(np.mean(value_losses)) if value_losses else None,
            "aux_loss": float(np.mean(aux_losses)) if aux_losses else None,
            "bc_loss": float(np.mean(bc_losses)) if bc_losses else None,
            "mean_kl": float(np.mean(kls)) if kls else 0.0,
            "entropy_combat": float(np.mean(ents_c)) if ents_c else None,
            "entropy_strategic": float(np.mean(ents_s)) if ents_s else None,
        }


class InferenceBatcher:
    """Dynamic micro-batching GPU inference worker for lock-free concurrent rollouts.

    Worker threads encode decisions in parallel without holding a model lock, submit
    EncodedDecision items to a shared queue, and receive (logits, v_return) from a
    single batched BF16 forward pass on the GPU.
    """

    def __init__(self, model: PPOv2Policy, *, max_batch: int = 32, max_wait_ms: float = 1.5):
        self.model = model
        self.max_batch = max(1, int(max_batch))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self._queue: Queue = Queue()
        self._thread: Thread | None = None
        self.total_requests = 0
        self.total_batches = 0

    def __enter__(self) -> "InferenceBatcher":
        self.model.eval()
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @property
    def avg_batch_size(self) -> float:
        return self.total_requests / max(1, self.total_batches)

    def _loop(self):
        while True:
            first = self._queue.get()
            if first is None:
                return
            items = [first]
            stop_after = False
            deadline = monotonic() + self.max_wait_s
            while len(items) < self.max_batch:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    try:
                        nxt = self._queue.get_nowait()
                    except Empty:
                        break
                else:
                    try:
                        nxt = self._queue.get(timeout=remaining)
                    except Empty:
                        break
                if nxt is None:
                    stop_after = True
                    break
                items.append(nxt)

            encoded_batch = [entry[0] for entry in items]
            try:
                with torch.inference_mode():
                    out = self.model.forward_batch(encoded_batch)
                    logits_cpu = out["logits"].cpu().numpy()
                    values_cpu = out["v_return"].cpu().numpy()
                self.total_requests += len(items)
                self.total_batches += 1
                for idx, (enc, ev, slot) in enumerate(items):
                    slot["result"] = (logits_cpu[idx, :enc.n_actions], float(values_cpu[idx]))
                    ev.set()
            except BaseException as exc:
                for _, ev, slot in items:
                    slot["error"] = exc
                    ev.set()
            if stop_after:
                return

    def predict(self, encoded: EncodedDecision) -> tuple[np.ndarray, float]:
        ev = Event()
        slot: dict = {}
        self._queue.put((encoded, ev, slot))
        ev.wait()
        if "error" in slot:
            raise slot["error"]
        return slot["result"]

    def decide(
        self,
        state: dict,
        actions: list[dict],
        combat: bool,
        reason: str | None,
        rng: np.random.Generator | None,
        *,
        combat_temp: float = 0.7,
        strategy_temp: float = 0.9,
    ) -> tuple[int, StepRecord | None]:
        tokens = encode_tokens(state, actions, reason)
        encoded = pack_tokens(tokens, len(actions), self.model.cfg.vocab_size)
        logits_np, value_scalar = self.predict(encoded)
        return self.model._sample_from_logits(
            state, actions, encoded, logits_np, value_scalar, combat, rng,
            combat_temp=combat_temp, strategy_temp=strategy_temp,
        )
