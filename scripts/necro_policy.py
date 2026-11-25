"""Untrained Necrobinder Transformer for the native beta CLI baseline (NumPy)."""
import hashlib
import json
from functools import lru_cache
import re

import numpy as np

from beta_env import action_cards


@lru_cache(maxsize=8192)
def _bucket(label):
    digest = hashlib.blake2s(label.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % 4096


def _visible_features(record, namespace):
    """Allowlisted display fields. The engine's seed, internal moves and pile order stay outside."""
    result = []
    for key in ("id", "name", "type", "target_type", "option_id", "room_type"):
        if record.get(key) is not None:
            result.append((f"{namespace}/{key}/{record[key]}", 1.0))
    for key in ("hp", "max_hp", "block", "amount", "damage", "hits", "cost", "energy",
                "round", "gold", "deck_size", "floor", "act", "alive", "upgraded",
                "draw_pile_count", "discard_pile_count", "can_play", "intends_attack"):
        value = record.get(key)
        if isinstance(value, (float, int)):
            number = float(value)
            result.append((f"{namespace}/{key}", np.sign(number) * np.log1p(abs(number))))
    return result


def _selection_card_features(card, namespace):
    result = _visible_features(card, namespace)
    if card.get("zone"):
        result.append((f"{namespace}/zone/{card['zone']}", 1.0))
    for keyword in card.get("keywords") or []:
        result.append((f"{namespace}/keyword/{keyword}", 1.0))
    text = card.get("description") or ""
    if isinstance(text, dict):
        text = text.get("en") or ""
    words = re.findall(r"[a-z0-9_]+", re.sub(r"\[[^\]]*\]", "", text).lower())
    result.extend((f"{namespace}/effect/{word}", 1.0) for word in dict.fromkeys(words))
    for name, number in (card.get("stats") or {}).items():
        if isinstance(number, (int, float)):
            result.append((f"{namespace}/stat/{name}", np.sign(number) * np.log1p(abs(number))))
    return result


def _tokens(state, actions):
    scene = state.get("decision", "")
    summary = [(f"decision/{scene}", 1.0)]
    for label, record in (("turn", state), ("player", state.get("player") or {}),
                          ("context", state.get("context") or {}), ("osty", state.get("osty") or {})):
        summary.extend(_visible_features(record, label))
    tokens = [summary]
    selection = scene in {"card_select", "bundle_select"} or (
        scene == "card_reward" and state.get("in_combat"))
    if selection:
        picked = set(state.get("picked_indices") or [])
        trigger = state.get("selection_trigger") or {}
        summary.extend(_visible_features(trigger, "selection/trigger"))
        if trigger.get("action"):
            summary.append((f"selection/trigger/{trigger['action']}", 1.0))
        for field, count in (("min", state.get("min_select", 1)), ("max", state.get("max_select", 1)),
                             ("picked", len(picked))):
            summary.append((f"selection/{field}/{count}", 1.0))
        for card in state.get("cards") or []:
            namespace = "selection/picked" if card["index"] in picked else "selection/offered"
            tokens.append([(namespace, 1.0)] + _selection_card_features(card, namespace))
    for group in ("enemies", "hand", "player_powers"):
        for item in state.get(group) or []:
            features = [(group, 1.0)] + _visible_features(item, group)
            for detail in ("intents", "powers"):
                for effect in item.get(detail) or []:
                    features.extend(_visible_features(effect, detail))
            tokens.append(features)

    def choice(group, index):
        for position, entity in enumerate(state.get(group) or []):
            if isinstance(entity, dict) and entity.get("index", position) == index:
                return _visible_features(entity, group)
        return []

    for cmd in actions:
        kind = cmd["action"]
        features = [(f"action/{kind}", 1.0)]
        args = cmd.get("args") or {}
        # Discrete options have stable meaning within a single observation.
        for key, value in sorted(args.items()):
            if not selection or key not in {"indices", "card_index", "bundle_index"}:
                features.append((f"argument/{key}/{value}", 1.0))
        if selection:
            cards = action_cards(state, cmd)
            features.append((f"selection/action/count/{len(cards)}", 1.0))
            for card in cards:
                features.extend(_selection_card_features(card, "selection/action/card"))
        elif "card_index" in args:
            group = "hand" if kind == "play_card" else "cards"
            features.extend(choice(group, args["card_index"]))
        if "target_index" in args:
            features.extend(choice("enemies", args["target_index"]))
        if "option_index" in args:
            features.extend(choice("options", args["option_index"]))
        tokens.append(features)
    return tokens


def _norm(x, scale, offset):
    centered = x - x.mean(axis=1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(axis=1, keepdims=True) + 1e-5) * scale + offset


class NecroPolicy:
    """Fixed PCG64 initialization. One masked-candidate forward pass per action."""
    width = 256
    heads = 4
    layers = 6
    ffn_width = 5024

    def __init__(self, seed=711072):
        self.seed = seed
        rng = np.random.Generator(np.random.PCG64(seed))
        self.weights = {}

        def weight(name, shape, scale=None):
            if scale is None:
                scale = shape[0] ** -0.5
            a = rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
            self.weights[name] = a
            return a

        def constant(name, size, value):
            self.weights[name] = np.full(size, value, np.float32)
            return self.weights[name]

        d, f = self.width, self.ffn_width
        self.emb = weight("embedding", (4096, d), scale=d ** -0.5)
        self.blocks = []
        for i in range(self.layers):
            k = f"block{i}/"
            self.blocks.append((
                constant(k+"ln1_scale", d, 1), constant(k+"ln1_bias", d, 0),
                weight(k+"qkv", (d, 3*d)), constant(k+"qkv_bias", 3*d, 0),
                weight(k+"out", (d, d)), constant(k+"out_bias", d, 0),
                constant(k+"ln2_scale", d, 1), constant(k+"ln2_bias", d, 0),
                weight(k+"ff1", (d, f)), constant(k+"ff1_bias", f, 0),
                weight(k+"ff2", (f, d)), constant(k+"ff2_bias", d, 0)))
        self.read_scale = constant("read_scale", d, 1)
        self.read_bias = constant("read_bias", d, 0)
        self.readout = weight("readout", (d,))
        self.readout_bias = constant("readout_bias", 1, 0)
        self.parameter_count = sum(a.size for a in self.weights.values())
        self.forward_calls = 0

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, param in self.weights.items():
            digest.update(json.dumps([name, param.shape]).encode("ascii"))
            digest.update(param.astype("<f4", copy=False).tobytes())
        return digest.hexdigest()

    def _candidate_embeddings(self, state, actions):
        if not actions:
            raise ValueError("A decision requires at least one candidate")
        tokens = self.tokens(state, actions)
        x = np.zeros((len(tokens), self.width), dtype=np.float32)
        for row, features in enumerate(tokens):
            for label, value in features:
                x[row] += self.emb[_bucket(label)] * np.float32(value)
        head_width = self.width // self.heads
        for s1, b1, qkv, qb, proj, pb, s2, b2, ff1, fb1, ff2, fb2 in self.blocks:
            all_qkv = _norm(x, s1, b1) @ qkv + qb
            q, k, v = all_qkv.reshape(len(x), 3, self.heads, head_width).transpose(1, 2, 0, 3)
            scores = (q @ k.transpose(0, 2, 1)) * np.float32(head_width ** -0.5)
            attention = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attention /= attention.sum(axis=-1, keepdims=True)
            attended = (attention @ v).transpose(1, 0, 2).reshape(len(x), self.width)
            x = x + attended @ proj + pb
            hidden = _norm(x, s2, b2) @ ff1 + fb1
            np.maximum(hidden, 0, out=hidden)
            x = x + hidden @ ff2 + fb2
        return _norm(x[-len(actions):], self.read_scale, self.read_bias)

    def fit_demonstration(self, state, actions, chosen, learning_rate=0.001):
        """One SGD cross-entropy update of the scorer on one labeled game decision.

        The Transformer stays frozen. Use training seeds only and initialize a
        separate instance for the untreated comparison. Returns pre-update loss.
        """
        if not 0 <= chosen < len(actions):
            raise ValueError("The demonstrated index must identify an offered action")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        candidates = self._candidate_embeddings(state, actions)
        logits = candidates @ self.readout
        shifted = logits - logits.max()
        normalizer = np.exp(shifted).sum()
        loss = np.log(normalizer) - shifted[chosen]
        gradient = np.exp(shifted) / normalizer
        gradient[chosen] -= 1
        updated = self.readout - np.float32(learning_rate) * (gradient @ candidates)
        if not np.isfinite(updated).all() or not np.isfinite(loss):
            raise FloatingPointError("Nonfinite demonstration update")
        self.readout[:] = updated
        return float(loss)

    tokens = staticmethod(_tokens)

    def scores(self, state, actions):
        candidates = self._candidate_embeddings(state, actions)
        logits = candidates @ self.readout + self.readout_bias
        if not np.isfinite(logits).all():
            raise FloatingPointError("Nonfinite policy scores")
        self.forward_calls += 1
        return logits

    def __call__(self, state, actions):
        return int(np.argmax(self.scores(state, actions)))
