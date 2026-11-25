"""A trainable last Transformer block with an action head and a state-value head."""
from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from necro_policy import _bucket
from strategic_features import features
from strategy_updates import update_config
from train_combat import combat_inputs


STRATEGY_TEMPERATURE = update_config("warmup", 12, 12).temperature


@dataclass
class Decision:
    tokens: torch.Tensor
    adjustments: torch.Tensor
    combat: bool
    choice: int
    log_prob: float
    value: float
    room: tuple


class PPOPolicy(nn.Module):
    def __init__(self, base, weights, device):
        super().__init__()
        self.base = base
        self.device = torch.device(device)
        self.width, self.heads = base.width, base.heads
        self.register_buffer("embedding", torch.from_numpy(base.emb.copy()))
        self.register_buffer("base_readout", torch.from_numpy(base.readout.copy()))
        self.register_buffer("read_scale", torch.from_numpy(base.read_scale.copy()))
        self.register_buffer("read_bias", torch.from_numpy(base.read_bias.copy()))
        self.blocks = nn.ModuleList()
        for index, block in enumerate(base.blocks):
            self.blocks.append(nn.ParameterList(nn.Parameter(torch.from_numpy(p.copy()),
                                      requires_grad=index == len(base.blocks) - 1) for p in block))
        self.combat = nn.Parameter(torch.from_numpy(weights["combat"].copy()))
        self.strategy = nn.Parameter(torch.from_numpy(weights["strategy"].copy()))
        self.value = nn.Linear(base.width, 1)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        self.to(device)

    def block(self, x, params, mask=None):
        s1, b1, qkv, qb, proj, pb, s2, b2, ff1, fb1, ff2, fb2 = params
        batch, tokens, width = x.shape
        hidden = F.layer_norm(x, (width,), s1, b1)
        heads = (hidden @ qkv + qb).reshape(batch, tokens, 3, self.heads, width // self.heads)
        q, k, v = heads.permute(2, 0, 3, 1, 4).unbind()
        attention = (q @ k.transpose(-1, -2)) / sqrt(width // self.heads)
        if mask is not None:
            attention = attention.masked_fill(~mask[:, None, None, :], -torch.inf)
        attended = (attention.softmax(-1) @ v).transpose(1, 2).reshape(batch, tokens, width)
        x = x + attended @ proj + pb
        hidden = F.layer_norm(x, (width,), s2, b2)
        return x + F.relu(hidden @ ff1 + fb1) @ ff2 + fb2

    @torch.no_grad()
    def encode(self, state, actions):
        tokens = self.base.tokens(state, actions)
        indices, rows, values = [], [], []
        for row, token in enumerate(tokens):
            for label, value in token:
                indices.append(_bucket(label))
                rows.append(row)
                values.append(value)
        x = self.embedding.new_zeros((len(tokens), self.width))
        index = torch.tensor(indices, device=self.device)
        weighted = self.embedding[index] * self.embedding.new_tensor(values)[:, None]
        x.index_add_(0, torch.tensor(rows, device=self.device), weighted)
        x = x.unsqueeze(0)
        for block in list(self.blocks)[:-1]:
            x = self.block(x, block)
        return x[0]

    def forward(self, tokens, mask=None):
        hidden = self.block(tokens, self.blocks[-1], mask)
        return F.layer_norm(hidden, (self.width,), self.read_scale, self.read_bias)

    @torch.no_grad()
    def decide(self, state, actions, combat, reason, rng):
        tokens = self.encode(state, actions)
        hidden = self(tokens.unsqueeze(0))[0]
        candidates = hidden[-len(actions):]
        if combat:
            _, adjustments = combat_inputs(self.base, state, actions, embeddings=candidates.cpu().numpy())
            adjustments = adjustments[:, self.width:]
        else:
            adjustments = features(state, actions, "boss_deck", reason)
        adjustments = torch.from_numpy(adjustments.copy())
        logits = self.scores(candidates, adjustments.to(self.device), combat)
        if rng is None:
            return int(logits.argmax()), None
        log_probs = self.log_distribution(logits, combat)
        probabilities = log_probs.exp().cpu().numpy().astype(np.float64)
        choice = int(rng.choice(len(actions), p=probabilities / probabilities.sum()))
        context = state["context"]
        sample = Decision(tokens.cpu(), adjustments, combat, choice, float(log_probs[choice]),
                          float(self.value(hidden[0])), (context["act"], context["floor"]))
        return choice, sample

    def scores(self, candidates, adjustments, combat):
        readout = self.base_readout + self.combat[:self.width] if combat else self.base_readout
        weights = self.combat[self.width:] if combat else self.strategy
        return candidates @ readout + adjustments @ weights

    @staticmethod
    def log_distribution(logits, combat):
        return (logits / (.7 if combat else STRATEGY_TEMPERATURE)).log_softmax(-1)

    def update(self, samples, optimizer, rng, *, epochs=3, minibatch=24):
        """Use fixed behavior log probabilities for all epochs of one rollout batch."""
        if not samples:
            raise ValueError("PPO received no confirmed decisions")
        advantages = np.array([item[1] for item in samples], dtype=np.float32)
        advantages = (advantages - advantages.mean()) / max(advantages.std(), .01)
        losses, divergences = [], []
        update_count = 0
        for _ in range(epochs):
            order = rng.permutation(len(samples))
            for start in range(0, len(order), minibatch):
                indices = order[start:start + minibatch]
                decisions, _, targets = zip(*(samples[i] for i in indices), strict=True)
                padded = nn.utils.rnn.pad_sequence([d.tokens for d in decisions], batch_first=True).to(self.device)
                lengths = torch.tensor([len(d.tokens) for d in decisions], device=self.device)
                mask = torch.arange(padded.shape[1], device=self.device)[None, :] < lengths[:, None]
                hidden = self(padded, mask)
                log_probs, entropies = [], []
                for row, sample in zip(hidden, decisions, strict=True):
                    end = len(sample.tokens)
                    candidates = row[end - len(sample.adjustments):end]
                    logits = self.scores(candidates, sample.adjustments.to(self.device), sample.combat)
                    log_prob = self.log_distribution(logits, sample.combat)
                    log_probs.append(log_prob[sample.choice])
                    entropies.append(-(log_prob.exp() * log_prob).sum())
                old = torch.tensor([d.log_prob for d in decisions], device=self.device)
                log_ratio = torch.stack(log_probs) - old
                ratio = log_ratio.exp()
                advantage = torch.tensor(advantages[indices], device=self.device)
                targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
                values = self.value(hidden[:, 0]).squeeze(-1)
                actor = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
                loss = actor + .5 * F.mse_loss(values, targets) - .01 * torch.stack(entropies).mean()
                divergence = float((ratio - 1 - log_ratio).mean().detach())
                divergences.append(divergence)
                if divergence > .03:
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), .5, error_if_nonfinite=True)
                optimizer.step()
                update_count += 1
                losses.append(float(loss.detach()))
        return dict(decisions=len(samples), updates=update_count,
                    loss=float(np.mean(losses)) if losses else None, mean_kl=float(np.mean(divergences)))
