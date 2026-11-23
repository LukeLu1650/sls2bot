"""Necrobinder episode runner for the pinned sts2-cli JSON-lines host.

The policy sees current decision fields, legal-candidate commands, and public
combat information. The seed, combat draw-pile listing, and full-map query are
omitted. Explicit card prompts retain the ordering and indices from the host.
Candidates use the host's legality flags where available. The host does not
expose an exhaustive legal-action API: card picks use combinations in displayed
order (up to 256 choices), potions are usable only in combat here, and merchant
potion purchases use the actual game-reported free slots. The
subset omits potion discards. The host automatically collects non-card combat
and treasure rewards.
"""

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from itertools import combinations, islice
import json
import math
from pathlib import Path
from queue import Empty, Queue
import subprocess
from tempfile import TemporaryFile
from threading import Thread
from time import monotonic


_MAX_LINE_BYTES = 4 * 1024 * 1024
_MAX_CARD_SELECTIONS = 256
_MODES = {"standard": "Standard", "blitz": "Blitz"}
_PUBLIC_FIELDS = {
    "combat_play": (
        "round", "energy", "max_energy", "hand", "enemies", "player_powers",
        "draw_pile_count", "discard_pile_count", "discard_pile", "osty", "orbs",
        "orb_slots", "stars",
    ),
    "map_select": ("choices", "act", "act_name", "floor"),
    "event_choice": ("event_name", "description", "options"),
    "rest_site": ("options",),
    "card_reward": ("cards", "can_skip", "from_event", "gold_earned"),
    "card_select": ("cards", "min_select", "max_select"),
    "bundle_select": ("bundles",),
    "shop": (
        "is_blitz_shop", "blitz_choices_remaining", "blitz_entries", "cards",
        "relics", "potions", "card_removal_cost",
    ),
}
_PUBLIC_CONTEXT = ("act", "act_name", "floor", "room_type", "game_mode", "ascension", "boss")
_PUBLIC_PLAYER = ("name", "hp", "max_hp", "block", "gold", "relics", "potions", "deck_size", "deck",
                  "potion_capacity", "has_open_potion_slots")
_PASSIVE_POTIONS = {"Fairy in a Bottle", "Deprecated Potion"}


def _pick_fields(data, fields):
    return {key: data[key] for key in fields if key in data}


def public_observation(state: dict) -> dict:
    """Project a pinned-protocol response into policy-visible decision state."""
    decision = state["decision"]
    return deepcopy({
        "type": "decision",
        "decision": decision,
        "context": _pick_fields(state.get("context", {}), _PUBLIC_CONTEXT),
        "player": _pick_fields(state.get("player", {}), _PUBLIC_PLAYER),
        **_pick_fields(state, _PUBLIC_FIELDS.get(decision, ())),
    })


def _action(name, **args):
    command = {"cmd": "action", "action": name}
    if args:
        command["args"] = args
    return command


def _targeted_actions(name, index_key, item, targets):
    args = {index_key: item["index"]}
    if item["target_type"] == "AnyEnemy":
        return [_action(name, **args, target_index=index) for index in targets]
    return [_action(name, **args)]


def _shop_actions(state):
    actions = []
    if state.get("is_blitz_shop"):
        if state["blitz_choices_remaining"] > 0:
            actions.extend(
                _action("buy_blitz_entry", index=entry["index"])
                for entry in state["blitz_entries"]
                if entry["available"] and entry["is_stocked"]
            )
    else:
        player = state["player"]
        gold = player["gold"]
        categories = [("cards", "card"), ("relics", "relic")]
        capacity = player.get("potion_capacity")
        free_slots = player.get("has_open_potion_slots")
        if free_slots is None and capacity is not None:
            free_slots = len(player.get("potions") or []) < capacity
        if free_slots:
            categories.append(("potions", "potion"))
        for category, item in categories:
            actions.extend(
                _action(f"buy_{item}", **{f"{item}_index": entry["index"]})
                for entry in state[category]
                if entry["is_stocked"] and entry["cost"] <= gold
            )
        cost = state.get("card_removal_cost")
        if cost is not None and cost <= gold:
            actions.append(_action("remove_card"))
    actions.append(_action("leave_room"))
    return actions


def _card_selections(state):
    indices = [card["index"] for card in state["cards"]]
    lo, hi = state["min_select"], state["max_select"]
    picks = (
        pick
        for count in range(max(1, lo), min(hi, len(indices)) + 1)
        for pick in combinations(indices, count)
    )
    actions = [_action("skip_select")] if lo == 0 else []
    actions.extend(
        _action("select_cards", indices=",".join(map(str, pick)))
        for pick in islice(picks, _MAX_CARD_SELECTIONS - len(actions))
    )
    return actions


def legal_actions(state: dict) -> list[dict]:
    """Enumerate the supported subset of actions from the current decision."""
    decision = state["decision"]
    if decision == "combat_play":
        targets = [enemy["index"] for enemy in state["enemies"] if enemy["hp"] > 0]
        actions = []
        for card in state["hand"]:
            if card["can_play"]:
                actions.extend(_targeted_actions("play_card", "card_index", card, targets))
        for potion in state["player"].get("potions") or []:
            if potion["name"] not in _PASSIVE_POTIONS:
                actions.extend(_targeted_actions("use_potion", "potion_index", potion, targets))
        return [*actions, _action("end_turn")]
    if decision == "map_select":
        return [_action("select_map_node", **_pick_fields(node, ("col", "row")))
                for node in state["choices"]]
    if decision == "event_choice":
        return [_action("choose_option", option_index=opt["index"])
                for opt in state["options"] if not opt["is_locked"]]
    if decision == "rest_site":
        return [_action("choose_option", option_index=opt["index"])
                for opt in state["options"] if opt["is_enabled"]]
    if decision == "card_reward":
        actions = [_action("select_card_reward", card_index=card["index"])
                   for card in state["cards"]]
        if state["can_skip"]:
            actions.append(_action("skip_card_reward"))
        return actions
    if decision == "card_select":
        return _card_selections(state)
    if decision == "bundle_select":
        return [_action("select_bundle", bundle_index=bundle["index"]) for bundle in state["bundles"]]
    if decision == "shop":
        return _shop_actions(state)
    return []


class _EpisodeFailure(Exception):
    def __init__(self, status, message, code=None):
        super().__init__(message)
        self.status = status
        self.code = code


class _Host:
    def __init__(self, path, stderr):
        path = path.resolve(strict=True)
        self.process = subprocess.Popen(
            ["dotnet", str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        self.requests = Queue()
        self.responses = Queue()
        self.waiting = False
        self.worker = Thread(target=self._serve, daemon=True)
        self.worker.start()

    def _serve(self):
        try:
            while (request := self.requests.get()) is not False:
                if request is not None:
                    self.process.stdin.write(json.dumps(request).encode("ascii") + b"\n")
                    self.process.stdin.flush()
                self.responses.put(self._read())
        except Exception as exc:
            self.responses.put(exc)

    def _read(self):
        while True:
            line = self.process.stdout.readline(_MAX_LINE_BYTES + 1)
            if not line:
                raise EOFError(f"Host stdout closed (exit code {self.process.poll()})")
            if len(line) > _MAX_LINE_BYTES:
                raise ValueError("Host stdout line exceeded size limit")
            # The game can log plain-text diagnostics to stdout between JSON lines.
            line = line.lstrip(b"\xef\xbb\xbf \r\t")
            if line.startswith(b"{"):
                data = json.loads(line)
                if not isinstance(data.get("type"), str):
                    raise ValueError("Host response is missing its type")
                return data

    def exchange(self, command, deadline):
        self.waiting = True
        self.requests.put(command)
        response = self.responses.get(timeout=max(0, deadline - monotonic()))
        self.waiting = False
        if isinstance(response, Exception):
            raise response
        return response

    def close(self, deadline):
        try:
            if (not self.waiting and self.worker.is_alive()
                    and self.process.poll() is None and monotonic() < deadline):
                response = self.exchange({"cmd": "quit"}, deadline)
                if response["type"] == "quit_result":
                    self.process.wait(timeout=max(0, deadline - monotonic()))
        except (Empty, EOFError, OSError, ValueError, subprocess.TimeoutExpired):
            pass
        finally:
            self.requests.put(False)
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.worker.join()
            self.process.stdin.close()
            self.process.stdout.close()


def _policy_action(policy, observation, actions, deadline):
    result = Queue()

    def invoke():
        try:
            result.put((True, policy(observation, deepcopy(actions))))
        except BaseException as exc:
            result.put((False, exc))

    Thread(target=invoke, daemon=True).start()
    try:
        ok, choice = result.get(timeout=max(0, deadline - monotonic()))
    except Empty:
        raise _EpisodeFailure("timeout", "Policy exceeded the step or episode deadline", "policy_timeout") from None
    if not ok:
        raise _EpisodeFailure("policy_error", f"{type(choice).__name__}: {choice}", "policy_exception")
    if type(choice) is not int or not 0 <= choice < len(actions):
        raise _EpisodeFailure("policy_error", "Policy must return an in-range integer index", "invalid_policy_index")
    return actions[choice]


def _expect(response, expected):
    if response["type"] == "error":
        raise _EpisodeFailure("engine_error", response.get("message", "Engine error"), response.get("error_code"))
    if response["type"] != expected:
        raise _EpisodeFailure("engine_error", f"Expected {expected}, got {response['type']}", "unexpected_response")
    return response


def run_episode(
    host: Path,
    seed: str,
    mode: str,
    ascension: int,
    policy: Callable[[dict, list[dict]], int],
    *,
    step_timeout: float = 20.0,
    episode_timeout: float = 300.0,
    first_combat_only: bool = False,
    max_steps: int = 2000,
) -> dict:
    """Run one solo episode with a new dotnet process, then reap the process.

    ``host`` is the compiled Sts2Headless DLL. ``policy(state, actions)``
    returns a zero-based action index. Policy state should be held in the
    callable or a closure. Timed-out Python callbacks cannot be interrupted;
    they run in a daemon thread and restricted callbacks should be cooperative.

    Results distinguish victory, zero-HP defeat, engine or policy errors,
    unsupported decisions, and exhausted budgets. ``steps`` and ``actions``
    count commands sent or attempted; ``decisions`` counts responses seen.
    Modes are case-insensitive. Ascension clamping is reported from the host.
    """
    mode = mode.lower()
    if mode not in _MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected Standard or Blitz")
    if not seed or not isinstance(seed, str):
        raise ValueError("Seed must be a nonempty string")
    if type(ascension) is not int or not 0 <= ascension < 2**31:
        raise ValueError("Ascension must be a nonnegative 32-bit integer")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if any(not math.isfinite(x) or x <= 0 for x in (step_timeout, episode_timeout)):
        raise ValueError("Timeouts must be positive and finite")

    start = monotonic()
    deadline = start + episode_timeout
    result = {
        "seed": seed, "mode": mode, "requested_ascension": ascension, "ascension": None,
        "status": None, "victory": None, "hp": None, "max_hp": None, "act": None,
        "floor": None, "steps": 0, "error": None, "error_code": None,
    }
    decisions, actions_taken = Counter(), Counter()
    first_combat_started = False
    bridge = None
    with TemporaryFile() as stderr:
        try:
            bridge = _Host(host, stderr)
            _expect(bridge.exchange(None, min(deadline, monotonic() + step_timeout)), "ready")
            state = bridge.exchange({
                "cmd": "start_run", "character": "Necrobinder", "seed": seed, "lang": "en",
                "game_mode": _MODES[mode], "ascension": ascension,
            }, min(deadline, monotonic() + step_timeout))
            while True:
                decision = _expect(state, "decision")["decision"]
                decisions[decision] += 1
                context, player = state.get("context", {}), state.get("player", {})
                result.update(_pick_fields(context, ("act", "floor", "ascension")))
                result.update(_pick_fields(state, ("act", "floor")))
                result.update(_pick_fields(player, ("hp", "max_hp")))
                if decision == "game_over":
                    victory = state.get("victory")
                    hp = player.get("hp")
                    if victory is True and type(hp) is int and hp > 0:
                        result.update(status="victory", victory=True)
                    elif victory is False and type(hp) is int and hp == 0:
                        result.update(status="defeat", victory=False)
                    else:
                        raise _EpisodeFailure("engine_error", "Inconsistent terminal victory or HP", "invalid_terminal")
                    break
                if player.get("hp") == 0:
                    raise _EpisodeFailure("engine_error", "Zero HP outside game_over", "invalid_terminal")
                if first_combat_only and first_combat_started and decision not in {"combat_play", "card_select"}:
                    result.update(status="combat_cleared", victory=None)
                    break
                if result["steps"] == max_steps:
                    raise _EpisodeFailure("step_limit", "Action step limit reached", "max_steps")
                step_deadline = min(deadline, monotonic() + step_timeout)
                observation = public_observation(state)
                choices = legal_actions(observation)
                if not choices:
                    raise _EpisodeFailure("unsupported_decision", f"No supported actions for {decision}", decision)
                command = _policy_action(policy, observation, choices, step_deadline)
                first_combat_started |= decision == "combat_play"
                if monotonic() >= step_deadline:
                    raise _EpisodeFailure("timeout", "Step or episode deadline reached", "step_timeout")
                result["steps"] += 1
                actions_taken[command["action"]] += 1
                state = bridge.exchange(command, step_deadline)
        except _EpisodeFailure as exc:
            result.update(status=exc.status, error=str(exc), error_code=exc.code)
        except Empty:
            result.update(status="timeout", error="Host did not respond before the deadline", error_code="host_timeout")
        except (EOFError, OSError, ValueError, KeyError, TypeError) as exc:
            result.update(status="engine_error", error=f"{type(exc).__name__}: {exc}", error_code="host_protocol")
        finally:
            if bridge is not None:
                bridge.close(min(deadline, monotonic() + 1.0))
        if result["status"] in {"engine_error", "timeout", "unsupported_decision"}:
            stderr.seek(max(0, stderr.seek(0, 2) - 4096))
            result["stderr_tail"] = stderr.read().decode("utf-8", errors="replace")
    result.update(decisions=dict(decisions), actions=dict(actions_taken), elapsed_seconds=monotonic() - start)
    return result
