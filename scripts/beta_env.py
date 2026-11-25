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
    "crystal_sphere": ("cells", "divinations"),
    "shop": (
        "is_blitz_shop", "blitz_choices_remaining", "blitz_entries", "cards",
        "relics", "potions", "card_removal_cost", "card_removal_stocked",
    ),
}
_PUBLIC_CONTEXT = ("act", "act_name", "floor", "room_type", "game_mode", "ascension", "boss")
_PUBLIC_PLAYER = ("name", "hp", "max_hp", "block", "gold", "relics", "potions", "deck_size", "deck",
                  "potion_capacity", "has_open_potion_slots")
_PASSIVE_POTIONS = {"Fairy in a Bottle", "Deprecated Potion"}
_SELECTION_SCENES = {"card_select", "card_reward", "bundle_select"}
_PROMPT_COMBAT_FIELDS = ("round", "energy", "hand", "enemies", "player_powers", "osty",
                         "draw_pile_count", "discard_pile_count")


def _pick_fields(data, fields):
    return {key: data[key] for key in fields if key in data}


def public_observation(state: dict) -> dict:
    """Project a pinned-protocol response into policy-visible decision state."""
    decision = state["decision"]
    fields = _PUBLIC_FIELDS.get(decision, ())
    if decision in _SELECTION_SCENES:
        fields += ("in_combat",) + _PROMPT_COMBAT_FIELDS
    return deepcopy({
        "type": "decision",
        "decision": decision,
        "context": _pick_fields(state.get("context", {}), _PUBLIC_CONTEXT),
        "player": _pick_fields(state.get("player", {}), _PUBLIC_PLAYER),
        **_pick_fields(state, fields),
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


def _shop_snapshot(state):
    player = state.get("player") or {}
    return (
        player.get("gold"),
        player.get("deck_size"),
        len(player.get("potions") or ()),
        len(player.get("relics") or ()),
        tuple(e.get("is_stocked") for e in (state.get("cards") or ())),
        tuple(e.get("is_stocked") for e in (state.get("relics") or ())),
        tuple(e.get("is_stocked") for e in (state.get("potions") or ())),
        state.get("blitz_choices_remaining"),
    )


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
        if cost is not None and state.get("card_removal_stocked", True) and cost <= gold:
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


def action_cards(state, action):
    """Resolve the displayed cards referenced by a selection action."""
    kind, args = action["action"], action.get("args") or {}
    if kind == "select_bundle":
        return next((bundle["cards"] for bundle in state.get("bundles") or []
                     if bundle["index"] == args.get("bundle_index")), [])
    if kind == "select_cards":
        indices = map(int, args["indices"].split(","))
    elif kind in {"pick_card", "select_card_reward"}:
        indices = [args["card_index"]]
    elif kind == "confirm_selection":
        indices = state.get("picked_indices") or []
    else:
        return []
    cards = {card["index"]: card for card in state.get("cards") or []}
    return [cards[index] for index in indices if index in cards]


def is_combat_decision(state, active_room=None):
    scene = state["decision"]
    if scene == "combat_play":
        return True
    if scene not in _SELECTION_SCENES:
        return False
    if "in_combat" in state:
        return state["in_combat"]
    # Older host builds only report the room and whether a reward was generated.
    context = state["context"]
    return (context.get("act"), context.get("floor")) == active_room and (
        scene != "card_reward" or state.get("from_event", False))


def _combat_trigger(state, action):
    kind, args = action["action"], action.get("args") or {}
    if kind == "play_card":
        items, index = state["hand"], args["card_index"]
    elif kind == "use_potion":
        items, index = state["player"]["potions"], args["potion_index"]
    else:
        return {"action": kind}
    item = next(item for item in items if item["index"] == index)
    return {"action": kind, **_pick_fields(item, ("id", "name"))}


def _choose_card_selection(policy, state, deadline):
    """Mirror F7's individual picks and submit one legal command to the host."""
    picked = []
    indices = [card["index"] for card in state["cards"]]
    hi = min(state["max_select"], len(indices))
    lo = state["min_select"]
    if not 0 <= lo <= hi:
        raise ValueError(f"Invalid card selection bounds: {lo}-{hi} for {len(indices)} cards")
    state = {**state, "max_select": hi}
    while True:
        actions = [_action("pick_card", card_index=index) for index in indices
                   if index not in picked] if len(picked) < hi else []
        if len(picked) >= lo:
            actions.append(_action("confirm_selection"))
        command = _policy_action(policy, {**state, "picked_indices": list(picked)}, actions, deadline)
        if command["action"] == "confirm_selection":
            return _action("select_cards", indices=",".join(map(str, picked))) if picked else _action("skip_select")
        picked.append(command["args"]["card_index"])


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
        # Omit Trial's Abandon-Run confirmation popup option (UI modal, not a playable game action).
        options = [
            opt for opt in state["options"]
            if not opt["is_locked"] and opt.get("text_key") != "TRIAL.pages.REJECT.options.DOUBLE_DOWN"
        ]
        if not options:
            options = [opt for opt in state["options"] if not opt["is_locked"]]
        return [_action("choose_option", option_index=opt["index"]) for opt in options]
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
    if decision == "crystal_sphere":
        return [_action("reveal_crystal_cell", x=cell["x"], y=cell["y"])
                for cell in state["cells"] if cell["hidden"]]
    if decision == "shop":
        return _shop_actions(state)
    return []


class _EpisodeFailure(Exception):
    def __init__(self, status, message, code=None, stack=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.stack = stack


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
        raise _EpisodeFailure("engine_error", response.get("message", "Engine error"),
                              response.get("error_code"), response.get("stack_trace"))
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
    strategic_observations: bool = False,
    sequential_combat_selection: bool = False,
    start_command: dict | None = None,
    observe=None,
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
    combat_room = None
    combat_trigger = None
    current_map = None
    map_act = None
    boss_rooms = set()
    boss_entries = []
    boss_approach = None
    shop_removal_used_rooms = set()
    shop_blocked_actions = {}
    pending_shop_action = None
    bridge = None
    with TemporaryFile() as stderr:
        try:
            bridge = _Host(host, stderr)
            _expect(bridge.exchange(None, min(deadline, monotonic() + step_timeout)), "ready")
            state = bridge.exchange(start_command if start_command is not None else {
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
                if observe is not None:
                    observe(state)
                if strategic_observations and decision == "combat_play" and context.get("room_type") == "Boss":
                    room = (result["act"], result["floor"])
                    if room not in boss_rooms:
                        approach = boss_approach if boss_approach and boss_approach["act"] == room[0] else None
                        boss_entries.append({"act": room[0], "floor": room[1], "hp": result["hp"],
                                             "max_hp": result["max_hp"], "before_travel": approach,
                                             "boss": context.get("boss")})
                        boss_rooms.add(room)
                        boss_approach = None
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
                in_combat = is_combat_decision(state, combat_room)
                if first_combat_only and first_combat_started and not in_combat:
                    result.update(status="combat_cleared", victory=None)
                    break
                if result["steps"] == max_steps:
                    raise _EpisodeFailure("step_limit", "Action step limit reached", "max_steps")
                step_deadline = min(deadline, monotonic() + step_timeout)
                observation = public_observation(state)
                current_room = (result["act"], result["floor"])
                if pending_shop_action is not None:
                    prev_room, prev_cmd_key, prev_snap = pending_shop_action
                    pending_shop_action = None
                    if decision == "shop" and current_room == prev_room and _shop_snapshot(state) == prev_snap:
                        shop_blocked_actions.setdefault(prev_room, set()).add(prev_cmd_key)
                if decision == "shop" and current_room in shop_removal_used_rooms:
                    observation["card_removal_cost"] = None
                    observation["card_removal_stocked"] = False
                if in_combat:
                    room = context.get("act"), context.get("floor")
                    if room != combat_room:
                        combat_trigger = None
                    combat_room = room
                    if decision in _SELECTION_SCENES and combat_trigger is not None:
                        observation["selection_trigger"] = combat_trigger
                else:
                    combat_room = combat_trigger = None
                if strategic_observations and not in_combat:
                    if result["act"] != map_act or decision == "map_select":
                        board = _expect(bridge.exchange({"cmd": "get_map"}, step_deadline), "map")
                        current_map = _pick_fields(board, ("rows", "current_coord"))
                        map_act = result["act"]
                    observation["visible_map"] = current_map
                if in_combat and sequential_combat_selection and decision == "card_select":
                    command = _choose_card_selection(policy, observation, step_deadline)
                else:
                    choices = legal_actions(observation)
                    if decision == "shop":
                        blocked = shop_blocked_actions.get(current_room)
                        if blocked:
                            filtered = [
                                cmd for cmd in choices
                                if cmd["action"] == "leave_room" or json.dumps(cmd, sort_keys=True) not in blocked
                            ]
                            if filtered:
                                choices = filtered
                    if not choices:
                        raise _EpisodeFailure("unsupported_decision", f"No supported actions for {decision}", decision)
                    command = _policy_action(policy, observation, choices, step_deadline)
                if decision == "combat_play":
                    combat_trigger = _combat_trigger(observation, command)
                if decision == "shop" and command["action"] != "leave_room":
                    pending_shop_action = (current_room, json.dumps(command, sort_keys=True), _shop_snapshot(state))
                if command["action"] == "remove_card":
                    shop_removal_used_rooms.add(current_room)
                if strategic_observations and command["action"] == "select_map_node":
                    selected = command["args"]
                    at_boss = any(str(node.get("type")).lower() == "boss" and
                                  all(node.get(key) == selected[key] for key in ("row", "col"))
                                  for node in observation.get("choices", []))
                    boss_approach = (_pick_fields(result, ("act", "hp", "max_hp")) if at_boss else None)
                first_combat_started |= in_combat
                if monotonic() >= step_deadline:
                    raise _EpisodeFailure("timeout", "Step or episode deadline reached", "step_timeout")
                result["steps"] += 1
                actions_taken[command["action"]] += 1
                state = bridge.exchange(command, step_deadline)
        except _EpisodeFailure as exc:
            result.update(status=exc.status, error=str(exc), error_code=exc.code)
            if exc.stack:
                result["stack_trace"] = exc.stack
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
    if strategic_observations:
        result["boss_entries"] = boss_entries
    return result
