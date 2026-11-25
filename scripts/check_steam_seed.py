"""Compare a Steam run's recorded opening with the parity-enabled headless host."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryFile
from time import monotonic

from beta_build import BUILD, CHECKOUT, HOST, STEAM_HOST, verify_host
from beta_env import _Host
from beta_eval import write_json


def setup_command(history, progress):
    if history["build_id"] != BUILD or history["game_mode"] != "standard" or history["modifiers"]:
        raise ValueError(f"Expected a {BUILD} Standard run without modifiers")
    if len(history["players"]) != 1:
        raise ValueError("Expected a single-player Steam run")
    fields = ("seed", "acts", "ascension", "game_mode", "is_custom_game", "modifiers")
    run = {key: history[key] for key in fields}
    run["character"] = history["players"][0]["character"].split(".", 1)[1].lower()
    return {"cmd": "start_steam_run", "run": run, "progress": progress}


def action(name, **args):
    return {"cmd": "action", "action": name, "args": args}


def probe(host, start, neow, node, selection):
    with TemporaryFile() as stderr:
        bridge = _Host(host, stderr)

        def exchange(command):
            result = bridge.exchange(command, monotonic() + 15)
            if result["type"] == "error":
                raise RuntimeError(result)
            return result

        try:
            exchange(None)
            opening = exchange(start)
            options = {row["text_key"].split(".")[-1]: row["index"] for row in opening["options"]}
            state = exchange(action("choose_option", option_index=options[neow]))
            if state["decision"] == "card_select":
                if selection is None and state["min_select"] == 0:
                    state = exchange(action("skip_select"))
                elif selection is not None:
                    card = next(c for c in state["cards"] if c["id"] == selection)
                    state = exchange(action("select_cards", indices=str(card["index"])))
            if state["decision"] != "map_select":
                raise ValueError(f"Opening did not reach the map. Current decision is {state['decision']}.")
            state = exchange(action("select_map_node", col=node[0], row=node[1]))
            return dict(act=state["context"]["act_name"], act_id=state["context"].get("act_id"),
                        encounter=state["context"].get("encounter_id"),
                        enemies=[enemy["name"] for enemy in state.get("enemies", [])],
                        hp=state["player"]["hp"], neow_options=list(options))
        finally:
            bridge.close(monotonic() + 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="Steam history .run file")
    parser.add_argument("--progress", type=Path, help="Progress at run start, defaults to the same profile's progress.save")
    parser.add_argument("--node", nargs=2, type=int, required=True, metavar=("COL", "ROW"),
                        help="First map node from the Steam log (zero-based coordinates)")
    parser.add_argument("--out", type=Path, help="Save a summary without profile contents")
    args = parser.parse_args()
    history = json.loads(args.history.read_bytes())
    profile = args.progress or (args.history.parent.parent / "progress.save")
    progress = json.loads(profile.read_bytes())
    steam_command = setup_command(history, progress)
    rooms = history["map_point_history"][0]
    chosen = rooms[0]["player_stats"][0]
    neow = next(option["TextKey"] for option in chosen["ancient_choice"] if option["was_chosen"])
    selection = next((c["card"]["id"] for c in chosen.get("card_choices", []) if c["was_picked"]), None)
    actual = dict(act_id=history["acts"][0], encounter=rooms[1]["rooms"][0]["model_id"],
                  neow_options=[option["TextKey"] for option in chosen["ancient_choice"]])
    args_dict = steam_command["run"]
    original = dict(cmd="start_run", seed=args_dict["seed"], character=args_dict["character"],
                    ascension=args_dict["ascension"], game_mode="Standard", lang="en")
    verify_host()
    verify_host(steam=True)
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    report = dict(seed=history["seed"], node=args.node, neow=neow, steam_history=actual,
                  legacy=probe(HOST, original, neow, args.node, selection),
                  steam_parity=probe(STEAM_HOST, steam_command, neow, args.node, selection))
    report["matches"] = all(report["steam_parity"][k] == v for k, v in actual.items())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if not report["matches"]:
        raise SystemExit("The patched opening differs from the Steam history.")


if __name__ == "__main__":
    main()
