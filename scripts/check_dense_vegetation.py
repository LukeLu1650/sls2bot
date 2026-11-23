"""Exercise both known Dense Vegetation Rest failures on the pinned Windows host."""
import os

from beta_build import CHECKOUT, HOST, verify_host
from beta_env import run_episode
from beta_eval import RulePolicy
from train_strategy import RunPolicy, load_base


def check(seed, fallback):
    events = []

    def choose(state, actions):
        if state.get("event_name") == "Dense Vegetation" and len(events) < 2:
            key = "REST" if not events else "FIGHT"
            option = next(option for option in state["options"] if option["text_key"].endswith("." + key))
            events.append((key, state["player"]["hp"]))
            return next(index for index, action in enumerate(actions)
                        if action.get("args", {}).get("option_index") == option["index"])
        if len(events) == 2 and state["decision"] == "combat_play":
            names = [enemy["name"] for enemy in state["enemies"]]
            events.append(("combat", state["player"]["hp"], names))
        return fallback(state, actions)

    run = run_episode(HOST, seed, "standard", 10, choose, max_steps=130,
                      episode_timeout=28, strategic_observations=True)
    if (len(events) != 3 or events[1][1] != events[0][1] + 19
            or events[2][1] != events[1][1] or events[2][2] != ["Wriggler"] * 4
            or run["status"] not in {"victory", "defeat"}):
        raise RuntimeError(f"{seed}: unexpected event progression {events}, result {run}")
    print(f"{seed}: HP {events[0][1]} -> {events[1][1]}, four Wrigglers, {run['status']}")


if __name__ == "__main__":
    verify_host()
    os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
    check("R04CTH7DYALA", RunPolicy(load_base(), "anchor", None))
    check("8668D8L9UBCT", RulePolicy(5))
