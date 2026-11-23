"""Inspect training failures or record reproducible diagnostics while training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryFile
from time import monotonic
from unittest.mock import patch
from uuid import uuid4

import beta_env
import train_strategy as strategy
from beta_build import CHECKOUT, HOST, MANIFEST, sha, verify_host
from beta_eval import write_json


FAILURES = {"engine_error", "policy_error", "timeout", "unsupported_decision", "step_limit"}
STRATEGY_SCRIPT = Path(strategy.__file__).with_name("train_strategy.py")


def read_json(path):
    return json.loads(path.read_bytes())


def step_context(state):
    context = state.get("context") or {}
    player = state.get("player") or {}
    return {"decision": state.get("decision"), "event": state.get("event_name"),
            "options": [{k: option[k] for k in ("index", "title", "text_key", "option_id", "is_locked") if k in option}
                        for option in state.get("options") or []],
            "act": state.get("act", context.get("act")), "floor": state.get("floor", context.get("floor")),
            "room": context.get("room_type"), "hp": player.get("hp"),
            "round": state.get("round"), "enemies": [e.get("name") for e in state.get("enemies") or []]}


class Trace:
    def __init__(self):
        self.commands = []
        self.state = None
        self.request = None
        self.previous = None
        self.error = None

    def host_class(self):
        original = beta_env._Host
        trace = self

        class ObservedHost(original):
            def exchange(self, command, deadline):
                if command is not None and command.get("cmd") == "quit":
                    return super().exchange(command, deadline)
                if command is not None:
                    trace.commands.append(command)
                    trace.request = command
                    trace.previous = trace.state
                response = super().exchange(command, deadline)
                if response.get("type") == "decision":
                    trace.state = step_context(response)
                if response.get("type") == "error":
                    trace.error = response
                return response

        return ObservedHost

    def detail(self):
        option_index = (self.request or {}).get("args", {}).get("option_index")
        option = next((option for option in (self.previous or {}).get("options", [])
                       if option["index"] == option_index), None)
        return {"request": self.request, "context": self.previous, "chosen_option": option,
                "host_reply": self.error, "commands": self.commands}


def scope_report(folder):
    entries = []
    counts = Counter()

    def collect(rows, source, group, phase, round_number, ascension=None):
        for row in rows:
            counts[row["status"]] += 1
            if row["status"] in FAILURES:
                entries.append({"source": source.name, "group": group, "phase": phase,
                                "round": round_number, "ascension": ascension, "row": row})

    anchor = folder / "anchor.json"
    if anchor.is_file():
        collect(read_json(anchor)["rows"], anchor, "anchor", "eval", 0)
    for path in sorted(folder.glob("round-*.json")):
        report = read_json(path)
        for group, cohort in report["cohorts"].items():
            collect(cohort["train_rows"], path, group, "train", report["round"])
            collect(cohort.get("eval_rows", []), path, group, "eval", report["round"])
    for path in sorted(folder.glob("final-A*.json")):
        report = read_json(path)
        for group, cohort in report["cohorts"].items():
            collect(cohort["rows"], path, group, "final", report["round"], report.get("test_ascension"))
    return entries, counts


def case_key(item):
    row = item["row"]
    return tuple(row.get(key) for key in ("seed", "status", "error", "act", "floor", "steps", "requested_ascension"))


def describe(trace):
    context, option = trace["context"] or {}, trace["chosen_option"] or {}
    stack = (trace["host_reply"] or {}).get("stack_trace", "").splitlines()
    print(f"  {context.get('event') or context.get('decision')} | "
          f"{option.get('title') or option.get('option_id') or trace['request']}")
    if stack:
        print("  " + stack[0].strip())


def replay_legacy(item, checkpoint):
    """Reconstruct early train trajectories, and evaluations with a saved matching model."""
    protocol = checkpoint["protocol"]
    for name, digest in protocol["sources"].items():
        if sha(STRATEGY_SCRIPT.with_name(name)) != digest:
            return {"unavailable": f"Saved source fingerprint differs: {name}"}
    if sha(MANIFEST) != protocol["host"]:
        return {"unavailable": "Saved game-host fingerprint differs"}
    group = item["group"]
    phase = item["phase"]
    round_number = item["round"]
    weights = None
    rng = None
    expected = item["row"]
    seed = expected["seed"]
    if phase == "train":
        if round_number != 1 or "start_from" in protocol:
            return {"unavailable": "Pre-round parameters were not saved. Use record for future runs."}
        weights = strategy.np.zeros(strategy.FEATURE_COUNT, dtype=strategy.np.float32)
        rng = strategy.np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little"))
    elif group != "anchor":
        if round_number != checkpoint["round"]:
            return {"unavailable": "Matching post-round checkpoint is unavailable. Use record for future runs."}
        weights = strategy.np.asarray(checkpoint["weights"][group], dtype=strategy.np.float32)
    policy = strategy.RunPolicy(strategy.load_base(), group, weights, rng)
    trace = Trace()
    with patch.object(beta_env, "_Host", trace.host_class()):
        row = beta_env.run_episode(HOST, seed, "standard", item["ascension"] if item["ascension"] is not None
                                   else expected.get("requested_ascension", protocol["ascension"]),
                                   policy, max_steps=min(protocol["max_steps"], expected["steps"] + 5),
                                   episode_timeout=protocol["episode_timeout"], strategic_observations=True)
    return {"status": row["status"], "same_result": case_key({"row": row}) == case_key(item),
            "trace": trace.detail(), "result": row}


def scan(argv):
    parser = argparse.ArgumentParser(description="Group errors by seed and reproduce a bounded number of cases.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--replay", type=int, default=0, metavar="N")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.replay <= 10:
        parser.error("--replay must be between 0 and 10")
    if not args.folder.is_dir():
        parser.error(f"Run directory does not exist: {args.folder}")
    entries, counts = scope_report(args.folder)
    groups = defaultdict(list)
    for item in entries:
        groups[case_key(item)].append(item)
    recording_files = sorted((args.folder / "diagnostics").glob("trace-*.json"))
    recordings = defaultdict(list)
    standalone = defaultdict(list)
    for file in recording_files:
        recording = read_json(file)
        key = case_key(recording)
        recordings[key].append(file)
        if key not in groups:
            standalone[key].append({"source": file.name, "group": recording["variant"],
                                    "phase": recording["phase"], "row": recording["row"]})
    groups.update(standalone)
    report = {"status_counts": dict(counts), "unique_failure_cases": len(groups),
              "recorded_failure_files": len(recording_files), "trace_only_cases": len(standalone), "cases": []}
    print(f"Completed attempts: {sum(counts.values())} | {dict(counts)}")
    print(f"Distinct cases: {len(groups)} | Recorded traces: {len(recording_files)}")
    if args.replay:
        verify_host()
        os.environ["STS2_LIB"] = str(CHECKOUT / "lib")
        checkpoint = read_json(args.folder / "checkpoint.json")
    for index, members in enumerate(groups.values()):
        item = members[0]
        row = item["row"]
        files = recordings[case_key(item)]
        case = {"seed": row["seed"], "status": row["status"], "error": row["error"], "occurrences": len(members),
                "reports": [{k: v for k, v in member.items() if k != "row"} for member in members],
                "recordings": [str(file) for file in files]}
        print(f"{row['seed']} A{row['requested_ascension']} act={row['act']} floor={row['floor']} "
              f"{row['status']} x{len(members)}: {row['error']}")
        if files:
            recorded = read_json(files[0])
            describe(recorded["trace"])
        if index < args.replay:
            case["replay"] = (replay_commands(read_json(files[0]), HOST) if files else replay_legacy(item, checkpoint))
            result = case["replay"]
            if "unavailable" in result:
                print(f"  {result['unavailable']}")
            else:
                print(f"  Replayed: {result['status']}")
                describe(result["trace"])
        report["cases"].append(case)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)


def replay_commands(recording, host):
    commands = recording["trace"]["commands"]
    trace = Trace()
    deadline = monotonic() + 45
    error = None
    with TemporaryFile() as stderr:
        runner = trace.host_class()(host, stderr)
        try:
            response = runner.exchange(None, deadline)
            for command in commands:
                if response.get("type") == "error":
                    break
                response = runner.exchange(command, deadline)
            status = response.get("decision") or response.get("type")
        except (OSError, ValueError, EOFError, beta_env.Empty) as exc:
            status, error = type(exc).__name__, str(exc)
        finally:
            runner.close(monotonic() + 1)
    return {"status": status, "error": error, "same_host": recording["host"] == sha(MANIFEST),
            "trace": trace.detail()}


def record(argv):
    play = strategy.play
    original = strategy.run_episode

    def observed_play(base, variant, weights, seeds, args, training=False):
        def run(*positional, **keywords):
            trace = Trace()
            with patch.object(beta_env, "_Host", trace.host_class()):
                row = original(*positional, **keywords)
            if row["status"] in FAILURES:
                folder = args.out / "diagnostics"
                folder.mkdir(exist_ok=True)
                target = folder / f"trace-{row['seed']}-{variant}-{uuid4().hex[:12]}.json"
                write_json(target, {"row": row, "variant": variant, "phase": "train" if training else "eval",
                                    "host": sha(MANIFEST), "trace": trace.detail()})
                print(f"  Diagnostics: {target}", flush=True)
                describe(trace.detail())
            return row
        with patch.object(strategy, "run_episode", run):
            return play(base, variant, weights, seeds, args, training=training)

    with patch.object(strategy, "play", observed_play), patch.object(sys, "argv", [str(STRATEGY_SCRIPT), *argv]):
        strategy.main()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        record(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "scan":
        scan(sys.argv[2:])
    else:
        print("Usage: diagnose_strategy.py scan RUN_FOLDER [--replay N] [--out REPORT]\n"
              "       diagnose_strategy.py record [train_strategy.py arguments]")
        sys.exit(0 if len(sys.argv) == 2 and sys.argv[1] in {"--help", "-h"} else 2)


if __name__ == "__main__":
    main()
