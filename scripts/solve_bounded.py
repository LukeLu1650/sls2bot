"""Run one pinned CombatSolver offline search with Windows process limits."""
import argparse
import ctypes as ct
from ctypes import wintypes as win
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "external/CombatSolver"
REVISION = "d72c52a0252248ec12ab027e335af1c92a48ccb5"
HARNESS = SOLVER / ".local/tool-build/OfflineSearchHarness/bin/Release/net9.0/OfflineSearchHarness.dll"


class BasicJobLimits(ct.Structure):
    _fields_ = [("process_time", ct.c_int64), ("job_time", ct.c_int64), ("flags", win.DWORD),
                ("working_set_min", ct.c_size_t), ("working_set_max", ct.c_size_t),
                ("process_count", win.DWORD), ("affinity", ct.c_size_t),
                ("priority", win.DWORD), ("scheduling", win.DWORD)]


class ExtendedJobLimits(ct.Structure):
    _fields_ = [("basic", BasicJobLimits), ("io", ct.c_uint64 * 6),
                ("process_memory", ct.c_size_t), ("job_memory", ct.c_size_t),
                ("peak_process_memory", ct.c_size_t), ("peak_job_memory", ct.c_size_t)]


def run_limited(command, output, timeout, memory_mib):
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ct.c_void_p, win.LPCWSTR]
    kernel.CreateJobObjectW.restype = win.HANDLE
    kernel.SetInformationJobObject.argtypes = [win.HANDLE, win.DWORD, ct.c_void_p, win.DWORD]
    kernel.QueryInformationJobObject.argtypes = [win.HANDLE, win.DWORD, ct.c_void_p, win.DWORD, ct.c_void_p]
    kernel.OpenProcess.argtypes = [win.DWORD, win.BOOL, win.DWORD]
    kernel.OpenProcess.restype = win.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [win.HANDLE, win.HANDLE]
    kernel.CloseHandle.argtypes = [win.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ct.WinError(ct.get_last_error())
    limits = ExtendedJobLimits()
    # Commit limit for the entire job and terminate any child when the job closes.
    limits.basic.flags = 0x200 | 0x2000
    limits.job_memory = memory_mib * 1024**2
    job_info_class = 9
    process = None
    started = time.monotonic()
    result = {"status": "launch_error"}
    try:
        if not kernel.SetInformationJobObject(job, job_info_class, ct.byref(limits), ct.sizeof(limits)):
            raise ct.WinError(ct.get_last_error())
        with output.open("wb") as log:
            process = subprocess.Popen(command, cwd=SOLVER, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
                                       env={**os.environ, "DOTNET_PROCESSOR_COUNT": "1", "DOTNET_gcServer": "0"})
            handle = kernel.OpenProcess(0x100 | 0x1, False, process.pid)
            if not handle:
                raise ct.WinError(ct.get_last_error())
            try:
                if not kernel.AssignProcessToJobObject(job, handle):
                    raise ct.WinError(ct.get_last_error())
            finally:
                kernel.CloseHandle(handle)
            try:
                code = process.wait(timeout=max(.01, timeout - (time.monotonic() - started)))
                result = dict(status="completed" if code == 0 else "solver_failed", exit_code=code)
            except subprocess.TimeoutExpired:
                result = dict(status="process_timeout")
            if kernel.QueryInformationJobObject(job, job_info_class, ct.byref(limits), ct.sizeof(limits), None):
                result["peak_commit_bytes"] = limits.peak_job_memory
    except OSError as error:
        result["error"] = str(error)
    finally:
        kernel.CloseHandle(job)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    input_files = parser.add_mutually_exclusive_group()
    input_files.add_argument("--request", type=Path, help="An OfflineSearchHarness request JSON")
    input_files.add_argument("--generated-scenario", type=Path, help="A CombatSolver generated scenario JSON")
    parser.add_argument("--seed", default="421AA787S1RU")
    parser.add_argument("--encounter", default="CORPSE_SLUGS_WEAK")
    parser.add_argument("--milliseconds", type=int, default=1000)
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--beam", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=15, help="Whole process deadline, in seconds")
    parser.add_argument("--memory-mib", type=int, default=1024, help="Whole-process commit limit")
    args = parser.parse_args()
    if os.name != "nt" or min(args.milliseconds, args.nodes, args.beam, args.timeout, args.memory_mib) < 1:
        parser.error("Run on Windows and use positive limits")
    if not HARNESS.is_file():
        parser.error(f"Build the pinned OfflineSearchHarness first: {HARNESS}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOLVER, text=True).strip()
    if revision != REVISION:
        parser.error(f"Expected CombatSolver source revision {REVISION}")

    out = args.out.resolve()
    if args.request:
        scenario = ["--request", str(args.request.resolve(strict=True))]
    elif args.generated_scenario:
        generated = args.generated_scenario.resolve(strict=True)
        request = out / "request.json"
        scenario = ["--request", str(request)]
    else:
        scenario = ["--character", "NECROBINDER", "--encounter", args.encounter, "--seed", args.seed]
    out.mkdir(parents=True, exist_ok=False)
    if args.generated_scenario:
        request.write_text(json.dumps({"generatedScenarioPath": str(generated)}), encoding="utf-8")
    command = ["dotnet", str(HARNESS), *scenario, "--search-mode", "Evaluate", "--dop", "1",
               "--budget-ms", str(args.milliseconds), "--nodes", str(args.nodes), "--beam", str(args.beam),
               "--out", str(out), "--workspace", str(out / "workspace")]
    status = run_limited(command, out / "solver.log", args.timeout, args.memory_mib)
    status["limits"] = dict(search_ms=args.milliseconds, nodes=args.nodes, beam=args.beam, threads=1,
                            process_seconds=args.timeout, commit_mib=args.memory_mib)
    status["solver_revision"] = revision
    (out / "worker.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status))
    if status["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
