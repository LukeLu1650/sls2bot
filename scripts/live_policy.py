"""Pipe-based v0.114.0 AlphaDefect inference worker for NecroPilot.

Protocol (UTF-8 JSON lines on stdin/stdout; diagnostics go strictly to stderr):
  Startup -> stdout:
    {"type": "ready", "checkpoint": "...", "version": 1902, "device": "cpu", "ema": true, "character": "NECROBINDER"}
  Request <- stdin:
    {"type": "infer", "frame": "<base64 raw Observation.Serialize() bytes>"}
  Response -> stdout:
    {"type": "infer", "index": 0, "value": 1.23, "ms": 8.4,
     "top": [{"index": 0, "prob": 0.62, "logit": 1.8}, ...]}
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def _resolve_device(spec: str, torch_mod) -> str:
    spec = (spec or "cpu").strip().lower()
    if spec == "auto":
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    if spec.startswith("cuda") and not torch_mod.cuda.is_available():
        print("[live_policy] CUDA requested but unavailable; falling back to cpu", file=sys.stderr, flush=True)
        return "cpu"
    return spec


def _load_model(ckpt_path: Path, device_spec: str, use_ema: bool):
    import torch
    from sts2rl.infer import InferEngine
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = _resolve_device(device_spec, torch)
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = blob.get("config") or {}
    model_cfg = cfg_dict.get("model") or {}
    net = PolicyValueNet(ModelConfig(**model_cfg)).eval()

    ema_sd = blob.get("ema_state") or blob.get("ema")
    actually_ema = bool(use_ema and isinstance(ema_sd, dict) and ema_sd)
    state = {**blob["state"], **ema_sd} if actually_ema else blob["state"]
    load_state(net, state, blob.get("content_names"))
    net = net.to(device)

    threads = 4 if device == "cpu" else 0
    engine = InferEngine(net, device=device, precision="fp32", threads=threads)
    meta = {
        "checkpoint": f"{ckpt_path.parent.name}/{ckpt_path.name}",
        "version": int(blob.get("version", 0)),
        "device": device,
        "ema": actually_ema,
        "character": str(cfg_dict.get("character") or ""),
    }
    return engine, meta


def _infer_one(engine, raw_frame: bytes) -> dict:
    from sts2env.protocol import parse
    from sts2rl.features import Ragged
    from sts2rl.greedy import merged_argmax

    t0 = perf_counter()
    obs = parse(raw_frame)
    n = obs.num_actions
    if n <= 0:
        raise ValueError("Observation contains zero candidate actions")

    # mode=1 marks a full run so macro decisions route through the macro policy/value heads.
    ragged = Ragged.from_obs([obs], mode=np.ones(1, dtype=np.int64))
    logits_arr, values_arr = engine.run(ragged)
    logits = logits_arr[0, :n].astype(np.float64)
    value = float(values_arr[0])

    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    chosen = int(merged_argmax(obs, probs))

    order = list(np.argsort(-probs, kind="stable")[:3])
    if chosen not in order:
        order = [chosen, *order[:2]]
    else:
        order = [chosen] + [i for i in order if i != chosen]

    top = [
        {"index": int(i), "prob": float(probs[i]), "logit": float(logits[i])}
        for i in order
    ]
    ms = (perf_counter() - t0) * 1000.0
    return {"type": "infer", "index": chosen, "value": value, "ms": ms, "top": top}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to pilot.cfg")
    args = parser.parse_args(argv)

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        for extra in cfg.get("pythonpath") or []:
            if extra and extra not in sys.path:
                sys.path.insert(0, extra)

        ckpt = Path(cfg["checkpoint"])
        if not ckpt.is_absolute():
            ckpt = (args.config.parent / ckpt).resolve()

        engine, meta = _load_model(ckpt, cfg.get("device", "cpu"), bool(cfg.get("use_ema", True)))
        _emit({"type": "ready", **meta})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("type") != "infer":
                _emit({"type": "error", "error": f"Unknown request type: {req.get('type')}"})
                continue
            raw = base64.b64decode(req["frame"])
            _emit(_infer_one(engine, raw))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
