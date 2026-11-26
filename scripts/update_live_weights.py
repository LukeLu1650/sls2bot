"""Hot-swap the checkpoint or inference device in the installed NecroPilot pilot.cfg."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from package_live import DEFAULT_GAME


def update(checkpoint: Path, game: Path, *, device: str | None, use_ema: bool | None) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    cfg_path = game / "mods" / "NecroPilot" / "pilot.cfg"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Installed NecroPilot pilot.cfg not found: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["checkpoint"] = str(checkpoint.resolve())
    if device is not None:
        cfg["device"] = device
    if use_ema is not None:
        cfg["use_ema"] = use_ema
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {cfg_path}:")
    print(f"  checkpoint = {cfg['checkpoint']}")
    print(f"  device     = {cfg.get('device')}")
    print(f"  use_ema    = {cfg.get('use_ema')}")
    print("Toggle F7/F6 OFF then ON in-game to reload the new checkpoint.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to AlphaDefect .pt checkpoint (e.g. latest.pt or milestones/v001800.pt)")
    parser.add_argument("--game", type=Path, default=DEFAULT_GAME, help="Slay the Spire 2 Steam root")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    ema = parser.add_mutually_exclusive_group()
    ema.add_argument("--ema", dest="use_ema", action="store_true", default=None)
    ema.add_argument("--no-ema", dest="use_ema", action="store_false")
    args = parser.parse_args(argv)
    update(args.checkpoint.resolve(), args.game.resolve(), device=args.device, use_ema=args.use_ema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
