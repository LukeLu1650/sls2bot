"""Build and package NecroPilot v0.2.0 for Slay the Spire 2 v0.114.0 (AlphaDefect backend)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAME = Path("C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2")
AD_ROOT = ROOT / "external/AlphaDefect-Public/game-v0.114.0"
AD_PYTHON = AD_ROOT / "python"
AD_CATALOG = AD_ROOT / "engine/catalog"
DEFAULT_CHECKPOINT = ROOT / "runs/baseline-ad-necro-v1140-a10-s0/latest.pt"
EXPECTED_BUILD = "v0.114.0"


def check_game(game: Path) -> dict:
    info_path = game / "release_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"release_info.json not found in {game}")
    info = json.loads(info_path.read_bytes())
    version = info.get("version")
    if version != EXPECTED_BUILD:
        raise ValueError(f"Expected {EXPECTED_BUILD}, found {version} ({info.get('commit')})")
    return info


def build_dll(game: Path | None = None) -> Path:
    cmd = [
        "dotnet", "build", str(ROOT / "live/NecroPilot.csproj"),
        "-c", "Release", "--disable-build-servers", "-nologo", "-v:q",
    ]
    if game is not None:
        cmd.append(f"-p:GameDir={game}")
    env = os.environ.copy()
    with TemporaryDirectory(prefix="necropilot-nuget-") as tmp:
        # Keep NuGet restore self-contained without touching user-global NuGet.Config
        nuget_dir = Path(tmp) / "NuGet"
        nuget_dir.mkdir(parents=True, exist_ok=True)
        (nuget_dir / "NuGet.Config").write_text(
            '<?xml version="1.0" encoding="utf-8"?><configuration><packageSources><clear /></packageSources></configuration>',
            encoding="utf-8",
        )
        env["APPDATA"] = tmp
        env["NUGET_PACKAGES"] = str(Path(tmp) / "pkgs")
        subprocess.run(cmd, check=True, env=env)
    dll = ROOT / "live/bin/Release/net9.0/NecroPilot.dll"
    if not dll.exists():
        raise FileNotFoundError(f"Built assembly not found: {dll}")
    return dll


def make_config(checkpoint: Path, *, python_exe: str, device: str, use_ema: bool,
                character_filter: str | None, action_delay_ms: int) -> dict:
    py_paths = [
        str((ROOT / "artifacts/ppo-cuda").resolve()),
        str(AD_PYTHON.resolve()),
    ]
    return {
        "python": str(Path(python_exe).resolve()),
        "pythonpath": py_paths,
        "checkpoint": str(checkpoint.resolve()),
        "device": device,
        "use_ema": use_ema,
        "character_filter": character_filter,
        "action_delay_ms": action_delay_ms,
        "bug_report_dir": str((ROOT / "runs/live-bug-reports").resolve()),
    }


def stage_folder(dest: Path, dll: Path, cfg: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dll, dest / "NecroPilot.dll")
    shutil.copy2(ROOT / "live/NecroPilot.json", dest / "NecroPilot.json")
    shutil.copy2(ROOT / "scripts/live_policy.py", dest / "live_policy.py")
    shutil.copy2(AD_CATALOG / "vocab.txt", dest / "vocab.txt")
    shutil.copy2(AD_CATALOG / "aliases.txt", dest / "aliases.txt")
    shutil.copy2(AD_CATALOG / "no_downside_relics.txt", dest / "no_downside_relics.txt")
    (dest / "pilot.cfg").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def package(game: Path, checkpoint: Path, *, out_zip: Path | None, stage_dir: Path | None,
            python_exe: str, device: str, use_ema: bool, character_filter: str | None,
            action_delay_ms: int, skip_game_check: bool = False) -> None:
    if not skip_game_check:
        info = check_game(game)
        print(f"Verified Steam game: {info['version']} ({info['commit']})")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    dll = build_dll(None if skip_game_check else game)
    cfg = make_config(
        checkpoint,
        python_exe=python_exe,
        device=device,
        use_ema=use_ema,
        character_filter=character_filter,
        action_delay_ms=action_delay_ms,
    )

    if stage_dir is not None:
        stage_folder(stage_dir, dll, cfg)
        print(f"Staged NecroPilot folder: {stage_dir.resolve()}")

    if out_zip is not None:
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="necropilot-zip-") as tmp:
            root = Path(tmp) / "NecroPilot"
            stage_folder(root, dll, cfg)
            with ZipFile(out_zip, "w", ZIP_DEFLATED) as archive:
                for p in sorted(root.rglob("*")):
                    if p.is_file():
                        archive.write(p, "NecroPilot/" + p.relative_to(root).as_posix())
        print(f"Packaged ZIP: {out_zip.resolve()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, default=DEFAULT_GAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, help="Output ZIP archive path")
    parser.add_argument("--stage", type=Path, help="Stage unpacked NecroPilot folder here")
    parser.add_argument("--python", default=sys.executable, help="Python executable for live_policy.py")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--no-ema", action="store_true", help="Use raw weights instead of EMA weights")
    parser.add_argument("--character-filter", default=None, help="Optional character entry restriction (e.g. NECROBINDER)")
    parser.add_argument("--action-delay-ms", type=int, default=150, help="Minimum delay between auto-pilot actions (ms)")
    parser.add_argument("--skip-game-check", action="store_true", help="Compile against artifacts/alphadefect/game/v0.114.0")
    args = parser.parse_args(argv)
    if args.out is None and args.stage is None:
        args.out = ROOT / "artifacts/NecroPilot-v0.2.0-v1140.zip"
    package(
        args.game.resolve(),
        args.checkpoint.resolve(),
        out_zip=args.out.resolve() if args.out else None,
        stage_dir=args.stage.resolve() if args.stage else None,
        python_exe=args.python,
        device=args.device,
        use_ema=not args.no_ema,
        character_filter=args.character_filter,
        action_delay_ms=args.action_delay_ms,
        skip_game_check=args.skip_game_check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
