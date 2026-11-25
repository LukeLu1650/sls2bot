"""Install both F7 weight files and their manifest from a compatible checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile, mkdtemp

from beta_build import ROOT, sha
from package_live import BASE_FILES, DEFAULT_GAME, WORKER_FILES, check_game, checkpoint_payload


def write_file(path, data):
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def update(checkpoint, game):
    check_game(game)
    mod = game / "mods" / "NecroPilot"
    bundled = json.loads((ROOT / "live/NecroPilot.json").read_bytes())
    installed = json.loads((mod / "NecroPilot.json").read_bytes())
    if (installed["id"], installed["version"]) != (bundled["id"], bundled["version"]):
        raise ValueError("Install the current NecroPilot package before updating weights")
    if sha(mod / "readout.npz") != sha(BASE_FILES / "readout.npz") or any(
            sha(mod / name) != sha(ROOT / "scripts" / name) for name in WORKER_FILES):
        raise ValueError("Installed base or worker differs. Install the current NecroPilot package first.")

    payload = checkpoint_payload(checkpoint, full=True)
    if os.name == "nt":
        running = subprocess.check_output([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "if (Get-Process -Name SlayTheSpire2 -ErrorAction SilentlyContinue) { 'running' }",
        ])
        if running.strip() == b"running":
            raise ValueError("Close Slay the Spire 2 before updating F7 weights")

    old = {name: (mod / name).read_bytes() for name in payload}
    if old == payload:
        print("The installed F7 weights already match this checkpoint.")
        return
    backup_root = ROOT / "runs" / "necropilot-weight-backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(mkdtemp(prefix="previous-", dir=backup_root))
    for name in old:
        shutil.copy2(mod / name, backup / name)

    changed = []
    try:
        for name, data in payload.items():
            # The manifest is replaced after both heads.
            write_file(mod / name, data)
            changed.append(name)
        if any((mod / name).read_bytes() != data for name, data in payload.items()):
            raise OSError("Installed files differ from the checkpoint")
    except (OSError, KeyboardInterrupt):
        for name in reversed(changed):
            write_file(mod / name, old[name])
        raise
    info = json.loads(payload["model.dat"])["checkpoint"]
    print(f"Installed {info['run']} round {info['round']} in {mod}")
    print(f"Previous files: {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Joint or curriculum checkpoint.json")
    parser.add_argument("--game", type=Path, default=DEFAULT_GAME, help="Slay the Spire 2 installation root")
    args = parser.parse_args()
    try:
        update(args.checkpoint.resolve(), args.game.resolve())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")
