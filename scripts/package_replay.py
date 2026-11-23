"""Build an installable local replay mod from a recorded training run."""
import argparse
import json
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

from beta_build import BUILD, GAME_COMMIT, ROOT


def package(source, game, output):
    game_info = json.loads((game / "release_info.json").read_bytes())
    if (game_info.get("version"), game_info.get("commit")) != (BUILD, GAME_COMMIT):
        raise ValueError(f"Expected {BUILD} game commit {GAME_COMMIT}")

    replay = json.loads((source / "replay.json").read_bytes())
    report = json.loads((source / "report.json").read_bytes())
    if (replay.get("schema"), replay.get("build"), replay.get("model_sha256")) != (
        1, BUILD, report["model_sha256"]
    ):
        raise ValueError("Replay schema, game build, or model fingerprint differs")
    if not replay.get("episodes") or not all(row["frames"] for row in replay["episodes"]):
        raise ValueError("Replay episodes must have recorded decisions")

    replay["first_combat"] = {
        key: [sum(row["status"] == "combat_cleared" for row in rows), len(rows)]
        for key, rows in report["cohorts"].items()
    }
    with (ROOT / "mod/NecroReplay.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    project = ROOT / "mod/NecroReplay.csproj"
    subprocess.run(["dotnet", "build", str(project), "-c", "Release", f"-p:GameDir={game}",
                    "--disable-build-servers", "-v:q"], check=True, cwd=ROOT)
    assembly = ROOT / "mod/bin/Release/net9.0/NecroReplay.dll"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        archive.write(assembly, "NecroReplay/NecroReplay.dll")
        archive.writestr("NecroReplay/NecroReplay.json", json.dumps(manifest, indent=2))
        archive.writestr("NecroReplay/replay.dat", json.dumps(replay, ensure_ascii=False, separators=(",", ":")))
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--game", type=Path, default=Path("C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2"))
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    package(options.source, options.game.resolve(), options.out)
