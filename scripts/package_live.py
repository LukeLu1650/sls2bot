"""Stage the game-side pilot and its Python scoring worker for Windows v0.113.1."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from beta_build import BUILD, GAME_COMMIT, ROOT


def replace(path, old, new):
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError(f"Unexpected upstream code in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def stage_upstream(source):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != "d667489560cc46154638e1208ff032ed96707aa3":
        raise ValueError("Expected upstream sts2ai v1.0")
    target = ROOT / "live/upstream"
    for name in ("Decisions", "Encoding"):
        shutil.copytree(source / "shared" / name, target / name, dirs_exist_ok=True)
    pilot = target / "Pilot.cs"
    shutil.copyfile(source / "mod/Sts2AiMod/Pilot.cs", pilot)
    replace(pilot, "var inv = mr.GetLocalInventory();", "var inv = RunManager.Instance.MerchantSynchronizer?.GetLocalInventory();\n        if (inv == null) return null;")
    replace(pilot, "((MerchantEntry)a.Payload!).OnTryPurchaseWrapper(inv)",
            "RunManager.Instance.MerchantSynchronizer!.PurchaseLocalEntry((MerchantEntry)a.Payload!)")
    combat = target / "Decisions/CombatActions.cs"
    replace(combat, """                if (player.CanRemovePotions)
                    d.Add(ActionType.DiscardPotion, vocab.Id(potion), slot, potion, index: slot, label: $"discard {potion.Id.Entry}");
""", "")
    replace(combat, " || !owner.CanRemovePotions", "")
    replace(target / "Decisions/DecisionBuilder.cs", "pot != null && p.CanRemovePotions", "pot != null")
    for delay, node in ((500, "screen"), (300, "node")):
        replace(pilot, f"await Task.Delay({delay});", f"""int generation = NecroPilot.Live.Generation;
                    await Task.Delay({delay});
                    if (!NecroPilot.Live.Active(generation) || !GodotObject.IsInstanceValid({node})) return;""")
    return target


def package(game, output):
    info = json.loads((game / "release_info.json").read_bytes())
    if (info.get("version"), info.get("commit")) != (BUILD, GAME_COMMIT):
        raise ValueError(f"Expected {BUILD} ({GAME_COMMIT})")
    source = ROOT / "external/sts2ai"
    target = stage_upstream(source)
    subprocess.run(["dotnet", "build", str(ROOT / "live/NecroPilot.csproj"), "-c", "Release",
                    f"-p:GameDir={game}", "--disable-build-servers", "-v:q"], check=True)
    names = {}
    for table, suffix in (("cards", ".title"), ("monsters", ".name"), ("powers", ".title")):
        entries = json.loads((ROOT / "external/sts2-cli/localization_eng" / f"{table}.json").read_bytes())
        names.update({table + "/" + key.removesuffix(suffix): val for key, val in entries.items() if key.endswith(suffix)})
    result = ROOT / "results/windows-v0.113.1/standard-continuation"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        def add(path, name=None):
            archive.write(path, "NecroPilot/" + (name or path.name))
        add(ROOT / "live/bin/Release/net9.0/NecroPilot.dll")
        add(ROOT / "live/NecroPilot.json")
        add(source / "LICENSE", "LICENSE.txt")
        archive.writestr("NecroPilot/pilot.cfg", json.dumps({"Python": sys.executable}))
        archive.writestr("NecroPilot/vocab.dat", '{"ids":[]}')
        archive.writestr("NecroPilot/names.dat", json.dumps(names))
        add(result / "report.json", "model.dat")
        add(result / "readout.npz")
        for name in ("live_policy.py", "train_standard.py", "beta_eval.py", "beta_env.py", "beta_build.py", "necro_policy.py"):
            add(ROOT / "scripts" / name)
        for path in target.rglob("*.cs"):
            archive.write(path, "Source/upstream/" + path.relative_to(target).as_posix())
        for name in ("NecroPilot.csproj", "Observation.cs", "Live.cs"):
            archive.write(ROOT / "live" / name, "Source/" + name)
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, default=Path("C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package(args.game.resolve(), args.out)
