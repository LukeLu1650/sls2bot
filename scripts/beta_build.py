"""Prepare the pinned Windows STS2 host in a private checkout, using local game files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = ROOT / "external/sts2-cli"
GIT_URL = "https://github.com/wuhao21/sts2-cli.git"
REVISION = "9842f7dbb52f2b35d2bad73728f9c7ac54679ed9"
BUILD = "v0.113.1"
GAME_COMMIT = "d09d48e5"
ASSEMBLY = "62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3"
DLLS = ("sts2", "SmartFormat", "SmartFormat.ZString", "Sentry", "Steamworks.NET",
        "MonoMod.Backports", "MonoMod.ILHelpers", "0Harmony", "System.IO.Hashing")
HOST = CHECKOUT / "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll"
MANIFEST = ROOT / "artifacts/windows-beta/host.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()


def verify_host():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (manifest["build"], manifest["game_commit"], manifest["host_revision"]) != (BUILD, GAME_COMMIT, REVISION):
        raise ValueError("Unexpected host manifest")
    if manifest["files"].get("lib/sts2.dll") != ASSEMBLY or revision() != REVISION:
        raise ValueError("Unexpected game or host revision")
    for rel, digest in manifest["files"].items():
        if sha(CHECKOUT / rel) != digest:
            raise ValueError(f"Staged file changed: {rel}")
    return manifest


def patch_relic_purchase(source, eol):
    start = source.index(eol + b"        try", source.index(b"private Dictionary<string, object?> DoBuyRelic("))
    end = source.index(eol + b"    }" + eol + eol + b"    private Dictionary<string, object?> DoBuyPotion(", start)
    return source[:start] + eol.join([
        b"",
        b"        if (_pendingShopPurchaseTask != null)",
        b"            return DetectDecisionPoint();",
        b"",
        b"        var relicId = entry.Model?.Id.Entry ?? \"?\";",
        b"        Log($\"Purchasing relic {relicId} for {entry.Cost}g\");",
        b"        _pendingShopPurchaseTask = Task.Run(() => RunManager.Instance.MerchantSynchronizer!.PurchaseLocalEntry(entry));",
        b"        return DetectDecisionPoint();",
    ]) + source[end:]


def patch_audio_api(source, marker, declarations):
    if source.count(marker) != 1:
        raise ValueError(f"Pinned Godot shim signature changed: {marker!r}")
    eol = b"\r\n" if b"\r\n" in source else b"\n"
    return source.replace(marker, eol.join([marker, *declarations]))


def prepare(game):
    info = json.loads((game / "release_info.json").read_text(encoding="utf-8-sig"))
    data = game / "data_sts2_windows_x86_64"
    if (info["version"], info["commit"], sha(data / "sts2.dll")) != (BUILD, GAME_COMMIT, ASSEMBLY):
        raise ValueError("Unsupported game build. Add a separate compatibility profile for this installation.")
    if not CHECKOUT.exists():
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(CHECKOUT)], check=True)
        subprocess.run(["git", "remote", "add", "origin", GIT_URL], cwd=CHECKOUT, check=True)
        subprocess.run(["git", "fetch", "--depth=1", "origin", REVISION], cwd=CHECKOUT, check=True)
        subprocess.run(["git", "checkout", "--detach", REVISION], cwd=CHECKOUT, check=True)
    if revision() != REVISION:
        raise ValueError(f"Expected dedicated checkout at {REVISION}: {CHECKOUT}")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "scripts/extract_localization.py"], cwd=CHECKOUT, check=True)
    lib = CHECKOUT / "lib"
    lib.mkdir(exist_ok=True)
    files = []
    for name in DLLS:
        path = lib / (name + ".dll")
        shutil.copyfile(data / path.name, path)
        files.append(path)
    subprocess.run([sys.executable, "scripts/extract_localization.py", str(game / "SlayTheSpire2.pck")], cwd=CHECKOUT, check=True)
    simulator = CHECKOUT / "src/Sts2Headless/RunSimulator.cs"
    event_patch = ROOT / "patches/HeadlessEventPresentation.cs"
    staged_patch = simulator.with_name(event_patch.name)
    original = simulator.read_bytes()
    stubs = CHECKOUT / "src/GodotStubs"
    ui, core, math = stubs / "UI.cs", stubs / "Core.cs", stubs / "Math.cs"
    original_ui, original_core, original_math = ui.read_bytes(), core.read_bytes(), math.read_bytes()
    patched_ui = patch_audio_api(original_ui, b"    public float VolumeDb { get; set; }", [
        b"    public float VolumeLinear { get; set; }",
        b"    public float PitchScale { get; set; }",
        b"    public StringName Bus { get; set; } = \"Master\";",
    ])
    patched_core = patch_audio_api(original_core, b"    public virtual bool IsQueuedForDeletion() => false;", [
        b"    public Error Connect(StringName signal, Callable callable, uint flags = 0) => Error.Ok;",
        b"    public void Disconnect(StringName signal, Callable callable) { }",
    ])
    patched_ui = patch_audio_api(patched_ui, b"public class AudioStream : Resource { }", [
        b"public static class AudioServer { public static AudioServerInstance? Singleton => null; }",
        b"public class AudioServerInstance : GodotObject",
        b"{",
        b"    public int GetBusIndex(StringName bus) => -1;",
        b"    public void SetBusVolumeDb(int index, float volumeDb) { }",
        b"}",
    ])
    patched_math = patch_audio_api(original_math, b"    public const float Pi = MathF.PI;", [
        b"    public static float LinearToDb(float linear) => 20f * MathF.Log10(linear);",
    ])
    marker = b'["deck_size"] = '
    setup_marker = b"        PatchAudioManager();"
    if original.count(marker) != 1 or original.count(setup_marker) != 1:
        raise ValueError("Pinned simulator initialization or PlayerSummary shape changed")
    if staged_patch.exists():
        raise ValueError(f"Unexpected existing compatibility patch: {staged_patch}")
    eol = b"\r\n" if b"\r\n" in original else b"\n"
    fields = [b'["potion_capacity"] = player.PotionSlots.Count,',
              b'            ["has_open_potion_slots"] = player.HasOpenPotionSlots,',
              b"            " + marker]
    try:
        shutil.copyfile(event_patch, staged_patch)
        patched = original.replace(marker, eol.join(fields))
        patched = patched.replace(setup_marker, eol.join([setup_marker, b"        HeadlessEventPresentation.Install();"]))
        patched = patch_relic_purchase(patched, eol)
        simulator.write_bytes(patched)
        ui.write_bytes(patched_ui)
        core.write_bytes(patched_core)
        math.write_bytes(patched_math)
        subprocess.run(["dotnet", "build", "src/Sts2Headless/Sts2Headless.csproj", "-c", "Debug",
                        "--disable-build-servers", "--nologo", "-v:q"], cwd=CHECKOUT, check=True)
    finally:
        simulator.write_bytes(original)
        ui.write_bytes(original_ui)
        core.write_bytes(original_core)
        math.write_bytes(original_math)
        staged_patch.unlink(missing_ok=True)
    files.extend([HOST, HOST.with_name("GodotSharp.dll")])
    for language in ("eng", "zhs"):
        files.extend(sorted((CHECKOUT / ("localization_" + language)).glob("*.json")))
    manifest = {"build": BUILD, "game_commit": GAME_COMMIT, "host_revision": REVISION,
                "platform": "Windows x64", "potion_slots_schema": 1, "event_patch_sha256": sha(event_patch),
                "source_pck_sha256": sha(game / "SlayTheSpire2.pck"),
                "files": {p.relative_to(CHECKOUT).as_posix(): sha(p) for p in files}}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temp = MANIFEST.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(MANIFEST)
    verify_host()
    print(f"Verified {BUILD} {GAME_COMMIT}: {MANIFEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, default=Path(os.getenv("ProgramFiles(x86)", "C:/Program Files (x86)"))
                        / "Steam/steamapps/common/Slay the Spire 2")
    prepare(parser.parse_args().game.resolve())
