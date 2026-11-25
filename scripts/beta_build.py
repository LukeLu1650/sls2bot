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
STEAM_HOST = CHECKOUT / "src/Sts2Headless/bin/Steam/net9.0/Sts2Headless.dll"
STEAM_MANIFEST = ROOT / "artifacts/windows-beta/host-steam.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CHECKOUT, text=True).strip()


def verify_host(*, steam=False):
    path = STEAM_MANIFEST if steam else MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest["build"], manifest["game_commit"], manifest["host_revision"]) != (BUILD, GAME_COMMIT, REVISION):
        raise ValueError("Unexpected host manifest")
    if manifest["files"].get("lib/sts2.dll") != ASSEMBLY or revision() != REVISION:
        raise ValueError("Unexpected game or host revision")
    for rel, digest in manifest["files"].items():
        if sha(CHECKOUT / rel) != digest:
            raise ValueError(f"Staged file changed: {rel}")
    if steam and manifest.get("steam_run_patch_sha256") != sha(ROOT / "patches/HeadlessSteamRun.cs"):
        raise ValueError("Rebuild the Steam parity host for the current patch")
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


def patch_selection_state(source, eol):
    for scene in (b"card_select", b"bundle_select", b"card_reward"):
        marker = b'["decision"] = "' + scene + b'",'
        start = source.index(marker)
        header = b"return new Dictionary<string, object?>"
        opening = source.rindex(header, 0, start)
        closing = source.index(eol + b"            };", start)
        source = (source[:opening] + source[opening:closing].replace(
            header, b"return WithSelectionCombat(player, new Dictionary<string, object?>", 1)
            + source[closing:].replace(b"            };", b"            });", 1))
    start = source.index(b"var scard = new Dictionary<string, object?>")
    marker = b'["rarity"] = card.Rarity.ToString(),'
    source = source[:start] + source[start:].replace(marker, eol.join([
        b'["target_type"] = card.TargetType.ToString(),',
        b'                    ["zone"] = SelectionCardZone(player, card),',
        b"                    " + marker,
    ]), 1)
    marker = b"    private Dictionary<string, object?> CombatPlayState(Player player)"
    helpers = """    private Dictionary<string, object?> WithSelectionCombat(
        Player player, Dictionary<string, object?> prompt)
    {
        bool inCombat = CombatManager.Instance.IsInProgress;
        prompt["in_combat"] = inCombat;
        if (inCombat)
        {
            var combat = CombatPlayState(player);
            foreach (var key in new[] { "round", "energy", "hand", "enemies", "player_powers",
                                       "osty", "draw_pile_count", "discard_pile_count" })
                if (combat.TryGetValue(key, out var value)) prompt[key] = value;
        }
        return prompt;
    }

    private static string? SelectionCardZone(Player player, CardModel card)
    {
        var pcs = player.PlayerCombatState;
        if (pcs?.Hand.Cards.Contains(card) == true) return "hand";
        if (pcs?.DiscardPile.Cards.Contains(card) == true) return "discard";
        if (pcs?.DrawPile.Cards.Contains(card) == true) return "draw";
        if (pcs?.ExhaustPile.Cards.Contains(card) == true) return "exhaust";
        if (player.Deck.Cards.Contains(card)) return "deck";
        return null;
    }

"""
    if source.count(marker) != 1:
        raise ValueError("Pinned combat snapshot signature changed")
    return source.replace(marker, helpers.encode().replace(b"\n", eol) + marker)


def patch_play_card_check(source, eol):
    before_marker = b"        var playAction = new PlayCardAction(card, target);"
    after_marker = b"        var handAfter = pcs.Hand.Cards;" + eol + \
                   b"        if (handAfter.Count == handCountBefore && cardIndex < handAfter.Count && handAfter[cardIndex] == card)"
    if source.count(before_marker) != 1 or source.count(after_marker) != 1:
        raise ValueError("Pinned DoPlayCard verification signature changed")
    source = source.replace(before_marker, eol.join([
        b"        var playsBefore = CombatManager.Instance.History?.CardPlaysStarted.Count() ?? 0;",
        before_marker,
    ]), 1)
    return source.replace(after_marker, eol.join([
        b"        var handAfter = pcs.Hand.Cards;",
        b"        var playsAfter = CombatManager.Instance.History?.CardPlaysStarted.Count() ?? 0;",
        b"        if (playsAfter == playsBefore && handAfter.Count == handCountBefore && cardIndex < handAfter.Count && handAfter[cardIndex] == card)",
    ]), 1)


def patch_shop_removal(source, eol):
    cost_marker = b'["card_removal_cost"] = removal?.Cost,'
    guard_marker = b'if (removal == null) return Error("No card removal available");'
    if source.count(cost_marker) != 1 or source.count(guard_marker) != 1:
        raise ValueError("Pinned shop card removal signature changed")
    source = source.replace(cost_marker, eol.join([
        b'["card_removal_cost"] = removal is { IsStocked: true, Used: false } ? removal.Cost : null,',
        b'            ["card_removal_stocked"] = removal is { IsStocked: true, Used: false },',
    ]), 1)
    return source.replace(
        guard_marker,
        b'if (removal == null || !removal.IsStocked || removal.Used) return Error("No card removal available");',
        1,
    )


def patch_steam_start(simulator, program, eol):
    changes = (
        (b"public class RunSimulator", b"public partial class RunSimulator"),
        (b"private Player? CreatePlayer(string characterName)",
         b"private Player? CreatePlayer(string characterName, UnlockState? unlocks = null)"),
        (b"(UnlockState.all, 1uL)", b"(unlocks ?? UnlockState.all, 1uL)"),
    )
    for old, new in changes:
        if old not in simulator:
            raise ValueError(f"Pinned Steam run signature changed: {old!r}")
        simulator = simulator.replace(old, new)
    marker = b'["room_type"] = _runState.CurrentRoom?.RoomType.ToString(),'
    if simulator.count(marker) != 1:
        raise ValueError("Pinned Steam run context signature changed")
    simulator = simulator.replace(marker, eol.join([
        b'["act_id"] = _runState.Act?.Id.ToString(),',
        b'            ["encounter_id"] = (_runState.CurrentRoom as CombatRoom)?.Encounter.Id.ToString(),',
        b"            " + marker,
    ]))
    for old, new in (
        (b'            switch (action)', b'            if (action == "reveal_crystal_cell") return RevealCrystalCell(args);'),
        (b'        WaitForShopPurchase();', b'        if (_crystalSphere is { IsFinished: false }) return CrystalSphereState(_runState.Players[0]);'),
        (b'            if (_cardSelector.HasPending || _cardSelector.HasPendingReward ||' + eol,
         b'            if (_crystalSphere is { IsFinished: false }) return;'),
    ):
        if simulator.count(old) != 1:
            raise ValueError(f"Pinned Crystal Sphere hook changed: {old!r}")
        indent = new[:len(new) - len(new.lstrip())]
        simulator = simulator.replace(old, new + eol + indent + old.lstrip())
    marker = b'            case "start_run":'
    if program.count(marker) != 1:
        raise ValueError("Pinned Steam run command signature changed")
    program_eol = b"\r\n" if b"\r\n" in program else b"\n"
    program = program.replace(marker, program_eol.join([
        b'            case "start_steam_run":',
        b"                return sim.StartSteamRun(cmd);",
        b"",
        marker,
        b'                RunSimulator.InstallLateEvents();',
    ]))
    return simulator, program


def prepare(game, *, steam=False):
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
    pck = game / "SlayTheSpire2.pck"
    pck_digest = sha(pck)
    if steam and MANIFEST.exists():
        legacy = verify_host()
        if pck_digest != legacy["source_pck_sha256"]:
            raise ValueError("The game pack differs from the existing Windows host")
        files = [CHECKOUT / rel for rel in legacy["files"] if rel.startswith("lib/")]
    else:
        lib = CHECKOUT / "lib"
        lib.mkdir(exist_ok=True)
        files = []
        for name in DLLS:
            path = lib / (name + ".dll")
            shutil.copyfile(data / path.name, path)
            files.append(path)
        subprocess.run([sys.executable, "scripts/extract_localization.py", str(pck)], cwd=CHECKOUT, check=True)
    simulator = CHECKOUT / "src/Sts2Headless/RunSimulator.cs"
    event_patch = ROOT / "patches/HeadlessEventPresentation.cs"
    staged_patch = simulator.with_name(event_patch.name)
    original = simulator.read_bytes()
    program = simulator.with_name("Program.cs")
    original_program = program.read_bytes()
    steam_patch = ROOT / "patches/HeadlessSteamRun.cs"
    staged_steam = simulator.with_name(steam_patch.name)
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
        b"    public Variant Call(StringName method, params Variant[] args) => default;",
        b"    public Variant CallDeferred(StringName method, params Variant[] args) => default;",
    ])
    node_call_deferred = b"    public void CallDeferred(StringName method, params Variant[] args) { }"
    if patched_core.count(node_call_deferred) != 1:
        raise ValueError("Pinned Node.CallDeferred stub signature changed")
    patched_core = patched_core.replace(node_call_deferred, b"", 1)
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
    if steam:
        patched_math = patch_audio_api(patched_math, b"    public const float Pi = MathF.PI;", [
            b"    public static float Log(float x) => MathF.Log(x);",
        ])
        patched_math = patch_audio_api(patched_math, b"    public static Vector2I Zero { get; } = new(0, 0);", [
            b"    public static Vector2I One { get; } = new(1, 1);",
        ])
    marker = b'["deck_size"] = '
    setup_marker = b"        PatchAudioManager();"
    if original.count(marker) != 1 or original.count(setup_marker) != 1:
        raise ValueError("Pinned simulator initialization or PlayerSummary shape changed")
    for path in [staged_patch, staged_steam]:
        if path.exists():
            raise ValueError(f"Unexpected existing compatibility patch: {path}")
    eol = b"\r\n" if b"\r\n" in original else b"\n"
    fields = [b'["potion_capacity"] = player.PotionSlots.Count,',
              b'            ["has_open_potion_slots"] = player.HasOpenPotionSlots,',
              b"            " + marker]
    try:
        shutil.copyfile(event_patch, staged_patch)
        patched = original.replace(marker, eol.join(fields))
        patched = patched.replace(setup_marker, eol.join([setup_marker, b"        HeadlessEventPresentation.Install();"]))
        patched = patch_relic_purchase(patched, eol)
        patched = patch_selection_state(patched, eol)
        patched = patch_play_card_check(patched, eol)
        patched = patch_shop_removal(patched, eol)
        if steam:
            shutil.copyfile(steam_patch, staged_steam)
            patched, patched_program = patch_steam_start(patched, original_program, eol)
            program.write_bytes(patched_program)
        simulator.write_bytes(patched)
        ui.write_bytes(patched_ui)
        core.write_bytes(patched_core)
        math.write_bytes(patched_math)
        subprocess.run(["dotnet", "build", "src/Sts2Headless/Sts2Headless.csproj", "-c", "Steam" if steam else "Debug",
                        "--disable-build-servers", "--nologo", "-v:q"], cwd=CHECKOUT, check=True)
    finally:
        simulator.write_bytes(original)
        ui.write_bytes(original_ui)
        core.write_bytes(original_core)
        math.write_bytes(original_math)
        staged_patch.unlink(missing_ok=True)
        if steam:
            program.write_bytes(original_program)
            staged_steam.unlink(missing_ok=True)
    host, manifest_path = (STEAM_HOST, STEAM_MANIFEST) if steam else (HOST, MANIFEST)
    files.extend([host, host.with_name("GodotSharp.dll")])
    for language in ("eng", "zhs"):
        files.extend(sorted((CHECKOUT / ("localization_" + language)).glob("*.json")))
    manifest = {"build": BUILD, "game_commit": GAME_COMMIT, "host_revision": REVISION,
                "platform": "Windows x64", "potion_slots_schema": 1, "selection_schema": 1,
                "event_patch_sha256": sha(event_patch),
                "source_pck_sha256": pck_digest,
                "files": {p.relative_to(CHECKOUT).as_posix(): sha(p) for p in files}}
    if steam:
        manifest["steam_run_patch_sha256"] = sha(steam_patch)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(manifest_path)
    verify_host(steam=steam)
    print(f"Verified {BUILD} {GAME_COMMIT}: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, default=Path(os.getenv("ProgramFiles(x86)", "C:/Program Files (x86)"))
                        / "Steam/steamapps/common/Slay the Spire 2")
    parser.add_argument("--steam-parity", action="store_true", help="Build the separate Steam-configured host")
    args = parser.parse_args()
    prepare(args.game.resolve(), steam=args.steam_parity)
