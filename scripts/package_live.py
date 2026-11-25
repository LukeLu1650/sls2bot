"""Stage the game-side pilot and its Python scoring worker for Windows v0.113.1."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from beta_build import BUILD, GAME_COMMIT, ROOT, sha

DEFAULT_GAME = Path("C:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2")
BASE_FILES = ROOT / "results/windows-v0.113.1/standard-continuation"
WORKER_FILES = ("live_policy.py", "train_standard.py", "beta_eval.py", "beta_env.py", "beta_build.py",
                "necro_policy.py", "train_combat.py", "train_strategy.py", "strategic_features.py", "strategy_updates.py")


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
    builder = target / "Decisions/DecisionBuilder.cs"
    replace(builder, "pot != null && p.CanRemovePotions", "pot != null")
    replace(builder, "MinSelect = minSelect, MaxSelect = maxSelect };",
            "MinSelect = minSelect, MaxSelect = maxSelect, CardOptions = opts, "
            "PickedIndices = Enumerable.Range(0, opts.Count).Where(i => picked.Contains(opts[i])).ToArray() };")
    replace(builder, 'c, label: "select " + c.Id.Entry', 'c, index: i, label: "select " + c.Id.Entry')
    replace(builder, "if (picked.Contains(c)) continue;", "if (picked.Count >= maxSelect || picked.Contains(c)) continue;")
    replace(target / "Decisions/LegalAction.cs", "    public int MinSelect, MaxSelect;", """    public int MinSelect, MaxSelect;
    public IReadOnlyList<MegaCrit.Sts2.Core.Models.CardModel> CardOptions = [];
    public int[] PickedIndices = [];""")
    replace(pilot, '    static readonly FieldInfo GridCards = typeof(NCardGridSelectionScreen).GetField("_cards", Priv)!;',
            '''    static readonly FieldInfo GridCards = typeof(NCardGridSelectionScreen).GetField("_cards", Priv)!;
    static readonly FieldInfo GridControl = typeof(NCardGridSelectionScreen).GetField("_grid", Priv)!;''')
    replace(pilot, "if (GridCards.GetValue(screen) is not IReadOnlyList<CardModel> cards || cards.Count == 0) return null;",
            """// Combat pile prompts update the grid directly as the pile changes.
        IReadOnlyList<CardModel>? cards = screen is NCombatPileCardSelectScreen
            ? (GridControl.GetValue(screen) as NCardGrid)?.CurrentlyDisplayedCards.ToArray()
            : GridCards.GetValue(screen) as IReadOnlyList<CardModel>;
        if (cards is not { Count: > 0 }) return null;""")
    replace(pilot, "return Settled(grid) ? GridSelectScreen(state, player, grid) : null;",
            """waitingFor = "card selection: " + grid.GetType().Name;
                return Settled(grid) ? GridSelectScreen(state, player, grid) : null;""")
    replace(pilot, 't.GetField("_selectedCards", Priv)', 'HarmonyLib.AccessTools.Field(t, "_selectedCards")')
    replace(pilot, 't.GetField("_prefs", Priv)', 'HarmonyLib.AccessTools.Field(t, "_prefs")')
    replace(pilot, '''        NConfirmButton? Confirm()
        {
            var preview = screen.GetNodeOrNull<Control>("%PreviewContainer");
            var previewConfirm = preview?.GetNodeOrNull<NConfirmButton>("%PreviewConfirm") ?? screen.GetNodeOrNull<NConfirmButton>("%PreviewConfirm");
            if (preview is { Visible: true } && previewConfirm is { IsEnabled: true }) return previewConfirm;
            var confirm = screen.GetNodeOrNull<NConfirmButton>("%Confirm");
            return confirm is { IsEnabled: true } ? confirm : null;
        }
        if (picked.Count >= max)
        {
            // enough cards picked: confirm (deck screens show a preview with its own confirm button)
            var c = Confirm();
            if (c == null) { StuckLog("card select confirm", () => $"{t.Name} picked={picked.Count}/{max}: no enabled confirm"); return null; }
            return new(new Decision { Type = DecisionType.CardSelect }, "card select: confirm", _ => { c.ForceClick(); return Task.CompletedTask; });
        }
        var d = build.CardSelect(state, player, cards, picked, min, max);
        return new(d, "card select (" + t.Name + ")", a =>
        {
            if (a.Type == ActionType.Confirm) Confirm()?.ForceClick();''',
            '''        var preview = NecroPilot.SelectionControls.Preview(screen);
        var confirm = preview.IsOpen ? preview.Button : NecroPilot.SelectionControls.GridConfirm(screen);
        if (preview.IsOpen || picked.Count >= max)
        {
            if (confirm == null || picked.Count < min)
            {
                StuckLog("card select confirm", () => $"{t.Name} picked={picked.Count}/{max}: no enabled confirm");
                return null;
            }
            var confirmation = build.CardSelect(state, player, cards, picked, min, max);
            confirmation.Actions.RemoveAll(a => a.Type != ActionType.Confirm);
            return new(confirmation, "card select: confirm", _ => { confirm.ForceClick(); return Task.CompletedTask; });
        }
        var d = build.CardSelect(state, player, cards, picked, min, max);
        if (screen is NDeckUpgradeSelectScreen && state.CurrentRoom is RestSiteRoom) d.Context = "smith";
        if (confirm == null) d.Actions.RemoveAll(a => a.Type == ActionType.Confirm);
        return new(d, "card select (" + t.Name + ")", a =>
        {
            if (a.Type == ActionType.Confirm) confirm?.ForceClick();''')
    replace(pilot, "var d = build.CardSelect(state, player, options, picked, min, max);",
            """var d = build.CardSelect(state, player, options, picked, min, max);
        if (confirm?.IsEnabled != true) d.Actions.RemoveAll(a => a.Type == ActionType.Confirm);""")
    replace(pilot, 'new(new Decision { Type = DecisionType.CardSelect }, "hand select: confirm"',
            'new(build.CardSelect(state, player, options, picked, min, max), "hand select: confirm"')
    replace(pilot, "        var room = state.CurrentRoom;\n        var roomNode = ",
            '''        var room = state.CurrentRoom;
        if (room is RestSiteRoom rest)
        {
            waitingFor = "rest site controls";
            return Rest(state, player, rest);
        }
        var roomNode = ''')
    replace(pilot, "            RestSiteRoom rs => Rest(state, player, rs, roomNode),\n", "")
    replace(pilot, '''    Pending? Rest(RunState state, Player player, RestSiteRoom rs, Node roomNode)
    {
        var node = FindFirst<NRestSiteRoom>(roomNode);
        if (node == null) return null;
        var buttons = FindAll<NRestSiteButton>(node);
        // nothing clickable yet (room fading in, or an option's animation still running): wait
        if (!node.ProceedButton.IsEnabled && !buttons.Any(b => b.IsEnabled && b.Visible)) return null;
        // Mend (heal an ally) needs UI targeting: unavailable, as in training (where co-op disables it)
        var d = build.Rest(state, player, rs.Options, o => o is not MendRestSiteOption);
        return new(d, "rest site", a =>
        {
            if (a.Type == ActionType.ChooseOption)
                buttons.FirstOrDefault(b => ReferenceEquals(b.Option, a.Payload))?.ForceClick();
            else if (node.ProceedButton.IsEnabled)
                node.ProceedButton.ForceClick();
            return Task.CompletedTask;
        });
    }''',
            '''    Pending? Rest(RunState state, Player player, RestSiteRoom room)
    {
        if (NRestSiteRoom.Instance is not { } node || !node.IsVisibleInTree()) return null;

        // The game enables travel after the rest-site choice has finished.
        if (node.ProceedButton is { IsEnabled: true } proceed && proceed.IsVisibleInTree())
            return new(new Decision { Type = DecisionType.Rest }, "rest site: proceed", _ =>
            {
                proceed.ForceClick();
                return Task.CompletedTask;
            });

        var buttons = FindAll<NRestSiteButton>(node)
            .Where(button => button.IsEnabled && button.IsVisibleInTree() && !button.IsQueuedForDeletion())
            .ToArray();
        var d = build.Rest(state, player, room.Options, option => option is not MendRestSiteOption &&
            buttons.Any(button => ReferenceEquals(button.Option, option)));
        d.Actions.RemoveAll(action => action.Type != ActionType.ChooseOption);
        if (d.Actions.Count == 0) return null;

        return new(d, "rest site", action =>
        {
            buttons.First(button => ReferenceEquals(button.Option, action.Payload)).ForceClick();
            return Task.CompletedTask;
        });
    }''')
    for delay, node in ((500, "screen"), (300, "node")):
        replace(pilot, f"await Task.Delay({delay});", f"""int generation = NecroPilot.Live.Generation;
                    await Task.Delay({delay});
                    if (!NecroPilot.Live.Active(generation) || !GodotObject.IsInstanceValid({node})) return;""")
    return target


def check_game(game):
    info = json.loads((game / "release_info.json").read_bytes())
    if (info.get("version"), info.get("commit")) != (BUILD, GAME_COMMIT):
        raise ValueError(f"Expected {BUILD} ({GAME_COMMIT})")


def checkpoint_payload(checkpoint, *, full):
    from live_policy import validate_combat_weights, validate_weights
    from strategic_features import FEATURE_COUNT
    from train_combat import COMBAT_FEATURES_VERSION, COMBAT_SCORE_SOURCES

    raw = checkpoint.read_bytes()
    saved = json.loads(raw)
    protocol = saved["protocol"]
    if (protocol.get("build", protocol.get("game")), protocol.get("mode"), protocol.get("ascension"),
            protocol.get("base"), protocol.get("combat_features")) != (
            BUILD, "standard", 0, sha(BASE_FILES / "readout.npz"), COMBAT_FEATURES_VERSION):
        raise ValueError("Checkpoint has an incompatible game, base or selection encoder")
    if any(protocol["sources"].get(name) != sha(ROOT / "scripts" / name) for name in COMBAT_SCORE_SOURCES):
        raise ValueError("Checkpoint scoring sources differ from this worker")
    weights = saved["weights"]
    combat = validate_combat_weights(weights["combat"] if isinstance(weights, dict) else weights)
    heads = {"combat.dat": json.dumps(combat.tolist()).encode()}
    if full:
        if not isinstance(weights, dict) or "strategy" not in weights:
            raise ValueError("A full checkpoint requires both combat and strategy weights")
        strategy = validate_weights(weights["strategy"], FEATURE_COUNT, "strategy")
        heads["strategy.dat"] = json.dumps(strategy.tolist()).encode()
    model = json.loads((BASE_FILES / "report.json").read_bytes())
    model["checkpoint"] = dict(run=checkpoint.parent.name, round=saved["round"],
                               sha256=sha256(raw).hexdigest(),
                               files={name: sha256(data).hexdigest() for name, data in heads.items()})
    return {**heads, "model.dat": json.dumps(model, ensure_ascii=False).encode("utf-8")}


def package(game, output, combat_checkpoint=None, *, joint_checkpoint=None):
    check_game(game)
    checkpoint = joint_checkpoint or combat_checkpoint
    payload = checkpoint_payload(checkpoint, full=joint_checkpoint is not None) if checkpoint else {}
    source = ROOT / "external/sts2ai"
    target = stage_upstream(source)
    subprocess.run(["dotnet", "build", str(ROOT / "live/NecroPilot.csproj"), "-c", "Release",
                    f"-p:GameDir={game}", "--disable-build-servers", "-v:q"], check=True)
    names = {}
    for table, suffix in (("cards", ".title"), ("monsters", ".name"), ("powers", ".title"),
                          ("potions", ".title"), ("relics", ".title")):
        entries = json.loads((ROOT / "external/sts2-cli/localization_eng" / f"{table}.json").read_bytes())
        names.update({table + "/" + key.removesuffix(suffix): val for key, val in entries.items() if key.endswith(suffix)})
        if table == "cards":
            names.update({"cards/" + key: val for key, val in entries.items() if key.endswith(".description")})
    for table in ("events", "ancients"):
        entries = json.loads((ROOT / "external/sts2-cli/localization_eng" / f"{table}.json").read_bytes())
        names.update({table + "/" + key: val for key, val in entries.items() if key.endswith(".title")})
    names = {key: re.sub(r"\[/?[a-zA-Z_][a-zA-Z0-9_=]*\]", "", value) for key, value in names.items()}
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
        if not payload:
            add(BASE_FILES / "report.json", "model.dat")
        add(BASE_FILES / "readout.npz")
        for name, data in payload.items():
            archive.writestr("NecroPilot/" + name, data)
        for name in WORKER_FILES:
            add(ROOT / "scripts" / name)
        for path in target.rglob("*.cs"):
            archive.write(path, "Source/upstream/" + path.relative_to(target).as_posix())
        for name in ("NecroPilot.csproj", "Observation.cs", "Live.cs", "SelectionControls.cs"):
            archive.write(ROOT / "live" / name, "Source/" + name)
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, default=DEFAULT_GAME)
    parser.add_argument("--out", type=Path, required=True)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--combat-checkpoint", type=Path, help="Combat weights from a combat or joint checkpoint")
    checkpoints.add_argument("--checkpoint", type=Path, help="Combat and strategy weights from a joint or curriculum checkpoint")
    args = parser.parse_args()
    package(args.game.resolve(), args.out, args.combat_checkpoint, joint_checkpoint=args.checkpoint)
