using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text.Json;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Validation;
using MegaCrit.Sts2.Core.TestSupport;
using MegaCrit.Sts2.Core.Unlocks;

namespace Sts2Headless;

public partial class RunSimulator
{
    public Dictionary<string, object?> StartSteamRun(JsonElement command)
    {
        try
        {
            _loc.Lang = "en";
            EnsureModelDbInitialized();
            InstallSteamCompatibility();
            if (command.TryGetProperty("trace_enemy_moves", out var traceMoves) && traceMoves.GetBoolean())
                EnableSteamEnemyMoveTrace();
            var source = command.GetProperty("run");
            if (source.GetProperty("game_mode").GetString() != "standard" ||
                source.GetProperty("modifiers").GetArrayLength() != 0)
                return Error("Steam parity setup requires Standard mode without modifiers.");

            var parsed = SaveManager.FromJson<SerializableProgress>(command.GetProperty("progress").GetRawText());
            if (!parsed.Success || parsed.SaveData == null)
                return Error($"Cannot read the Steam progress snapshot: {parsed.Status}");
            var progress = ProgressState.FromSerializable(parsed.SaveData, new DeserializationContext());
            var unlocks = new UnlockState(progress);
            var player = CreatePlayer(source.GetProperty("character").GetString() ?? "", unlocks);
            if (player == null)
                return Error("Unsupported character in Steam run configuration.");
            var seed = source.GetProperty("seed").GetString()
                ?? throw new ArgumentException("Steam seed must be set.");
            var acts = source.TryGetProperty("acts", out var recordedActs) ? recordedActs.EnumerateArray()
                .Select(entry => entry.GetString())
                .Select(id => id != null && id.StartsWith("ACT.", StringComparison.Ordinal)
                    ? ModelDb.GetById<ActModel>(new ModelId("ACT", id[4..])).ToMutable()
                    : throw new ArgumentException($"Expected an ACT model ID, got '{id}'."))
                .ToList() : ActModel.GetRandomList(new Rng(StringHelper.GetDeterministicHashCode(seed), "act_selection"), unlocks, false)
                    .Select(act => act.ToMutable()).ToList();
            if (acts.Count != 3)
                return Error("Steam parity setup requires the three recorded acts.");

            _runState = RunState.CreateForNewRun([player], acts, [], GameMode.Standard,
                source.GetProperty("is_custom_game").GetBoolean(),
                source.GetProperty("ascension").GetInt32(), seed);
            _combatTurnTimedOut = false;
            _pendingEventOptionTask = null;
            _pendingShopPurchaseTask = null;
            _pendingTreasureTask = null;
            ResetRoomTracking(player);
            _runState.ExtraFields.StartedWithNeow = true;

            // The retail setup generates rooms. Saving stays disabled in this headless worker.
            var manager = RunManager.Instance;
            manager.SetUpNewSingleplayer(_runState, shouldSave: false, dailyTime: null);
            LocalContext.NetId = manager.NetService.NetId;
            CombatManager.Instance.TurnStarted += state => { DetachUnsafeMonsterVfxHandlers(state); _turnStarted.Set(); };
            CombatManager.Instance.CombatEnded += _ => _combatEnded.Set();
            manager.FinalizeStartingRelics().GetAwaiter().GetResult();
            manager.Launch();
            manager.EnterAct(0, doTransition: false).GetAwaiter().GetResult();
            manager.CardSelectorStack.UseSelector(player, _cardSelector);
            LocPatches._bundleSimRef = this;
            return DetectDecisionPoint();
        }
        catch (Exception ex)
        {
            return ErrorWithTrace("StartSteamRun failed", ex);
        }
    }

    private static bool _steamCompatibilityInstalled;
    private static bool _steamEnemyMoveTraceInstalled;
    private static bool _lateEventsInstalled;
    private static CrystalSphereMinigame? _crystalSphere;

    private static bool OpenCrystalSphere(CrystalSphereMinigame __0)
    {
        _crystalSphere = __0;
        return false;
    }

    private static CrystalSphereCell[,] CrystalCells(CrystalSphereMinigame game) =>
        (CrystalSphereCell[,])AccessTools.Field(typeof(CrystalSphereMinigame), "cells").GetValue(game)!;

    private Dictionary<string, object?> CrystalSphereState(Player player)
    {
        var game = _crystalSphere!;
        return new()
        {
            ["type"] = "decision", ["decision"] = "crystal_sphere",
            ["context"] = RunContext(), ["player"] = PlayerSummary(player),
            ["divinations"] = game.DivinationCount,
            ["cells"] = CrystalCells(game).Cast<CrystalSphereCell>()
                .Select(cell => new { x = cell.X, y = cell.Y, hidden = cell.IsHidden }).ToArray(),
        };
    }

    private Dictionary<string, object?> RevealCrystalCell(Dictionary<string, object?>? args)
    {
        if (_crystalSphere is not { IsFinished: false } game || args == null ||
            !args.TryGetValue("x", out var xValue) || !args.TryGetValue("y", out var yValue))
            return Error("No pending Crystal Sphere cell selection");
        int x = Convert.ToInt32(xValue), y = Convert.ToInt32(yValue);
        var cell = CrystalCells(game).Cast<CrystalSphereCell>()
            .FirstOrDefault(cell => cell.X == x && cell.Y == y && cell.IsHidden);
        if (cell == null) return Error("Crystal Sphere cell is unavailable");
        var task = game.CellClicked(cell);
        var deadline = Environment.TickCount64 + 5000;
        while (!task.IsCompleted && Environment.TickCount64 < deadline)
        {
            _syncCtx.Pump();
            if (!task.IsCompleted) Thread.Sleep(1);
        }
        if (!task.IsCompleted) throw new TimeoutException("Crystal Sphere cell did not resolve");
        task.GetAwaiter().GetResult();
        if (game.IsFinished) _crystalSphere = null;
        return DetectDecisionPoint();
    }

    private static void EnableSteamEnemyMoveTrace()
    {
        if (_steamEnemyMoveTraceInstalled) return;
        var perform = AccessTools.Method(typeof(MonsterModel), nameof(MonsterModel.PerformMove));
        new Harmony("local.necrobinder.headless.steam.trace").Patch(perform,
            prefix: new HarmonyMethod(typeof(RunSimulator), nameof(TraceSteamEnemyMove)));
        _steamEnemyMoveTraceInstalled = true;
    }

    private static void TraceSteamEnemyMove(MonsterModel __instance)
    {
        if (__instance.NextMove is { } move)
            Console.Error.WriteLine($"[STEAM-TRACE] Enemy move {__instance.Id.Entry} {move.Id}");
    }

    private static void InstallSteamCompatibility()
    {
        if (_steamCompatibilityInstalled) return;
        var harmony = new Harmony("local.necrobinder.headless.steam");
        var enterWaterfall = AccessTools.Method(typeof(WaterfallOfObviousSecrets), "EnterWaterfall");
        harmony.Patch(enterWaterfall,
            transpiler: new HarmonyMethod(typeof(RunSimulator), nameof(SkipWaterfallPortrait)));
        var potionCost = AccessTools.Method(typeof(MerchantPotionEntry), "CalcCost");
        harmony.Patch(potionCost,
            transpiler: new HarmonyMethod(typeof(RunSimulator), nameof(EnablePotionPriceRoll)));
        InstallLateEvents();
        _steamCompatibilityInstalled = true;
    }

    internal static void InstallLateEvents()
    {
        if (_lateEventsInstalled) return;
        var harmony = new Harmony("local.necrobinder.headless.late-events");
        harmony.Patch(AccessTools.Method(typeof(NCrystalSphereScreen), "ShowScreen"),
            prefix: new HarmonyMethod(typeof(RunSimulator), nameof(OpenCrystalSphere)));
        foreach (string option in new[] { "CombineStrikes", "CombineDefends" })
        {
            var stateMachine = AccessTools.Method(typeof(Amalgamator), option)
                .GetCustomAttributes(typeof(AsyncStateMachineAttribute), false)
                .Cast<AsyncStateMachineAttribute>().Single().StateMachineType;
            harmony.Patch(AccessTools.Method(stateMachine, "MoveNext"),
                transpiler: new HarmonyMethod(typeof(RunSimulator), nameof(SkipAmalgamatorShake)));
        }
        _lateEventsInstalled = true;
    }

    private static void SkipScreenShake(NGame game, ShakeStrength strength) { }

    private static IEnumerable<CodeInstruction> SkipAmalgamatorShake(IEnumerable<CodeInstruction> input)
    {
        var instructions = input.ToList();
        var shake = AccessTools.Method(typeof(NGame), "ScreenShakeTrauma", [typeof(ShakeStrength)]);
        var calls = instructions.Where(instruction => instruction.Calls(shake)).ToArray();
        if (calls.Length != 2) throw new InvalidDataException("Unexpected Amalgamator screen shake calls.");
        foreach (var call in calls)
        {
            call.opcode = OpCodes.Call;
            call.operand = AccessTools.Method(typeof(RunSimulator), nameof(SkipScreenShake));
        }
        return instructions;
    }

    private static IEnumerable<CodeInstruction> EnablePotionPriceRoll(IEnumerable<CodeInstruction> input)
    {
        var instructions = input.ToList();
        var isOff = AccessTools.PropertyGetter(typeof(TestMode), nameof(TestMode.IsOff));
        var guards = instructions.Where(instruction => instruction.Calls(isOff)).ToArray();
        if (guards.Length != 1)
            throw new InvalidDataException("Unexpected merchant potion price guard.");
        // Retail rolls each potion's price before generating card offers from the same RNG.
        guards[0].opcode = OpCodes.Ldc_I4_1;
        guards[0].operand = null;
        return instructions;
    }

    private static IEnumerable<CodeInstruction> SkipWaterfallPortrait(IEnumerable<CodeInstruction> input)
    {
        var instructions = input.ToList();
        var isLocal = AccessTools.Method(typeof(LocalContext), nameof(LocalContext.IsMe), [typeof(Player)]);
        var guards = instructions.Where(instruction => instruction.Calls(isLocal)).ToArray();
        if (guards.Length != 1)
            throw new InvalidDataException("Unexpected Waterfall portrait guard.");
        var guard = guards[0];
        int index = instructions.IndexOf(guard);
        if (instructions[index + 1].opcode != OpCodes.Brfalse && instructions[index + 1].opcode != OpCodes.Brfalse_S)
            throw new InvalidDataException("Waterfall portrait branch was not found.");
        // The guarded branch only swaps a Godot portrait. The option page follows it.
        guard.opcode = OpCodes.Pop;
        guard.operand = null;
        instructions.Insert(index + 1, new CodeInstruction(OpCodes.Ldc_I4_0));
        return instructions;
    }
}
