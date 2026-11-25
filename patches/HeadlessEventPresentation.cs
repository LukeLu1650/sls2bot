using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

namespace Sts2Headless;

/// <summary>Compatibility patch for the assembly pinned by beta_build.py.</summary>
internal static class HeadlessEventPresentation
{
    private static NDebugAudioManager _silentAudio = null!;
    private static int _soundId;

    internal static void Install()
    {
        var harmony = new Harmony("local.necrobinder.headless.events");
        InstallDebugAudio(harmony);

        // DenseVegetation.Rest: skip local presentation/audio/shake block
        var rest = AccessTools.Method(typeof(DenseVegetation), "Rest");
        var asyncType = rest.GetCustomAttribute<AsyncStateMachineAttribute>()?.StateMachineType
            ?? throw new InvalidDataException("Dense Vegetation Rest state machine was not found.");
        harmony.Patch(AccessTools.Method(asyncType, "MoveNext"),
            transpiler: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(SkipRestPresentation)));

        // Unchecked NGame.Instance.ScreenShakeTrauma in PunchOff.Nab (Amalgamator is patched in HeadlessSteamRun)
        PatchAsyncScreenShake(harmony, typeof(PunchOff), "Nab");

        // Unchecked NEventRoom.Instance / NModalContainer calls in Act 3 event The Trial
        // (WaterfallOfObviousSecrets is patched in HeadlessSteamRun)
        harmony.Patch(AccessTools.Method(typeof(Trial), "Accept"),
            transpiler: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(SkipEventRoomPresentation)));
        harmony.Patch(AccessTools.Method(typeof(Trial), "AddVfxAnchoredToPortrait"),
            prefix: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(SkipVoidOperation)));
        harmony.Patch(AccessTools.Method(typeof(Trial), "DoubleDown"),
            prefix: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(CompletedTaskOperation)));
    }

    private static void PatchAsyncScreenShake(Harmony harmony, Type eventType, string methodName)
    {
        var method = AccessTools.Method(eventType, methodName);
        var smType = method.GetCustomAttribute<AsyncStateMachineAttribute>()?.StateMachineType
            ?? throw new InvalidDataException($"{eventType.Name}.{methodName} state machine was not found.");
        harmony.Patch(AccessTools.Method(smType, "MoveNext"),
            transpiler: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(SkipScreenShakeCalls)));
    }

    private static void InstallDebugAudio(Harmony harmony)
    {
        var audio = typeof(NDebugAudioManager);
        // This process never has a Godot audio tree. The prefixes replace all the
        // callable playback/volume operations, so the sentinel never uses Node internals.
        _silentAudio = (NDebugAudioManager)RuntimeHelpers.GetUninitializedObject(audio);
        harmony.Patch(AccessTools.PropertyGetter(audio, nameof(NDebugAudioManager.Instance)),
            prefix: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(SilentAudio)));
        harmony.Patch(AccessTools.Method(audio, nameof(NDebugAudioManager.Play)),
            prefix: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(SilentPlay)));
        foreach (string method in new[] { "Stop", "StopAll", "SetMasterAudioVolume", "SetSfxAudioVolume" })
            harmony.Patch(AccessTools.Method(audio, method),
                prefix: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(SkipAudioOperation)));
    }

    private static bool SilentAudio(ref NDebugAudioManager __result)
    {
        __result = _silentAudio;
        return false;
    }

    private static bool SilentPlay(ref int __result)
    {
        __result = Interlocked.Increment(ref _soundId);
        return false;
    }

    private static bool SkipAudioOperation() => false;

    private static IEnumerable<CodeInstruction> SkipRestPresentation(IEnumerable<CodeInstruction> input)
    {
        var instructions = input.ToList();
        var isLocal = AccessTools.Method(typeof(LocalContext), nameof(LocalContext.IsMe), [typeof(Player)]);
        var guards = instructions.Where(instruction => instruction.Calls(isLocal)).ToArray();
        if (guards.Length != 1)
            throw new InvalidDataException("Unexpected Dense Vegetation Rest presentation guard.");
        int index = instructions.IndexOf(guards[0]);
        if (instructions[index + 1].opcode != OpCodes.Brfalse && instructions[index + 1].opcode != OpCodes.Brfalse_S)
            throw new InvalidDataException("Dense Vegetation Rest presentation branch was not found.");

        // The pinned method heals before this guard. The guarded block plays/stops sounds,
        // waits for the presentation and shakes the screen. The Fight option is set afterwards.
        guards[0].opcode = OpCodes.Pop;
        guards[0].operand = null;
        instructions.Insert(index + 1, new CodeInstruction(OpCodes.Ldc_I4_0));
        return instructions;
    }

    private static bool SkipVoidOperation() => false;

    private static bool CompletedTaskOperation(ref Task __result)
    {
        __result = Task.CompletedTask;
        return false;
    }

    private static void SkipScreenShake(NGame? game, ShakeStrength strength) { }

    private static IEnumerable<CodeInstruction> SkipScreenShakeCalls(IEnumerable<CodeInstruction> input)
    {
        var instructions = input.ToList();
        var shake = AccessTools.Method(typeof(NGame), "ScreenShakeTrauma", [typeof(ShakeStrength)]);
        var calls = instructions.Where(instruction => instruction.Calls(shake)).ToArray();
        if (calls.Length == 0)
            throw new InvalidDataException("Expected NGame.ScreenShakeTrauma call was not found.");
        var replacement = AccessTools.Method(typeof(HeadlessEventPresentation), nameof(SkipScreenShake));
        foreach (var call in calls)
        {
            call.opcode = OpCodes.Call;
            call.operand = replacement;
        }
        return instructions;
    }

    private static IEnumerable<CodeInstruction> SkipEventRoomPresentation(IEnumerable<CodeInstruction> input)
    {
        var instructions = input.ToList();
        var isLocal = AccessTools.Method(typeof(LocalContext), nameof(LocalContext.IsMe), [typeof(Player)]);
        var eventRoomInstance = AccessTools.PropertyGetter(typeof(NEventRoom), nameof(NEventRoom.Instance));
        int patched = 0;
        for (int i = 0; i + 2 < instructions.Count; i++)
        {
            if (instructions[i].Calls(isLocal)
                && (instructions[i + 1].opcode == OpCodes.Brfalse || instructions[i + 1].opcode == OpCodes.Brfalse_S)
                && instructions[i + 2].Calls(eventRoomInstance))
            {
                instructions[i].opcode = OpCodes.Pop;
                instructions[i].operand = null;
                instructions.Insert(i + 1, new CodeInstruction(OpCodes.Ldc_I4_0));
                patched++;
                i++;
            }
        }
        if (patched == 0)
            throw new InvalidDataException("Expected NEventRoom presentation guard was not found.");
        return instructions;
    }
}
