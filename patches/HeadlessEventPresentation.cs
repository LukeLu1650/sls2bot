using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using HarmonyLib;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models.Events;

namespace Sts2Headless;

/// <summary>Compatibility patch for the assembly pinned by beta_build.py.</summary>
internal static class HeadlessEventPresentation
{
    internal static void Install()
    {
        var rest = AccessTools.Method(typeof(DenseVegetation), "Rest");
        var asyncType = rest.GetCustomAttribute<AsyncStateMachineAttribute>()?.StateMachineType
            ?? throw new InvalidDataException("Dense Vegetation Rest state machine was not found.");
        var moveNext = AccessTools.Method(asyncType, "MoveNext");
        new Harmony("local.necrobinder.headless.events").Patch(moveNext,
            transpiler: new HarmonyMethod(typeof(HeadlessEventPresentation), nameof(SkipRestPresentation)));
    }

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
}
