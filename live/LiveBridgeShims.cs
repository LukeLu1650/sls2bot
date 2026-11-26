using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// Live-game shims matching the minimal surface referenced by the shared AlphaDefect v0.114.0
/// Observation encoder (Engine/Observation.cs + Engine/Obs/**/*.cs). No headless-only code or
/// TestMode overrides are activated in the real client.
/// </summary>
public sealed record SelectInfo(
    string? PromptTable,
    string? PromptKey,
    string Method,
    EnchantmentModel? Enchantment,
    AbstractModel? Source = null);

public sealed class PendingCardChoice(
    IReadOnlyList<CardModel> options,
    int min,
    int max,
    bool cancelable,
    SelectInfo? info = null)
{
    public SelectInfo? Info { get; } = info;
    public IReadOnlyList<CardModel> Options { get; } = options;
    public int Min { get; } = min;
    public int Max { get; } = max;
    public bool Cancelable { get; } = cancelable;
}

public static class AgentCardSelector
{
    public static PotionModel? UsingPotion { get; set; }
    public static SelectInfo? ActiveSelectInfo { get; set; }
    public static bool ActiveCancelable { get; set; }
}

public static class FakeScreens
{
    public static CrystalSphereMinigame? CrystalSphere { get; set; }
}

public sealed class PendingRewards(RewardsSet set, IReadOnlySet<Reward> attempted)
{
    public RewardsSet Set { get; } = set;
    public List<Reward> Available => Set.Rewards
        .Where(r => !r.SuccessfullySelected && !attempted.Contains(r))
        .Where(r => r is not PotionReward || Set.Player.HasOpenPotionSlots)
        .ToList();
    public bool IsTerminal => Set.Room is CombatRoom;
}

public static class RewardsFlow
{
    public static PendingRewards? Pending { get; set; }
    public static RewardsSet? ActiveSet { get; set; }
    public static Reward? ActiveChoice { get; set; }
    public static IReadOnlySet<Reward>? ActiveAttempted { get; set; }
    public static bool AutoGold = true;
    public static bool AutoRelics = true;

    private static HashSet<string>? _noDownside;

    public static void LoadNoDownside(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                _noDownside = File.ReadAllLines(path)
                    .Select(l => l.Trim())
                    .Where(l => l.Length > 0 && !l.StartsWith('#'))
                    .ToHashSet(StringComparer.Ordinal);
                return;
            }
        }
        catch (Exception ex)
        {
            Log.Error("[NecroPilot] failed to read no_downside_relics.txt: " + ex.Message);
        }
        _noDownside = new HashSet<string>(StringComparer.Ordinal);
    }

    public static HashSet<string> NoDownside => _noDownside ??= new HashSet<string>(StringComparer.Ordinal);
    public static bool IsNoDownside(RelicModel? r) => r != null && NoDownside.Contains(r.Id.Entry);

    public static IEnumerable<Reward> Unclaimed(RewardsSet set, IReadOnlySet<Reward>? attempted, Reward? except) =>
        set.Rewards.Where(r => !r.SuccessfullySelected && !ReferenceEquals(r, except) &&
                               (attempted == null || !attempted.Contains(r)));

    public static void Clear()
    {
        Pending = null;
        ActiveSet = null;
        ActiveChoice = null;
        ActiveAttempted = null;
    }
}

public static class UiFlows
{
    public static int CombatsWon { get; internal set; }
}

public sealed class RunSession
{
    public static RunState State => RunManager.Instance.DebugOnlyGetState()!;
    public bool CombatOnly => false;
    public bool IsVictory => false;
    public int Steps { get; set; }
    internal PendingCardChoice? Selecting { get; set; }
    internal IReadOnlyList<int> SelectedIndices { get; set; } = Array.Empty<int>();

    internal static (int actual, int nominal, int extra) RestHealInfo(Player p)
    {
        var cr = p.Creature;
        int nominal = (int)HealRestSiteOption.GetHealAmount(p);
        int actual = Math.Clamp(nominal, 0, Math.Max(0, cr.MaxHp - cr.CurrentHp));
        int extra = 0;
        foreach (var m in p.RunState.IterateHookListeners(null))
        {
            int n = m.ModifyExtraRestSiteHealText(p, Array.Empty<MegaCrit.Sts2.Core.Localization.LocString>()).Count;
            if (n > 0 && m.ModifyRestSiteHealAmount(cr, 100m) == 100m) extra += n;
        }
        return (actual, nominal, extra);
    }
}

public static class RunSessionMap
{
    public static IEnumerable<MapPoint> Travelable(RunState s)
    {
        var map = s.Map;
        if (s.VisitedMapCoords.Count == 0) return new[] { map.StartingMapPoint };
        MapCoord last = s.VisitedMapCoords[^1];
        if (map.SecondBossMapPoint != null && last == map.BossMapPoint.coord) return new[] { map.SecondBossMapPoint };
        if (last.row == map.GetRowCount() - 1) return new[] { map.BossMapPoint };
        var point = map.GetPoint(last);
        return point == null ? new[] { map.StartingMapPoint } : MapTravel.GetTravelablePointsFrom(s, point);
    }
}

/// <summary>
/// Installs only non-invasive observation hooks needed for 100% feature parity with v0.114.0 training:
/// 1) Observation.RawPreview (active only while Observation.RawDamagePreview is true during encoding)
/// 2) Observation.DrawTop (passive tracker for known top-of-draw-pile cards)
/// 3) CardSelectCmd / PotionModel.OnUseWrapper metadata capture (prompt/method/source/enchantment)
/// 4) CombatManager.CombatWon counter
/// </summary>
public static class LiveHooks
{
    private static bool _installed;

    public static void Install()
    {
        if (_installed) return;
        _installed = true;
        try
        {
            var h = new Harmony("necropilot.v2.obs");
            Observation.RawPreview.Install(h);
            Observation.DrawTop.Install(h);

            // Track active CardSelectCmd metadata for accurate TokenKind.SelectPrompt encoding
            var setInfo = new HarmonyMethod(typeof(LiveHooks), nameof(OnCardSelectStart));
            var wrapSelect = new HarmonyMethod(typeof(LiveHooks), nameof(OnCardSelectWrap));
            foreach (var m in typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static))
            {
                if (m.ContainsGenericParameters) continue;
                var ps = m.GetParameters();
                if (!ps.Any(p => p.ParameterType == typeof(CardSelectorPrefs)) &&
                    m.Name != nameof(CardSelectCmd.FromChooseACardScreen))
                    continue;
                try
                {
                    if (typeof(Task).IsAssignableFrom(m.ReturnType))
                        h.Patch(m, prefix: setInfo, postfix: wrapSelect);
                    else
                        h.Patch(m, prefix: setInfo);
                }
                catch (Exception ex)
                {
                    Log.Error($"[NecroPilot] optional patch skipped for CardSelectCmd.{m.Name}: {ex.Message}");
                }
            }

            // Track active potion use for discovery-potion selection prompts
            var onUse = GameCompat.Method(typeof(PotionModel), nameof(PotionModel.OnUseWrapper), "live potion source tracker");
            if (onUse != null)
            {
                h.Patch(onUse,
                    prefix: new HarmonyMethod(typeof(LiveHooks), nameof(OnPotionStart)),
                    postfix: new HarmonyMethod(typeof(LiveHooks), nameof(OnPotionWrap)));
            }

            CombatManager.Instance.CombatWon += _ => UiFlows.CombatsWon++;
            Log.Info("[NecroPilot] observation hooks installed (RawPreview, DrawTop, CardSelectCmd, Potion).");
        }
        catch (Exception ex)
        {
            Log.Error("[NecroPilot] hook installation error: " + ex);
        }
    }

    private static void OnCardSelectStart(MethodBase __originalMethod, object[] __args)
    {
        var ps = __originalMethod.GetParameters();
        CardSelectorPrefs prefs = default;
        bool hasPrefs = false;
        foreach (var arg in __args)
        {
            if (arg is CardSelectorPrefs p) { prefs = p; hasPrefs = true; break; }
        }
        int si = Array.FindIndex(ps, q => q.Name == "source");
        var source = si >= 0 && si < __args.Length ? __args[si] as AbstractModel : null;
        var enchant = __args.OfType<EnchantmentModel>().FirstOrDefault();
        AgentCardSelector.ActiveCancelable = hasPrefs && prefs.Cancelable;
        AgentCardSelector.ActiveSelectInfo = new SelectInfo(
            hasPrefs ? prefs.Prompt?.LocTable : null,
            hasPrefs ? prefs.Prompt?.LocEntryKey : null,
            __originalMethod.Name,
            enchant,
            source);
    }

    private static void OnCardSelectWrap(object? __result)
    {
        var captured = AgentCardSelector.ActiveSelectInfo;
        if (__result is not Task task || task.IsCompleted)
        {
            ClearSelect(captured);
            return;
        }
        task.ContinueWith(_ => ClearSelect(captured), TaskScheduler.Default);
    }

    private static void ClearSelect(SelectInfo? captured)
    {
        if (ReferenceEquals(AgentCardSelector.ActiveSelectInfo, captured))
        {
            AgentCardSelector.ActiveSelectInfo = null;
            AgentCardSelector.ActiveCancelable = false;
        }
    }

    private static void OnPotionStart(PotionModel __instance) => AgentCardSelector.UsingPotion = __instance;

    private static void OnPotionWrap(PotionModel __instance, object? __result)
    {
        if (__result is not Task task || task.IsCompleted)
        {
            if (ReferenceEquals(AgentCardSelector.UsingPotion, __instance))
                AgentCardSelector.UsingPotion = null;
            return;
        }
        task.ContinueWith(_ =>
        {
            if (ReferenceEquals(AgentCardSelector.UsingPotion, __instance))
                AgentCardSelector.UsingPotion = null;
        }, TaskScheduler.Default);
    }
}
