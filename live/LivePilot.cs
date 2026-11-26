using System.Reflection;
using System.Runtime.CompilerServices;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using Environment = System.Environment;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.Events.Custom;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace NecroPilot;

/// <summary>
/// Detects the live game's active decision screen and builds a Sts2Headless.Decision whose
/// ActionKind / A / B / Key / F0..F2 / Shown / Locked layout matches RunSession.cs 1:1.
/// Every action executes through the live game's UI controls or synchronizers.
/// </summary>
internal sealed class LivePilot
{
    internal sealed record Pending(
        Decision Decision,
        string Context,
        bool AutoStep = false,
        bool Repeatable = false,
        int CooldownMs = 120);

    private static readonly BindingFlags Priv = BindingFlags.NonPublic | BindingFlags.Instance;
    private static readonly BindingFlags AnyInst = BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance;

    private static readonly FieldInfo? CardRewardOptions = typeof(NCardRewardSelectionScreen).GetField("_options", Priv);
    private static readonly FieldInfo? CardRewardExtras = typeof(NCardRewardSelectionScreen).GetField("_extraOptions", Priv);
    private static readonly FieldInfo? CardRewardAltContainer = typeof(NCardRewardSelectionScreen).GetField("_rewardAlternativesContainer", Priv);
    private static readonly FieldInfo? BundleList = typeof(NChooseABundleSelectionScreen).GetField("_bundles", Priv);
    private static readonly FieldInfo? GridCards = typeof(NCardGridSelectionScreen).GetField("_cards", Priv);
    private static readonly FieldInfo? GridControl = typeof(NCardGridSelectionScreen).GetField("_grid", Priv);
    private static readonly MethodInfo? GridClick = typeof(NCardGridSelectionScreen).GetMethod("OnCardClicked", Priv);
    private static readonly FieldInfo? ChooseCards = typeof(NChooseACardSelectionScreen).GetField("_cards", Priv);
    private static readonly FieldInfo? ChooseCanSkip = typeof(NChooseACardSelectionScreen).GetField("_canSkip", Priv);
    private static readonly FieldInfo? ChooseSkip = typeof(NChooseACardSelectionScreen).GetField("_skipButton", Priv);
    private static readonly FieldInfo? HandPrefs = typeof(NPlayerHand).GetField("_prefs", Priv);
    private static readonly FieldInfo? HandSelected = typeof(NPlayerHand).GetField("_selectedCards", Priv);
    private static readonly FieldInfo? HandFilter = typeof(NPlayerHand).GetField("_currentSelectionFilter", Priv);
    private static readonly FieldInfo? HandConfirm = typeof(NPlayerHand).GetField("_selectModeConfirmButton", Priv);
    private static readonly MethodInfo? HandPress = typeof(NPlayerHand).GetMethod("OnHolderPressed", Priv);

    private static readonly FieldInfo? CrystalMinigameField = typeof(NCrystalSphereScreen)
        .GetFields(AnyInst)
        .FirstOrDefault(f => typeof(CrystalSphereMinigame).IsAssignableFrom(f.FieldType));

    private static readonly FieldInfo? RewardsSetField = typeof(NRewardsScreen)
        .GetFields(AnyInst)
        .FirstOrDefault(f => typeof(RewardsSet).IsAssignableFrom(f.FieldType));

    private static readonly HashSet<string> ExcludedEventOptions = new(StringComparer.Ordinal)
    {
        "TRIAL.pages.REJECT.options.DOUBLE_DOWN",
    };

    private readonly RunSession _session = new();
    internal RunSession Session => _session;

    // Track per-screen reward clicks so skipped card rewards don't loop forever
    private readonly HashSet<Reward> _triedRewards = new(ReferenceEqualityComparer.Instance);
    private NRewardsScreen? _triedScreen;
    private RewardsSet? _lastRewardsSet;
    private Reward? _lastOpenedReward;

    // Used card-reward alternatives (e.g. REROLL once per reward)
    private readonly ConditionalWeakTable<CardRewardAlternative, object> _usedAlts = new();

    // Settling tracker that properly resets when NPlayerHand leaves and re-enters card selection
    private GodotObject? _settlingObj;
    private bool _handWasSelecting;
    private long _settlingSinceMs;

    // Shop / room open & purchase guards to avoid double-buying or race-condition stalls
    private long _shopOpenedAtMs;
    private long _shopBusyUntilMs;
    private Task? _pendingShopTask;
    private AbstractRoom? _lastRoom;
    private readonly HashSet<object> _failedShopEntries = new(ReferenceEqualityComparer.Instance);

    internal object DebugState() => new
    {
        steps = _session.Steps,
        tried_rewards = _triedRewards.Count,
        failed_shop_entries = _failedShopEntries.Count,
        pending_shop_task = _pendingShopTask?.Status.ToString(),
        shop_busy_ms_left = Math.Max(0, _shopBusyUntilMs - Environment.TickCount64),
        hand_was_selecting = _handWasSelecting,
        settling_obj = _settlingObj?.GetType().Name,
    };

    internal void ResetRunTracking()
    {
        _triedRewards.Clear();
        _triedScreen = null;
        _lastRewardsSet = null;
        _lastOpenedReward = null;
        _settlingObj = null;
        _handWasSelecting = false;
        _shopOpenedAtMs = 0;
        _shopBusyUntilMs = 0;
        _pendingShopTask = null;
        _lastRoom = null;
        _failedShopEntries.Clear();
        _session.Steps = 0;
        _session.Selecting = null;
        _session.SelectedIndices = Array.Empty<int>();
        FakeScreens.CrystalSphere = null;
        RewardsFlow.Clear();
    }

    internal Pending? Detect(RunState state, Player player, out string waitingFor)
    {
        waitingFor = "";
        _session.Selecting = null;
        _session.SelectedIndices = Array.Empty<int>();
        FakeScreens.CrystalSphere = null;
        RewardsFlow.Pending = null;

        if (!ReferenceEquals(state.CurrentRoom, _lastRoom))
        {
            _lastRoom = state.CurrentRoom;
            _failedShopEntries.Clear();
            _pendingShopTask = null;
            _shopBusyUntilMs = 0;
        }

        if (NGame.Instance?.Transition is { InTransition: true })
        {
            waitingFor = "Scene transition";
            return null;
        }

        // Track NPlayerHand selection state edge even while overlays are up
        bool handSelectingNow = CombatManager.Instance.IsInProgress &&
                                NCombatRoom.Instance?.Ui?.Hand is { IsInCardSelection: true };
        if (!handSelectingNow) _handWasSelecting = false;

        var overlay = NOverlayStack.Instance?.ScreenCount > 0 ? NOverlayStack.Instance.Peek() : null;

        // If the map is open on top of a finished overlay and accepts travel, map takes priority
        if (overlay != null && NMapScreen.Instance is { IsOpen: true, IsTravelEnabled: true } mapOver && mapOver.IsVisibleInTree())
        {
            RewardsFlow.Clear();
            return MapScreen(state, player, mapOver, out waitingFor);
        }

        switch (overlay)
        {
            case NGameOverScreen:
                waitingFor = "run over";
                return null;
            case NCardRewardSelectionScreen crs:
                return CardRewardScreen(state, player, crs, out waitingFor);
            case NCardGridSelectionScreen grid:
                waitingFor = "card selection: " + grid.GetType().Name;
                return Settled(grid, 550) ? GridSelectScreen(state, player, grid) : null;
            case NChooseACardSelectionScreen choose:
                waitingFor = "card selection: ChooseACard";
                return Settled(choose, 550) ? ChooseACardScreen(state, player, choose) : null;
            case NRewardsScreen rs:
                return RewardsScreen(state, player, rs, out waitingFor);
            case NChooseABundleSelectionScreen bs:
                waitingFor = "card bundle";
                return Settled(bs, 450) ? BundleScreen(state, player, bs) : null;
            case NCrystalSphereScreen cs:
                return CrystalSphereScreen(cs, out waitingFor);
            case Node n:
                waitingFor = "your choice: " + n.Name;
                return null;
        }

        // Not on rewards or card reward anymore
        RewardsFlow.Clear();

        if (NMapScreen.Instance is { IsOpen: true } map)
            return MapScreen(state, player, map, out waitingFor);

        if (CombatManager.Instance.IsInProgress)
        {
            if (NCombatRoom.Instance?.Ui?.Hand is { IsInCardSelection: true } hand)
            {
                waitingFor = "card selection: Hand";
                bool settled = SettledHand(hand, 450);
                return settled ? HandSelect(state, player, hand) : null;
            }
            if (!AwaitingCombatInput(player))
            {
                waitingFor = "Combat animations / enemy turn";
                return null;
            }
            return Combat(state, player);
        }

        var room = state.CurrentRoom;
        if (room is RestSiteRoom rest)
        {
            waitingFor = "Rest site controls";
            return Rest(state, player, rest);
        }

        var roomNode = ((SceneTree)Engine.GetMainLoop()).Root
            .GetNodeOrNull("/root/Game/RootSceneContainer/Run/RoomContainer");
        if (roomNode == null)
        {
            waitingFor = "Room loading";
            return null;
        }

        return room switch
        {
            EventRoom er => Event(state, player, er, roomNode, out waitingFor),
            MerchantRoom mr => Shop(state, player, mr, roomNode, out waitingFor),
            TreasureRoom => Treasure(state, player, roomNode, out waitingFor),
            _ => null,
        };
    }

    // ------------------------------------------------------------------ Settling helpers

    private bool Settled(GodotObject obj, long delayMs)
    {
        long now = Environment.TickCount64;
        if (!ReferenceEquals(obj, _settlingObj))
        {
            _settlingObj = obj;
            _settlingSinceMs = now;
            return false;
        }
        return now - _settlingSinceMs >= delayMs;
    }

    private bool SettledHand(NPlayerHand hand, long delayMs)
    {
        long now = Environment.TickCount64;
        if (!_handWasSelecting || !ReferenceEquals(hand, _settlingObj))
        {
            _handWasSelecting = true;
            _settlingObj = hand;
            _settlingSinceMs = now;
            return false;
        }
        return now - _settlingSinceMs >= delayMs;
    }

    // ------------------------------------------------------------------ Combat

    private static bool AwaitingCombatInput(Player player)
    {
        var cm = CombatManager.Instance;
        var rm = RunManager.Instance;
        var pcs = player.PlayerCombatState;
        if (!(cm.IsInProgress && !cm.IsOverOrEnding && pcs != null && pcs.Phase == PlayerTurnPhase.Play &&
              !cm.PlayerActionsDisabled && !cm.IsPlayerReadyToEndTurn(player)))
            return false;
        return rm.ActionQueueSet.IsEmpty && !rm.ActionExecutor.IsRunning;
    }

    private Pending? Combat(RunState state, Player player)
    {
        var pcs = player.PlayerCombatState;
        var cs = player.Creature.CombatState;
        if (pcs == null || cs == null) return null;

        var d = new Decision(DecisionKind.Combat) { Context = $"Turn {pcs.TurnNumber}" };
        var targets = new List<Creature?> { null };
        targets.AddRange(cs.Creatures.Where(c => c.IsAlive));

        foreach (var card in pcs.Hand.Cards)
        {
            bool playable;
            try { playable = card.CanPlay(out _, out _); } catch { playable = false; }
            if (!playable) continue;
            foreach (var t in targets)
            {
                bool ok;
                try { ok = card.CanPlayTargeting(t); } catch { ok = false; }
                if (!ok) continue;
                var target = t;
                var c = card;
                d.Add(ActionKind.PlayCard, $"Play {CardName(c)}{TargetLabel(target)}", () =>
                {
                    c.TryManualPlay(target);
                    return Task.CompletedTask;
                }, a: c, b: target);
            }
        }

        foreach (var potion in UsablePotions(player, inCombat: true))
        {
            foreach (var t in targets)
            {
                bool ok;
                try { ok = potion.IsValidTarget(t); } catch { ok = false; }
                if (!ok) continue;
                var target = t;
                var p = potion;
                d.Add(ActionKind.UsePotion, $"Use {p.Id.Entry}{TargetLabel(target)}", () =>
                {
                    p.EnqueueManualUse(target);
                    return Task.CompletedTask;
                }, a: p, b: target);
            }
        }

        int turn = pcs.TurnNumber;
        d.Add(ActionKind.EndTurn, "End Turn", () =>
        {
            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(player, turn));
            return Task.CompletedTask;
        });

        return new Pending(d, "combat", CooldownMs: 140);
    }

    // ------------------------------------------------------------------ Potions (shared)

    private static IEnumerable<PotionModel> UsablePotions(Player player, bool inCombat)
    {
        if (!player.CanUseOrRemovePotions || player.Creature.IsDead) yield break;
        foreach (var p in player.Potions)
        {
            if (p.IsQueued) continue;
            bool passes;
            try { passes = p.PassesCustomUsabilityCheck; } catch { passes = false; }
            if (!passes) continue;
            if (p.Usage == PotionUsage.AnyTime || (inCombat && p.Usage == PotionUsage.CombatOnly))
                yield return p;
        }
    }

    private static void AddOutOfCombatPotions(Player player, Decision d)
    {
        foreach (var potion in UsablePotions(player, inCombat: false))
        {
            bool valid;
            try { valid = potion.IsValidTarget(player.Creature) || potion.IsValidTarget(null); } catch { valid = false; }
            if (!valid) continue;
            var p = potion;
            d.Add(ActionKind.UsePotion, $"Use {p.Id.Entry}", () =>
            {
                p.EnqueueManualUse(null);
                return Task.CompletedTask;
            }, a: p);
        }
    }

    private static void AddPotionDiscards(Player player, Decision d)
    {
        if (!player.CanUseOrRemovePotions) return;
        var slots = player.PotionSlots;
        for (int i = 0; i < slots.Count; i++)
        {
            if (slots[i] is not { IsQueued: false } potion) continue;
            uint slot = (uint)i;
            var p = potion;
            d.Add(ActionKind.DiscardPotion, $"Discard {p.Id.Entry}", () =>
            {
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                    new DiscardPotionGameAction(player, slot, CombatManager.Instance.IsInProgress));
                return Task.CompletedTask;
            }, a: p);
        }
    }

    // ------------------------------------------------------------------ Card Reward

    private Pending? CardRewardScreen(RunState state, Player player, NCardRewardSelectionScreen screen, out string waitingFor)
    {
        waitingFor = "Card reward";
        if (CardRewardOptions?.GetValue(screen) is not IReadOnlyList<CardCreationResult> options || options.Count == 0)
            return null;
        var extras = CardRewardExtras?.GetValue(screen) as IReadOnlyList<CardRewardAlternative>
                     ?? Array.Empty<CardRewardAlternative>();

        // Expose sibling unclaimed rewards (Observation v18 parity)
        if (_lastRewardsSet != null)
        {
            RewardsFlow.ActiveSet = _lastRewardsSet;
            RewardsFlow.ActiveChoice = _lastOpenedReward;
            RewardsFlow.ActiveAttempted = _triedRewards;
        }

        var d = new Decision(DecisionKind.CardReward) { Context = "Card Reward" };
        if (RewardsFlow.ActiveSet is { } set)
        {
            foreach (var r in RewardsFlow.Unclaimed(set, RewardsFlow.ActiveAttempted, RewardsFlow.ActiveChoice))
            {
                d.Shown.Add((r, Zone.Offer, 0));
                if (r is CardReward other)
                {
                    var z = Observation.RewardZone(set, other);
                    foreach (var c in other.Cards) d.Shown.Add((c, z, 0));
                }
            }
        }

        foreach (var opt in options)
        {
            var card = opt.Card;
            d.Add(ActionKind.PickCardReward, $"Take {CardName(card)}", () =>
            {
                var holder = screen.GetCardHolder(card);
                if (holder != null && GodotObject.IsInstanceValid(holder))
                    holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
                return Task.CompletedTask;
            }, a: card);
        }

        var altContainer = CardRewardAltContainer?.GetValue(screen) as Node;
        var altButtons = altContainer?.GetChildren().OfType<NClickableControl>().ToList() ?? new List<NClickableControl>();
        for (int i = 0; i < extras.Count; i++)
        {
            var alt = extras[i];
            if (_usedAlts.TryGetValue(alt, out _)) continue;
            int idx = i;
            d.Add(ActionKind.CardRewardAlt, alt.OptionId, () =>
            {
                if (alt.OptionId == "REROLL") _usedAlts.AddOrUpdate(alt, alt);
                if (idx >= 0 && idx < altButtons.Count && GodotObject.IsInstanceValid(altButtons[idx]))
                    altButtons[idx].ForceClick();
                return Task.CompletedTask;
            }, key: Vocab.Alt(alt.OptionId));
        }

        if (extras.Count == 0 && altButtons.Count > 0)
        {
            d.Add(ActionKind.CardRewardAlt, "SKIP", () =>
            {
                if (GodotObject.IsInstanceValid(altButtons[0])) altButtons[0].ForceClick();
                return Task.CompletedTask;
            }, key: Vocab.Alt("SKIP"));
        }

        return d.Choices.Count > 0 ? new Pending(d, "card reward", CooldownMs: 250) : null;
    }

    // ------------------------------------------------------------------ Bundle Select

    private Pending? BundleScreen(RunState state, Player player, NChooseABundleSelectionScreen screen)
    {
        if (BundleList?.GetValue(screen) is not IReadOnlyList<IReadOnlyList<CardModel>> bundles || bundles.Count == 0)
            return null;
        var nodes = FindAll<NCardBundle>(screen);
        var d = new Decision(DecisionKind.Bundle) { Context = "Choose Bundle" };
        for (int i = 0; i < bundles.Count; i++)
        {
            int k = i;
            foreach (var c in bundles[i]) d.Shown.Add((c, i == 0 ? Zone.Pack1 : Zone.Pack2, 0));
            d.Add(ActionKind.PickBundle, "Bundle " + string.Join("+", bundles[i].Select(CardName)), async () =>
            {
                if (k >= 0 && k < nodes.Count && GodotObject.IsInstanceValid(nodes[k]))
                    nodes[k].Hitbox.ForceClick();
                int gen = Live.Generation;
                await Task.Delay(450);
                if (!Live.Active(gen) || !GodotObject.IsInstanceValid(screen)) return;
                FindFirst<NConfirmButton>(screen)?.ForceClick();
            }, a: bundles[i].FirstOrDefault(), f0: i);
        }
        return d.Choices.Count > 0 ? new Pending(d, "card bundle", CooldownMs: 300) : null;
    }

    // ------------------------------------------------------------------ Rewards Screen

    private Pending? RewardsScreen(RunState state, Player player, NRewardsScreen screen, out string waitingFor)
    {
        waitingFor = "Rewards screen";
        if (!ReferenceEquals(screen, _triedScreen))
        {
            _triedScreen = screen;
            _triedRewards.Clear();
            _lastOpenedReward = null;
            _lastRewardsSet = RewardsSetField?.GetValue(screen) as RewardsSet;
        }

        var buttons = FindAll<NRewardButton>(screen)
            .Where(b => b.IsEnabled && b.IsVisibleInTree() && b.Reward != null && !_triedRewards.Contains(b.Reward))
            .ToList();

        // Auto-claim gold (matching training RewardsFlow.AutoGold)
        if (RewardsFlow.AutoGold && buttons.FirstOrDefault(b => b.Reward is GoldReward) is { } goldBtn)
        {
            var auto = new Decision(DecisionKind.Rewards) { Context = "Claim Gold" };
            auto.Add(ActionKind.TakeReward, "Auto-claim Gold", () =>
            {
                _triedRewards.Add(goldBtn.Reward!);
                if (GodotObject.IsInstanceValid(goldBtn)) goldBtn.ForceClick();
                return Task.CompletedTask;
            }, a: goldBtn.Reward);
            return new Pending(auto, "claim gold", AutoStep: true, Repeatable: true, CooldownMs: 220);
        }

        // Auto-claim standalone no-downside relics (matching training RewardsFlow.AutoRelics)
        if (RewardsFlow.AutoRelics && buttons.FirstOrDefault(b =>
                b.Reward is RelicReward { ParentRewardSet: null } rr && RewardsFlow.IsNoDownside(rr.Relic)) is { } relicBtn)
        {
            var rr = (RelicReward)relicBtn.Reward!;
            var auto = new Decision(DecisionKind.Rewards) { Context = "Auto-claim Relic" };
            auto.Add(ActionKind.TakeReward, $"Auto-take {rr.Relic?.Id.Entry}", () =>
            {
                _triedRewards.Add(rr);
                if (GodotObject.IsInstanceValid(relicBtn)) relicBtn.ForceClick();
                return Task.CompletedTask;
            }, a: rr);
            return new Pending(auto, "claim relic", AutoStep: true, Repeatable: true, CooldownMs: 280);
        }

        var proceed = FindFirst<NProceedButton>(screen);
        if (proceed is not { IsEnabled: true } || !proceed.IsVisibleInTree()) proceed = null;

        var set = _lastRewardsSet;
        if (set != null) RewardsFlow.Pending = new PendingRewards(set, _triedRewards);

        var openButtons = buttons
            .Where(b => b.Reward is not PotionReward || player.HasOpenPotionSlots)
            .ToList();

        var d = new Decision(DecisionKind.Rewards) { Context = "Rewards" };
        foreach (var btn in openButtons)
        {
            var r = btn.Reward!;
            var capturedBtn = btn;
            d.Add(ActionKind.TakeReward, "Take " + RewardLabel(r), () =>
            {
                _triedRewards.Add(r);
                _lastOpenedReward = r;
                if (GodotObject.IsInstanceValid(capturedBtn)) capturedBtn.ForceClick();
                return Task.CompletedTask;
            }, a: r);
            if (r is CardReward cr && set != null)
            {
                var z = Observation.RewardZone(set, cr);
                foreach (var c in cr.Cards) d.Shown.Add((c, z, 0));
            }
        }

        if (proceed != null)
        {
            var capturedProceed = proceed;
            d.Add(ActionKind.Leave, "Proceed", () =>
            {
                if (GodotObject.IsInstanceValid(capturedProceed)) capturedProceed.ForceClick();
                return Task.CompletedTask;
            });
        }

        AddOutOfCombatPotions(player, d);

        // If belt is full and there's an unclaimed potion reward, allow discarding a potion
        var waitingPotions = buttons.Select(b => b.Reward).OfType<PotionReward>()
            .Where(p => p.Potion != null && !p.SuccessfullySelected).ToList();
        if (!player.HasOpenPotionSlots && waitingPotions.Count > 0)
        {
            AddPotionDiscards(player, d);
            foreach (var pr in waitingPotions) d.Shown.Add((pr, Zone.Offer, 0));
        }

        if (d.Choices.Count == 0) return null;
        bool onlyProceed = d.Choices.Count == 1 && d.Choices[0].Kind == ActionKind.Leave;
        return new Pending(d, onlyProceed ? "rewards: proceed" : "rewards", AutoStep: onlyProceed, CooldownMs: 260);
    }

    // ------------------------------------------------------------------ Card Selection (Grid / Choose / Hand)

    private Pending? GridSelectScreen(RunState state, Player player, NCardGridSelectionScreen screen)
    {
        IReadOnlyList<CardModel>? cards = screen is NCombatPileCardSelectScreen
            ? (GridControl?.GetValue(screen) as NCardGrid)?.CurrentlyDisplayedCards.ToArray()
            : GridCards?.GetValue(screen) as IReadOnlyList<CardModel>;
        if (cards is not { Count: > 0 }) return null;

        var t = screen.GetType();
        var pickedList = (AccessTools.Field(t, "_selectedCards")?.GetValue(screen) as IEnumerable<CardModel>)?.ToList()
                         ?? new List<CardModel>();
        var prefs = AccessTools.Field(t, "_prefs")?.GetValue(screen) is CardSelectorPrefs p ? p : default;
        int max = Math.Min(Math.Max(prefs.MaxSelect, 1), cards.Count);
        int min = Math.Min(prefs.MinSelect, max);

        var info = AgentCardSelector.ActiveSelectInfo ?? InferSelectInfo(screen, prefs);
        bool upgradeAny = screen is NDeckUpgradeSelectScreen || (info.Method?.Contains("ForUpgrade") == true);
        if (info.Method == "FromDeckForUpgrade" && max > 0) min = Math.Max(min, 1);

        var pickedIndices = Enumerable.Range(0, cards.Count).Where(i => pickedList.Contains(cards[i])).ToList();
        _session.Selecting = new PendingCardChoice(cards, min, max, prefs.Cancelable, info);
        _session.SelectedIndices = pickedIndices;

        var preview = SelectionControls.Preview(screen);
        var confirm = preview.IsOpen ? preview.Button : SelectionControls.GridConfirm(screen);

        if (preview.IsOpen || pickedList.Count >= max)
        {
            if (confirm == null || pickedList.Count < min) return null;
            var auto = new Decision(DecisionKind.CardSelect) { Context = "Confirm selection", Selected = pickedList.Count };
            auto.Add(ActionKind.ConfirmSelection, $"Confirm ({pickedList.Count})", () =>
            {
                if (GodotObject.IsInstanceValid(confirm)) confirm.ForceClick();
                return Task.CompletedTask;
            });
            return new Pending(auto, "card select: confirm", AutoStep: true, Repeatable: true, CooldownMs: 240);
        }

        var d = new Decision(DecisionKind.CardSelect)
        {
            Context = $"Select {min}..{max} ({pickedList.Count} picked)",
            Selected = pickedList.Count,
        };
        for (int i = 0; i < cards.Count; i++)
        {
            var card = cards[i];
            if (pickedList.Contains(card)) continue;
            var c = card;
            CardModel? previewUp = null;
            if (upgradeAny)
            {
                try { previewUp = Observation.UpgradedCopy(c); } catch { }
            }
            d.Add(ActionKind.SelectCard, $"Select {CardName(c)}", () =>
            {
                if (GodotObject.IsInstanceValid(screen))
                    GridClick?.Invoke(screen, new object[] { c });
                return Task.CompletedTask;
            }, a: c, b: previewUp);
        }

        if (pickedList.Count >= min && confirm != null)
        {
            var capturedConfirm = confirm;
            d.Add(ActionKind.ConfirmSelection, $"Confirm ({pickedList.Count})", () =>
            {
                if (GodotObject.IsInstanceValid(capturedConfirm)) capturedConfirm.ForceClick();
                return Task.CompletedTask;
            });
        }

        return d.Choices.Count > 0 ? new Pending(d, "card select (" + t.Name + ")", CooldownMs: 240) : null;
    }

    private Pending? ChooseACardScreen(RunState state, Player player, NChooseACardSelectionScreen screen)
    {
        if (ChooseCards?.GetValue(screen) is not IReadOnlyList<CardModel> cards || cards.Count == 0)
            return null;
        bool canSkip = ChooseCanSkip?.GetValue(screen) is true;
        int min = canSkip ? 0 : 1;
        var info = AgentCardSelector.ActiveSelectInfo ?? new SelectInfo(null, null, "FromChooseACardScreen", null);
        _session.Selecting = new PendingCardChoice(cards, min, 1, false, info);
        _session.SelectedIndices = Array.Empty<int>();

        var d = new Decision(DecisionKind.CardSelect) { Context = "Choose a card", Selected = 0 };
        foreach (var card in cards)
        {
            var c = card;
            d.Add(ActionKind.SelectCard, $"Select {CardName(c)}", () =>
            {
                if (GodotObject.IsInstanceValid(screen) &&
                    FindAll<NCardHolder>(screen).FirstOrDefault(h => ReferenceEquals(h.CardModel, c)) is { } holder)
                    holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);
                return Task.CompletedTask;
            }, a: c);
        }

        if (canSkip && ChooseSkip?.GetValue(screen) is NClickableControl skipBtn && skipBtn.IsEnabled)
        {
            d.Add(ActionKind.ConfirmSelection, "Skip", () =>
            {
                if (GodotObject.IsInstanceValid(skipBtn)) skipBtn.ForceClick();
                return Task.CompletedTask;
            });
        }

        return d.Choices.Count > 0 ? new Pending(d, "choose a card", CooldownMs: 260) : null;
    }

    private Pending? HandSelect(RunState state, Player player, NPlayerHand hand)
    {
        var pcs = player.PlayerCombatState;
        if (pcs == null) return null;
        var prefs = HandPrefs?.GetValue(hand) is CardSelectorPrefs p ? p : default;
        var pickedList = (HandSelected?.GetValue(hand) as IEnumerable<CardModel>)?.ToList() ?? new List<CardModel>();
        var filter = HandFilter?.GetValue(hand) as Func<CardModel, bool>;
        var options = pcs.Hand.Cards.Where(c => filter == null || filter(c)).ToList();
        if (options.Count == 0) return null;

        int max = Math.Min(Math.Max(prefs.MaxSelect, 1), options.Count);
        int min = Math.Min(prefs.MinSelect, max);
        var info = AgentCardSelector.ActiveSelectInfo ??
                   new SelectInfo(prefs.Prompt?.LocTable, prefs.Prompt?.LocEntryKey, "FromHand", null);
        bool upgradeAny = info.Method?.Contains("ForUpgrade") == true;

        var pickedIndices = Enumerable.Range(0, options.Count).Where(i => pickedList.Contains(options[i])).ToList();
        _session.Selecting = new PendingCardChoice(options, min, max, prefs.Cancelable, info);
        _session.SelectedIndices = pickedIndices;

        var confirm = HandConfirm?.GetValue(hand) as NConfirmButton;
        bool confirmReady = confirm is { IsEnabled: true } && confirm.IsVisibleInTree();

        if (pickedList.Count >= max)
        {
            if (!confirmReady) return null;
            var auto = new Decision(DecisionKind.CardSelect) { Context = "Confirm hand selection", Selected = pickedList.Count };
            auto.Add(ActionKind.ConfirmSelection, $"Confirm ({pickedList.Count})", () =>
            {
                if (GodotObject.IsInstanceValid(confirm)) confirm!.ForceClick();
                return Task.CompletedTask;
            });
            return new Pending(auto, "hand select: confirm", AutoStep: true, Repeatable: true, CooldownMs: 220);
        }

        var holders = FindAll<NHandCardHolder>(hand);
        var d = new Decision(DecisionKind.CardSelect)
        {
            Context = $"Hand select {min}..{max} ({pickedList.Count})",
            Selected = pickedList.Count,
        };
        foreach (var card in options)
        {
            if (pickedList.Contains(card)) continue;
            var c = card;
            CardModel? previewUp = null;
            if (upgradeAny)
            {
                try { previewUp = Observation.UpgradedCopy(c); } catch { }
            }
            d.Add(ActionKind.SelectCard, $"Select {CardName(c)}", () =>
            {
                if (GodotObject.IsInstanceValid(hand) &&
                    holders.FirstOrDefault(h => GodotObject.IsInstanceValid(h) && ReferenceEquals(h.CardModel, c)) is { } holder)
                    HandPress?.Invoke(hand, new object[] { holder });
                return Task.CompletedTask;
            }, a: c, b: previewUp);
        }

        if (pickedList.Count >= min && confirmReady)
        {
            d.Add(ActionKind.ConfirmSelection, $"Confirm ({pickedList.Count})", () =>
            {
                if (GodotObject.IsInstanceValid(confirm)) confirm!.ForceClick();
                return Task.CompletedTask;
            });
        }

        return d.Choices.Count > 0 ? new Pending(d, "hand select", CooldownMs: 200) : null;
    }

    private static SelectInfo InferSelectInfo(NCardGridSelectionScreen screen, CardSelectorPrefs prefs)
    {
        string method = screen switch
        {
            NDeckUpgradeSelectScreen => "FromDeckForUpgrade",
            NDeckEnchantSelectScreen => "FromDeckForEnchant",
            NDeckTransformSelectScreen => "FromDeckForTransform",
            NCombatPileCardSelectScreen => "FromCombatPile",
            NDeckCardSelectScreen => "FromDeck",
            _ => "FromDeck",
        };
        return new SelectInfo(prefs.Prompt?.LocTable, prefs.Prompt?.LocEntryKey, method, null);
    }

    // ------------------------------------------------------------------ Crystal Sphere Minigame

    private Pending? CrystalSphereScreen(NCrystalSphereScreen screen, out string waitingFor)
    {
        waitingFor = "Crystal Sphere";
        var proceed = screen.GetNodeOrNull<NProceedButton>("%ProceedButton");
        if (proceed is { IsEnabled: true } && proceed.IsVisibleInTree())
        {
            var auto = new Decision(DecisionKind.Minigame) { Context = "Crystal Sphere Done" };
            auto.Add(ActionKind.Leave, "Proceed", () =>
            {
                if (GodotObject.IsInstanceValid(proceed)) proceed.ForceClick();
                return Task.CompletedTask;
            });
            return new Pending(auto, "crystal sphere: proceed", AutoStep: true, Repeatable: true, CooldownMs: 250);
        }

        var minigame = CrystalMinigameField?.GetValue(screen) as CrystalSphereMinigame;
        if (minigame == null || minigame.IsFinished) return null;
        FakeScreens.CrystalSphere = minigame;

        var cellsNode = screen.GetNodeOrNull<Control>("%Cells");
        if (cellsNode == null) return null;
        var cellMap = FindAll<NCrystalSphereCell>(cellsNode)
            .Where(c => c.Visible && c.Entity.IsHidden)
            .ToDictionary(c => (c.Entity.X, c.Entity.Y));
        if (cellMap.Count == 0) return null;

        var d = new Decision(DecisionKind.Minigame) { Context = $"Crystal Sphere ({minigame.DivinationCount} left)" };
        int w = minigame.GridSize.X, h = minigame.GridSize.Y;
        for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
        {
            if (!cellMap.TryGetValue((x, y), out var uiCell)) continue;
            var captured = uiCell;
            d.Add(ActionKind.MinigameCell, $"Small ({x},{y})", () =>
            {
                minigame.SetTool(CrystalSphereMinigame.CrystalSphereToolType.Small);
                if (GodotObject.IsInstanceValid(captured))
                    captured.EmitSignal(NClickableControl.SignalName.Released, captured);
                return Task.CompletedTask;
            }, f0: x, f1: y, f2: 0);
            d.Add(ActionKind.MinigameCell, $"Big ({x},{y})", () =>
            {
                minigame.SetTool(CrystalSphereMinigame.CrystalSphereToolType.Big);
                if (GodotObject.IsInstanceValid(captured))
                    captured.EmitSignal(NClickableControl.SignalName.Released, captured);
                return Task.CompletedTask;
            }, f0: x, f1: y, f2: 1);
        }

        return d.Choices.Count > 0 ? new Pending(d, "crystal sphere", CooldownMs: 320) : null;
    }

    // ------------------------------------------------------------------ Map

    private Pending? MapScreen(RunState state, Player player, NMapScreen map, out string waitingFor)
    {
        waitingFor = "Map";
        if (!map.IsVisibleInTree() || !map.IsTravelEnabled || map.IsTraveling)
        {
            waitingFor = "Map not ready";
            return null;
        }

        var next = RunSessionMap.Travelable(state).OrderBy(p => p.coord.col).ToList();
        if (next.Count == 0)
        {
            waitingFor = "Act transition";
            return null;
        }

        var points = FindAll<NMapPoint>(map);
        NMapPoint? NodeFor(MapPoint mp) =>
            points.FirstOrDefault(p => ReferenceEquals(p.Point, mp) || p.Point.coord.Equals(mp.coord));

        if (next.Any(mp => NodeFor(mp) is not { IsEnabled: true }))
        {
            waitingFor = "Map nodes enabling";
            return null;
        }

        var d = new Decision(DecisionKind.Map)
        {
            Context = $"Act {state.CurrentActIndex + 1} Map",
        };
        foreach (var mp in next)
        {
            var uiNode = NodeFor(mp)!;
            d.Add(ActionKind.MapMove, $"→ ({mp.coord.col},{mp.coord.row}) {mp.PointType}", () =>
            {
                if (GodotObject.IsInstanceValid(uiNode)) uiNode.ForceClick();
                return Task.CompletedTask;
            }, a: mp);
        }
        AddOutOfCombatPotions(player, d);

        return new Pending(d, "map", CooldownMs: 350);
    }

    // ------------------------------------------------------------------ Event & FakeMerchant

    private Pending? Event(RunState state, Player player, EventRoom er, Node roomNode, out string waitingFor)
    {
        waitingFor = "Event";
        var ev = er.LocalMutableEvent;

        if (FindFirst<NFakeMerchant>(roomNode) is { } fake)
            return FakeMerchantScreen(state, player, fake, ev as FakeMerchant, out waitingFor);

        var ancient = FindFirst<NAncientEventLayout>(roomNode);
        var hitbox = ancient?.GetNodeOrNull<NButton>("%DialogueHitbox");
        bool dialogue = hitbox is { Visible: true, IsEnabled: true };

        var allButtons = FindAll<NEventOptionButton>(roomNode);
        var clickable = allButtons
            .Where(b => !b.Option.IsLocked && b.Visible &&
                        (ancient != null ? !dialogue && b.MouseFilter == Control.MouseFilterEnum.Stop : b.IsEnabled))
            .ToList();

        var proceedButton = allButtons.FirstOrDefault(b =>
            b.Option.IsProceed && !b.Option.IsLocked && b.Visible && !dialogue &&
            (b.IsEnabled || b.MouseFilter == Control.MouseFilterEnum.Stop));

        if (clickable.Count == 0)
        {
            if (dialogue && hitbox != null)
            {
                var auto = new Decision(DecisionKind.Event) { Context = "Event dialogue" };
                auto.Add(ActionKind.EventOption, "Next dialogue", () =>
                {
                    if (GodotObject.IsInstanceValid(hitbox))
                        hitbox.EmitSignal(NClickableControl.SignalName.Released, hitbox);
                    return Task.CompletedTask;
                });
                // Repeatable=true fixes the multi-page Ancient dialogue 1.7s stall bug!
                return new Pending(auto, "event dialogue", AutoStep: true, Repeatable: true, CooldownMs: 260);
            }
            return null;
        }

        // Only offer options that are actually clickable in the UI and not excluded (fixes Trial DOUBLE_DOWN & desync!)
        var d = new Decision(DecisionKind.Event) { Context = ev.Id.Entry };
        foreach (var opt in ev.CurrentOptions)
        {
            if (opt.IsLocked || opt.WasChosen || (opt.TextKey != null && ExcludedEventOptions.Contains(opt.TextKey))) continue;
            var btn = clickable.FirstOrDefault(b => ReferenceEquals(b.Option, opt));
            if (btn == null) continue;
            var capturedBtn = btn;
            string shortKey = opt.TextKey?.Split('.').LastOrDefault() ?? opt.TextKey ?? "Option";
            d.Add(ActionKind.EventOption, shortKey, () =>
            {
                if (GodotObject.IsInstanceValid(capturedBtn)) capturedBtn.ForceClick();
                return Task.CompletedTask;
            }, a: opt, b: HoverObject(opt));
        }

        if (d.Choices.Count == 0 || d.Choices.All(c => c.A is EventOption { IsProceed: true }))
        {
            if (proceedButton == null) return null;
            var auto = new Decision(DecisionKind.Event) { Context = "Event proceed" };
            auto.Add(ActionKind.EventOption, "Proceed", () =>
            {
                if (GodotObject.IsInstanceValid(proceedButton)) proceedButton.ForceClick();
                return Task.CompletedTask;
            }, a: proceedButton.Option);
            return new Pending(auto, "event proceed", AutoStep: true, Repeatable: true, CooldownMs: 280);
        }

        AddOutOfCombatPotions(player, d);
        return new Pending(d, "event " + ev.Id.Entry, CooldownMs: 320);
    }

    private Pending? FakeMerchantScreen(RunState state, Player player, NFakeMerchant fake, FakeMerchant? model, out string waitingFor)
    {
        waitingFor = "Fake Merchant";
        if (model == null || model.StartedFight)
        {
            var proceed = fake.GetNodeOrNull<NProceedButton>("%ProceedButton");
            if (proceed is not { IsEnabled: true, Visible: true }) return null;
            var auto = new Decision(DecisionKind.Shop) { Context = "Leave Fake Merchant" };
            auto.Add(ActionKind.Leave, "Leave", () =>
            {
                if (GodotObject.IsInstanceValid(proceed)) proceed.ForceClick();
                return Task.CompletedTask;
            });
            return new Pending(auto, "fake merchant: leave", AutoStep: true, CooldownMs: 300);
        }

        if (fake.Inventory is { IsOpen: false })
        {
            var proceed = fake.GetNodeOrNull<NProceedButton>("%ProceedButton");
            if (proceed is not { IsEnabled: true, Visible: true }) return null;
            // Let the policy shop directly via model.Inventory or leave
            return BuildShopDecision(player, model.Inventory, () =>
            {
                if (GodotObject.IsInstanceValid(proceed)) proceed.ForceClick();
                return Task.CompletedTask;
            }, isFakeMerchant: true, out waitingFor);
        }
        else
        {
            var back = fake.Inventory != null ? FindFirst<NBackButton>(fake.Inventory) : null;
            var proceed = fake.GetNodeOrNull<NProceedButton>("%ProceedButton");
            return BuildShopDecision(player, model.Inventory, async () =>
            {
                if (back != null && GodotObject.IsInstanceValid(back) && back.IsEnabled) back.ForceClick();
                int gen = Live.Generation;
                await Task.Delay(300);
                if (!Live.Active(gen) || proceed == null || !GodotObject.IsInstanceValid(proceed)) return;
                proceed.ForceClick();
            }, isFakeMerchant: true, out waitingFor);
        }
    }

    private static object? HoverObject(EventOption opt)
    {
        if (opt.HoverTips == null) return null;
        foreach (var tip in opt.HoverTips)
        {
            object? m = tip is CardHoverTip ct ? ct.Card
                : tip is HoverTip ht ? ht.CanonicalModel : null;
            if (m is CardModel or PotionModel || (m is RelicModel r && r.Id != opt.Relic?.Id)) return m;
        }
        return null;
    }

    // ------------------------------------------------------------------ Rest Site

    private Pending? Rest(RunState state, Player player, RestSiteRoom room)
    {
        if (NRestSiteRoom.Instance is not { } node || !node.IsVisibleInTree()) return null;

        if (node.ProceedButton is { IsEnabled: true } proceed && proceed.IsVisibleInTree())
        {
            var auto = new Decision(DecisionKind.RestSite) { Context = "Leave Rest Site" };
            auto.Add(ActionKind.Leave, "Proceed", () =>
            {
                if (GodotObject.IsInstanceValid(proceed)) proceed.ForceClick();
                return Task.CompletedTask;
            });
            return new Pending(auto, "rest site: proceed", AutoStep: true, Repeatable: true, CooldownMs: 300);
        }

        var buttons = FindAll<NRestSiteButton>(node)
            .Where(b => b.IsEnabled && b.IsVisibleInTree() && !b.IsQueuedForDeletion())
            .ToArray();
        if (buttons.Length == 0) return null;

        bool canSmith = room.Options.Any(o => o is SmithRestSiteOption && o.IsEnabled &&
                                              buttons.Any(b => ReferenceEquals(b.Option, o)));

        var d = new Decision(DecisionKind.RestSite) { Context = "Rest Site" };
        foreach (var opt in room.Options)
        {
            if (!opt.IsEnabled || opt is MendRestSiteOption) continue;
            if (opt is HealRestSiteOption && canSmith && RunSession.RestHealInfo(player) is { actual: 0, extra: 0 })
                continue;
            var btn = buttons.FirstOrDefault(b => ReferenceEquals(b.Option, opt));
            if (btn == null) continue;
            var captured = btn;
            d.Add(ActionKind.RestOption, opt.OptionId, () =>
            {
                if (GodotObject.IsInstanceValid(captured)) captured.ForceClick();
                return Task.CompletedTask;
            }, a: opt);
        }

        if (d.Choices.Count == 0) return null;
        AddOutOfCombatPotions(player, d);
        return new Pending(d, "rest site", CooldownMs: 350);
    }

    // ------------------------------------------------------------------ Shop

    private Pending? Shop(RunState state, Player player, MerchantRoom mr, Node roomNode, out string waitingFor)
    {
        waitingFor = "Shop";
        if (_pendingShopTask is { IsCompleted: false } || Environment.TickCount64 < _shopBusyUntilMs)
        {
            waitingFor = "Shop purchase resolving";
            return null;
        }
        _pendingShopTask = null;

        var node = FindFirst<NMerchantRoom>(roomNode);
        if (node == null) return null;

        if (node.Inventory is { IsOpen: false })
        {
            var auto = new Decision(DecisionKind.Shop) { Context = "Open Shop" };
            auto.Add(ActionKind.Leave, "Open Shop", () =>
            {
                if (GodotObject.IsInstanceValid(node)) node.OpenInventory();
                _shopOpenedAtMs = Environment.TickCount64;
                return Task.CompletedTask;
            });
            return new Pending(auto, "open shop", AutoStep: true, CooldownMs: 300);
        }

        if (Environment.TickCount64 - _shopOpenedAtMs < 650)
        {
            waitingFor = "Shop opening";
            return null;
        }

        var inv = RunManager.Instance.MerchantSynchronizer?.GetLocalInventory();
        if (inv == null) return null;

        return BuildShopDecision(player, inv, async () =>
        {
            if (GodotObject.IsInstanceValid(node))
                FindFirst<NBackButton>(node)?.ForceClick();
            int gen = Live.Generation;
            await Task.Delay(320);
            if (!Live.Active(gen) || !GodotObject.IsInstanceValid(node)) return;
            node.ProceedButton.ForceClick();
        }, isFakeMerchant: false, out waitingFor);
    }

    private Pending? BuildShopDecision(Player player, MerchantInventory inv, Func<Task> leaveAction,
                                       bool isFakeMerchant, out string waitingFor)
    {
        waitingFor = "Shop";
        var d = new Decision(DecisionKind.Shop)
        {
            Context = (isFakeMerchant ? "Fake Shop" : "Shop") + $" ({player.Gold}g)",
        };

        foreach (var e in inv.AllEntries)
        {
            if (e.IsStocked && !e.EnoughGold && !_failedShopEntries.Contains(e) &&
                e is not MerchantCardRemovalEntry { Used: true })
            {
                d.Locked.Add((EntryObject(e), e.Cost));
            }
            if (!e.IsStocked || !e.EnoughGold || _failedShopEntries.Contains(e)) continue;
            if (e is MerchantPotionEntry pot)
            {
                if (!player.HasOpenPotionSlots) continue;
                bool canProcure = true;
                try
                {
                    if (pot.Model != null)
                        canProcure = MegaCrit.Sts2.Core.Hooks.Hook.ShouldProcurePotion(
                            player.RunState, player.Creature.CombatState, pot.Model, player);
                }
                catch { }
                if (!canProcure) continue;
            }
            if (e is MerchantCardRemovalEntry removal && removal.Used) continue;

            var entry = e;
            d.Add(ActionKind.Buy, $"Buy {EntryLabel(entry)} ({entry.Cost}g)", () =>
            {
                _shopBusyUntilMs = Environment.TickCount64 + 450;
                // Do NOT deadlock on overlay-opening purchases (e.g. Card Removal / Cauldron),
                // but track the task so we don't double-buy while it settles.
                int goldBefore = player.Gold;
                Task<bool> purchaseTask = isFakeMerchant
                    ? entry.OnTryPurchaseUnsynchronized(inv)
                    : RunManager.Instance.MerchantSynchronizer!.PurchaseLocalEntry(entry);
                _pendingShopTask = purchaseTask.ContinueWith(t =>
                {
                    if (t.IsFaulted)
                    {
                        Log.Error("[NecroPilot] shop purchase failed: " + t.Exception?.GetBaseException().Message);
                        _failedShopEntries.Add(entry);
                    }
                    else if (!t.Result && player.Gold == goldBefore)
                    {
                        _failedShopEntries.Add(entry);
                    }
                    else
                    {
                        _failedShopEntries.Clear();
                    }
                });
                return Task.CompletedTask;
            }, a: EntryObject(entry), f0: entry.Cost);
        }

        d.Add(ActionKind.Leave, "Leave Shop", leaveAction);
        AddOutOfCombatPotions(player, d);

        if (!player.HasOpenPotionSlots)
        {
            bool affordablePotion = inv.AllEntries.Any(e =>
                e is MerchantPotionEntry && e.IsStocked && e.EnoughGold && !_failedShopEntries.Contains(e));
            if (affordablePotion) AddPotionDiscards(player, d);
            foreach (var e in inv.AllEntries)
                if (e is MerchantPotionEntry { Model: { } p } && e.IsStocked)
                    d.Shown.Add((p, Zone.Offer, e.Cost));
        }

        return new Pending(d, isFakeMerchant ? "fake merchant" : "shop", CooldownMs: 320);
    }

    // ------------------------------------------------------------------ Treasure

    private Pending? Treasure(RunState state, Player player, Node roomNode, out string waitingFor)
    {
        waitingFor = "Treasure";
        var node = FindFirst<NTreasureRoom>(roomNode);
        if (node == null) return null;

        var chest = node.GetNodeOrNull<NClickableControl>("%Chest") ?? node.GetNodeOrNull<NClickableControl>("Chest");
        if (chest is { IsEnabled: true, Visible: true })
        {
            var auto = new Decision(DecisionKind.Treasure) { Context = "Open Chest" };
            auto.Add(ActionKind.PickRelic, "Open Chest", () =>
            {
                if (GodotObject.IsInstanceValid(chest)) chest.ForceClick();
                return Task.CompletedTask;
            });
            return new Pending(auto, "open chest", AutoStep: true, CooldownMs: 450);
        }

        var relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
        if (relics == null || relics.Count == 0)
        {
            if (node.ProceedButton is { IsEnabled: true } && node.ProceedButton.IsVisibleInTree())
            {
                var auto = new Decision(DecisionKind.Treasure) { Context = "Leave Treasure" };
                auto.Add(ActionKind.Leave, "Proceed", () =>
                {
                    if (GodotObject.IsInstanceValid(node)) node.ProceedButton.ForceClick();
                    return Task.CompletedTask;
                });
                return new Pending(auto, "leave treasure room", AutoStep: true, CooldownMs: 300);
            }
            return null;
        }

        var holders = FindAll<NTreasureRoomRelicHolder>(node);
        if (!holders.Any(h => h.IsEnabled && h.Visible)) return null;

        // Auto-claim single no-downside relic (matching training RewardsFlow.AutoRelics)
        if (RewardsFlow.AutoRelics && relics.Count == 1 && RewardsFlow.IsNoDownside(relics[0]))
        {
            var holder = holders.FirstOrDefault(h => h.Index == 0 && h.IsEnabled) ?? holders.FirstOrDefault(h => h.IsEnabled);
            if (holder != null)
            {
                var auto = new Decision(DecisionKind.Treasure) { Context = "Auto-take Relic" };
                auto.Add(ActionKind.PickRelic, $"Auto-take {relics[0].Id.Entry}", () =>
                {
                    if (GodotObject.IsInstanceValid(holder)) holder.ForceClick();
                    return Task.CompletedTask;
                }, a: relics[0]);
                return new Pending(auto, "treasure: auto relic", AutoStep: true, CooldownMs: 350);
            }
        }

        var d = new Decision(DecisionKind.Treasure) { Context = "Treasure" };
        for (int i = 0; i < relics.Count; i++)
        {
            int idx = i;
            var relic = relics[i];
            d.Add(ActionKind.PickRelic, $"Take {relic.Id.Entry}", () =>
            {
                var h = holders.FirstOrDefault(x => x.Index == idx && x.IsEnabled && GodotObject.IsInstanceValid(x))
                        ?? holders.FirstOrDefault(x => x.IsEnabled && GodotObject.IsInstanceValid(x));
                h?.ForceClick();
                return Task.CompletedTask;
            }, a: relic);
        }

        if (node.ProceedButton is { IsEnabled: true })
        {
            d.Add(ActionKind.SkipRelic, "Skip Relic", () =>
            {
                if (GodotObject.IsInstanceValid(node)) node.ProceedButton.ForceClick();
                return Task.CompletedTask;
            });
        }
        AddOutOfCombatPotions(player, d);

        return new Pending(d, "treasure", CooldownMs: 350);
    }

    // ------------------------------------------------------------------ Helpers

    private static string CardName(CardModel c) => c.Id.Entry + (c.IsUpgraded ? "+" : "");
    private static string TargetLabel(Creature? t) => t == null ? "" : t.IsPlayer ? " → self" : $" → {t.Monster?.Id.Entry}";

    private static object EntryObject(MerchantEntry e) => e switch
    {
        MerchantCardEntry c when c.CreationResult?.Card is { } card => card,
        MerchantRelicEntry { Model: { } r } => r,
        MerchantPotionEntry { Model: { } p } => p,
        _ => e,
    };

    private static string EntryLabel(MerchantEntry e) => e switch
    {
        MerchantCardEntry c when c.CreationResult?.Card is { } card => CardName(card),
        MerchantRelicEntry r => r.Model?.Id.Entry ?? "Relic",
        MerchantPotionEntry p => p.Model?.Id.Entry ?? "Potion",
        MerchantCardRemovalEntry => "Card Removal",
        _ => e.GetType().Name,
    };

    private static string RewardLabel(Reward r) => r switch
    {
        GoldReward g => $"{g.Amount}g",
        PotionReward p => p.Potion?.Id.Entry ?? "Potion",
        RelicReward rr => rr.Relic?.Id.Entry ?? "Relic",
        CardReward => "Cards",
        _ => r.GetType().Name,
    };

    private static List<T> FindAll<T>(Node start) where T : Node
    {
        var found = new List<T>();
        void Walk(Node n)
        {
            if (!GodotObject.IsInstanceValid(n)) return;
            if (n is T t) found.Add(t);
            foreach (var c in n.GetChildren()) Walk(c);
        }
        if (GodotObject.IsInstanceValid(start)) Walk(start);
        return found;
    }

    private static T? FindFirst<T>(Node start) where T : Node
    {
        if (!GodotObject.IsInstanceValid(start)) return null;
        if (start is T t) return t;
        foreach (var c in start.GetChildren())
            if (FindFirst<T>(c) is { } r) return r;
        return null;
    }
}
