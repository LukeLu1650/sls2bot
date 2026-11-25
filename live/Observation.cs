using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Ai.Shared.Decisions;
using Sts2Ai.Shared.Encoding;

namespace NecroPilot;

internal sealed class Observation(string path)
{
    private readonly Dictionary<string, string> _names =
        JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(path))!;

    private string Name(string table, string id) => _names.GetValueOrDefault(table + "/" + id, id);
    private string? Text(LocString? key) => key == null ? null :
        _names.GetValueOrDefault(key.LocTable + "/" + key.LocEntryKey);

    private object Card(CardModel card, int index, Creature[] enemies)
    {
        Dictionary<string, int> Values(Creature? target)
        {
            card.DynamicVars.ClearPreview();
            try
            {
                card.UpdateDynamicVarPreview(target == null ? CardPreviewMode.Normal : CardPreviewMode.MultiCreatureTargeting,
                    target, card.DynamicVars);
                return card.DynamicVars.Values.ToDictionary(v => v.Name.ToLowerInvariant(), v => (int)v.PreviewValue);
            }
            catch (InvalidOperationException) { return []; }
            finally { card.DynamicVars.ClearPreview(); }
        }
        var stats = Values(null);
        var previews = new List<Dictionary<string, int>>();
        if (card.Type == CardType.Attack && card.TargetType is TargetType.AnyEnemy or TargetType.AllEnemies)
        {
            for (int i = 0; i < enemies.Length; i++)
            {
                var values = Values(enemies[i]);
                if (!values.TryGetValue("calculateddamage", out int damage) || damage <= 0)
                    damage = values.GetValueOrDefault("damage");
                var preview = new Dictionary<string, int> { ["target_index"] = i, ["damage"] = damage };
                if (values.GetValueOrDefault("repeat") > 1) preview["total_damage"] = damage * values["repeat"];
                previews.Add(preview);
            }
        }
        return new { index, id = card.Id.ToString(), name = Name("cards", card.Id.Entry), cost = card.EnergyCost.GetResolved(),
                     type = card.Type.ToString(), can_play = card.CanPlay(), target_type = card.TargetType.ToString(),
                     stats, damage_by_target = previews };
    }

    private static Dictionary<string, int> BaseStats(CardModel card) =>
        card.DynamicVars.Values.ToDictionary(v => v.Name.ToLowerInvariant(), v => (int)v.BaseValue);

    private static CardModel CardAtCurrentLevel(CardModel card)
    {
        var clone = ModelDb.GetById<CardModel>(card.Id).ToMutable();
        for (int i = 0; i < card.CurrentUpgradeLevel; i++)
        {
            clone.UpgradeInternal();
            clone.FinalizeUpgradeInternal();
        }
        return clone;
    }

    private static object? Upgrade(CardModel card)
    {
        if (!card.IsUpgradable) return null;
        try
        {
            var clone = CardAtCurrentLevel(card);
            clone.UpgradeInternal();
            clone.FinalizeUpgradeInternal();
            return new { cost = clone.EnergyCost.GetResolved(), stats = BaseStats(clone) };
        }
        catch (InvalidOperationException) { return null; }
    }

    private object OfferedCard(Player player, CardModel card, int index, bool previewUpgrade = false) => new
    {
        index, id = card.Id.ToString(), name = Name("cards", card.Id.Entry), cost = card.EnergyCost.GetResolved(),
        type = card.Type.ToString(), target_type = card.TargetType.ToString(), upgraded = card.IsUpgraded,
        description = _names.GetValueOrDefault("cards/" + card.Id.Entry + ".description"),
        stats = BaseStats(card),
        keywords = card.Keywords?.Where(k => k != CardKeyword.None).Select(k => k.ToString()).ToArray(),
        zone = SelectionZone(player, card),
        after_upgrade = previewUpgrade ? Upgrade(card) : null
    };

    private static object? VisibleMap(RunState run)
    {
        if (run.Map is not { } map) return null;
        var rows = Enumerable.Range(0, map.GetRowCount()).Select(row => map.GetPointsInRow(row)
            .OfType<MapPoint>().Select(point => new
            {
                col = point.coord.col, row = point.coord.row, type = point.PointType.ToString(),
                children = point.Children.Select(child => new { col = child.coord.col, row = child.coord.row }).ToArray()
            }).ToArray()).Where(row => row.Length > 0).ToArray();
        return new { rows, current_coord = run.CurrentMapCoord is { } current ? new { col = current.col, row = current.row } : null };
    }

    private object EventOptionInfo(LegalAction action, EventModel? ev)
    {
        var option = (EventOption)action.Payload!;
        var vars = ev?.DynamicVars.Values.ToDictionary(v => v.Name, v => (int)v.BaseValue);
        var entry = option.TextKey?.Split('.').LastOrDefault();
        var title = Text(option.Title) ?? _names.GetValueOrDefault("relics/" + entry) ??
            _names.GetValueOrDefault("cards/" + entry) ?? entry?.Replace('_', ' ') ?? action.Label;
        return new { index = action.Index, title, text_key = option.TextKey, vars };
    }

    private object? ShopCard(MerchantCardEntry entry, int index)
    {
        if (entry.CreationResult?.Card is not { } card) return null;
        return new
        {
            index, name = Name("cards", card.Id.Entry), cost = entry.Cost, card_cost = card.EnergyCost.GetResolved(),
            type = card.Type.ToString(), description = _names.GetValueOrDefault("cards/" + card.Id.Entry + ".description"),
            stats = BaseStats(CardAtCurrentLevel(card))
        };
    }

    private static string? SelectionZone(Player player, CardModel card)
    {
        var pcs = player.PlayerCombatState;
        if (pcs?.Hand.Cards.Contains(card) == true) return "hand";
        if (pcs?.DiscardPile.Cards.Contains(card) == true) return "discard";
        if (pcs?.DrawPile.Cards.Contains(card) == true) return "draw";
        if (pcs?.ExhaustPile.Cards.Contains(card) == true) return "exhaust";
        return player.Deck.Cards.Contains(card) ? "deck" : null;
    }

    internal object? CombatTrigger(LegalAction action) => action.Type switch
    {
        ActionType.PlayCard when action.Payload is CardModel card =>
            new { action = "play_card", id = card.Id.ToString(), name = Name("cards", card.Id.Entry) },
        ActionType.UsePotion when action.Payload is PotionModel potion =>
            new { action = "use_potion", name = Name("potions", potion.Id.Entry) },
        ActionType.EndTurn => new { action = "end_turn" },
        _ => null
    };

    public string Encode(RunState run, Player player, Decision decision, LegalAction[] actions, object? trigger,
                         string? selectionReason)
    {
        var pcs = player.PlayerCombatState;
        var combat = player.Creature.CombatState;
        var hand = pcs?.Hand.Cards.ToArray() ?? [];
        var enemies = combat?.Enemies.Where(enemy => enemy.IsAlive).ToArray() ?? [];
        var allies = combat?.PlayerCreatures.ToList() ?? [];
        var potions = player.Potions.ToArray();
        bool inCombat = CombatManager.Instance.IsInProgress;
        var bossId = run.Act?.BossEncounter?.Id.Entry;
        if (bossId?.EndsWith("_BOSS", StringComparison.Ordinal) == true) bossId = bossId[..^5];
        var shop = decision.Type == DecisionType.Shop ? RunManager.Instance.MerchantSynchronizer?.GetLocalInventory() : null;
        var shopCards = shop?.CardEntries.ToArray() ?? [];
        var shopRelics = shop?.RelicEntries.ToArray() ?? [];
        var shopPotions = shop?.PotionEntries.ToArray() ?? [];
        var parts = new Dictionary<string, object?>
        {
            ["decision"] = decision.Type switch
            {
                DecisionType.Combat => "combat_play",
                DecisionType.Map => "map_select",
                DecisionType.Rest => "rest_site",
                DecisionType.CardReward => "card_reward",
                DecisionType.CardSelect => "card_select",
                DecisionType.BundleSelect => "bundle_select",
                DecisionType.Shop => "shop",
                DecisionType.Event => "event_choice",
                _ => "rewards"
            },
            ["context"] = new { act = run.CurrentActIndex + 1, floor = run.ActFloor, ascension = run.AscensionLevel,
                                room_type = run.CurrentRoom?.RoomType.ToString(),
                                boss = bossId == null ? null : new { id = bossId } },
            ["player"] = new { hp = player.Creature.CurrentHp, max_hp = player.Creature.MaxHp, block = player.Creature.Block,
                               gold = player.Gold, deck_size = player.Deck.Cards.Count, potion_capacity = player.PotionSlots.Count,
                               deck = inCombat ? null : player.Deck.Cards.Select((card, index) => OfferedCard(player, card, index)).ToArray(),
                               potions = potions.Select((potion, index) => new
                               {
                                   index, name = Name("potions", potion.Id.Entry)
                               }).ToArray() },
            ["in_combat"] = inCombat,
            ["round"] = pcs?.TurnNumber, ["energy"] = pcs?.Energy,
            ["draw_pile_count"] = pcs?.DrawPile.Cards.Count, ["discard_pile_count"] = pcs?.DiscardPile.Cards.Count,
            ["hand"] = hand.Select((card, index) => Card(card, index, enemies)).ToArray(),
            ["enemies"] = enemies.Select((enemy, index) => new
            {
                index, name = Name("monsters", enemy.Monster?.Id.Entry ?? "?"), hp = enemy.CurrentHp,
                max_hp = enemy.MaxHp, block = enemy.Block, intends_attack = enemy.Monster?.IntendsToAttack ?? false,
                powers = Powers(enemy),
                intents = enemy.Monster?.NextMove?.Intents.Select(intent =>
                {
                    var values = new Dictionary<string, object> { ["type"] = intent.IntentType.ToString() };
                    if (intent is AttackIntent attack)
                    {
                        int damage = attack.GetTotalDamage(allies, enemy);
                        values["damage"] = attack.Repeats > 1 ? attack.GetSingleDamage(allies, enemy) : damage;
                        if (attack.Repeats > 1)
                        {
                            values["hits"] = attack.Repeats;
                            values["total_damage"] = damage;
                        }
                    }
                    return values;
                }).ToArray()
            }).ToArray(),
            ["player_powers"] = Powers(player.Creature),
        };
        var osty = player.Osty;
        parts["osty"] = osty == null ? new { alive = false } :
            (object)new { alive = osty.IsAlive, hp = osty.CurrentHp, max_hp = osty.MaxHp, block = osty.Block,
                          name = Name("monsters", osty.Monster?.Id.Entry ?? "OSTY") };
        if (inCombat && decision.Type is (DecisionType.CardSelect or DecisionType.CardReward or DecisionType.BundleSelect))
            parts["selection_trigger"] = trigger;
        if (decision.Type == DecisionType.CardSelect)
        {
            bool smith = decision.Context == "smith";
            parts["cards"] = decision.CardOptions.Select((card, index) => OfferedCard(player, card, index, smith)).ToArray();
            parts["min_select"] = decision.MinSelect;
            parts["max_select"] = decision.MaxSelect;
            parts["picked_indices"] = decision.PickedIndices;
            parts["selection_reason"] = smith ? "smith" : selectionReason;
        }
        if (decision.Type == DecisionType.CardReward)
            parts["cards"] = actions.Where(a => a.Payload is CardModel)
                .Select(a => OfferedCard(player, (CardModel)a.Payload!, a.Index)).ToArray();
        if (decision.Type == DecisionType.BundleSelect)
            parts["bundles"] = actions.Where(a => a.Payload is IReadOnlyList<CardModel>)
                .Select(a => new { index = a.Index, cards = ((IReadOnlyList<CardModel>)a.Payload!)
                    .Select((card, index) => OfferedCard(player, card, index)).ToArray() }).ToArray();
        if (!inCombat)
            parts["visible_map"] = VisibleMap(run);
        if (decision.Type == DecisionType.Map)
            parts["choices"] = actions.Select(a => a.Payload).OfType<MapPoint>()
                .Select(point => new { col = point.coord.col, row = point.coord.row, type = point.PointType.ToString() }).ToArray();
        if (shop != null)
        {
            parts["cards"] = shopCards.Select(ShopCard).OfType<object>().ToArray();
            parts["relics"] = shopRelics.Select((entry, index) =>
                new { index, name = Name("relics", entry.Model?.Id.Entry ?? "?"), cost = entry.Cost }).ToArray();
            parts["potions"] = shopPotions.Select((entry, index) =>
                new { index, name = Name("potions", entry.Model?.Id.Entry ?? "?"), cost = entry.Cost }).ToArray();
            parts["card_removal_cost"] = shop.CardRemovalEntry?.Cost;
        }
        if (decision.Type == DecisionType.Event)
            parts["options"] = actions.Where(a => a.Payload is EventOption)
                .Select(a => EventOptionInfo(a, (run.CurrentRoom as EventRoom)?.LocalMutableEvent)).ToArray();
        if (decision.Type == DecisionType.Rest)
            parts["options"] = actions.Where(a => a.Type == ActionType.ChooseOption)
                .Select(a => new { index = a.Index, option_id = ((RestSiteOption)a.Payload!).OptionId }).ToArray();

        var inputs = actions.Select(action =>
        {
            var args = new Dictionary<string, object>();
            string name = action.Type switch
            {
                ActionType.PlayCard => "play_card",
                ActionType.UsePotion => "use_potion",
                ActionType.EndTurn => "end_turn",
                ActionType.ChooseMapNode => "select_map_node",
                ActionType.ChooseOption when decision.Type == DecisionType.BundleSelect => "select_bundle",
                ActionType.ChooseOption => "choose_option",
                ActionType.SelectCard => decision.Type == DecisionType.CardReward ? "select_card_reward" : "pick_card",
                ActionType.Skip => "skip_card_reward",
                ActionType.Confirm => "confirm_selection",
                ActionType.RemoveCardService => "remove_card",
                ActionType.Buy when action.Payload is MerchantCardEntry => "buy_card",
                ActionType.Buy when action.Payload is MerchantRelicEntry => "buy_relic",
                ActionType.Buy => "buy_potion",
                ActionType.Proceed when decision.Type == DecisionType.Shop => "leave_room",
                _ => action.Type.ToString()
            };
            if (action.Type == ActionType.PlayCard) args["card_index"] = Array.IndexOf(hand, action.Payload);
            if (action.Type == ActionType.UsePotion && action.Payload is PotionModel potion)
                args["potion_index"] = Array.IndexOf(potions, potion);
            if (action.Type == ActionType.ChooseOption)
                args[decision.Type == DecisionType.BundleSelect ? "bundle_index" : "option_index"] = action.Index;
            if (action.Type == ActionType.SelectCard)
                args["card_index"] = action.Index;
            if (action.Type == ActionType.Buy)
            {
                if (action.Payload is MerchantCardEntry card) args["card_index"] = Array.IndexOf(shopCards, card);
                if (action.Payload is MerchantRelicEntry relic) args["relic_index"] = Array.IndexOf(shopRelics, relic);
                if (action.Payload is MerchantPotionEntry offer) args["potion_index"] = Array.IndexOf(shopPotions, offer);
            }
            if (action.Payload is MapPoint point) { args["col"] = point.coord.col; args["row"] = point.coord.row; }
            if (action.Target is Creature enemy && Array.IndexOf(enemies, enemy) is var target && target >= 0)
                args["target_index"] = target;
            return new { action = name, args };
        }).ToArray();
        return JsonSerializer.Serialize(new { state = parts, actions = inputs });
    }

    private object[] Powers(Creature creature) => creature.Powers.Select(p =>
        (object)new { name = Name("powers", p.Id.Entry), amount = p.Amount }).ToArray();
}
