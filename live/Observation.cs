using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Runs;
using Sts2Ai.Shared.Decisions;
using Sts2Ai.Shared.Encoding;

namespace NecroPilot;

internal sealed class Observation(string path)
{
    private readonly Dictionary<string, string> _names =
        JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(path))!;

    private string Name(string table, string id) => _names.GetValueOrDefault(table + "/" + id, id);

    private object Card(CardModel card, int index, Creature[] enemies, bool inHand = true)
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
        var stats = inHand ? Values(null) : null;
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
                     type = card.Type.ToString(), can_play = inHand && card.CanPlay(), target_type = card.TargetType.ToString(),
                     stats, damage_by_target = previews };
    }

    public string Encode(RunState run, Player player, Decision decision, LegalAction[] actions)
    {
        var pcs = player.PlayerCombatState;
        var combat = player.Creature.CombatState;
        var hand = pcs?.Hand.Cards.ToArray() ?? [];
        var enemies = combat?.Enemies.Where(enemy => enemy.IsAlive).ToArray() ?? [];
        var allies = combat?.PlayerCreatures.ToList() ?? [];
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
                                boss = run.Act?.BossEncounter?.Id.Entry },
            ["player"] = new { hp = player.Creature.CurrentHp, max_hp = player.Creature.MaxHp, block = player.Creature.Block,
                               gold = player.Gold, deck_size = player.Deck.Cards.Count, potion_capacity = player.PotionSlots.Count },
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
        if (decision.Type == DecisionType.CardReward)
            parts["cards"] = actions.Where(a => a.Payload is CardModel)
                .Select(a => Card((CardModel)a.Payload!, a.Index, [], inHand: false)).ToArray();
        if (decision.Type is DecisionType.Event or DecisionType.Rest)
            parts["options"] = actions.Where(a => a.Type == ActionType.ChooseOption)
                .Select(a => new { index = a.Index, title = a.Label }).ToArray();

        var inputs = actions.Select(action =>
        {
            var args = new Dictionary<string, object>();
            string name = action.Type switch
            {
                ActionType.PlayCard => "play_card",
                ActionType.UsePotion => "use_potion",
                ActionType.EndTurn => "end_turn",
                ActionType.ChooseMapNode => "select_map_node",
                ActionType.ChooseOption => "choose_option",
                ActionType.SelectCard when decision.Type == DecisionType.BundleSelect => "select_bundle",
                ActionType.SelectCard => decision.Type == DecisionType.CardReward ? "select_card_reward" : "select_cards",
                ActionType.Skip => "skip_card_reward",
                ActionType.Confirm => "skip_select",
                ActionType.RemoveCardService => "remove_card",
                ActionType.Buy when action.Payload is MerchantCardEntry => "buy_card",
                ActionType.Buy when action.Payload is MerchantRelicEntry => "buy_relic",
                ActionType.Buy => "buy_potion",
                ActionType.Proceed when decision.Type == DecisionType.Shop => "leave_room",
                _ => action.Type.ToString()
            };
            if (action.Type == ActionType.PlayCard) args["card_index"] = Array.IndexOf(hand, action.Payload);
            if (action.Type == ActionType.UsePotion && action.Payload is PotionModel potion)
                args["potion_index"] = player.Potions.ToList().IndexOf(potion);
            if (action.Type == ActionType.ChooseOption) args["option_index"] = action.Index;
            if (action.Type == ActionType.SelectCard)
            {
                if (decision.Type == DecisionType.CardReward) args["card_index"] = action.Index;
                else if (decision.Type == DecisionType.BundleSelect) args["bundle_index"] = action.Index;
                else args["indices"] = action.Index.ToString();
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
