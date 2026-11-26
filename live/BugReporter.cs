using System.Reflection;
using System.Text.Json;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace NecroPilot;

/// <summary>
/// Captures ring-buffered Pilot history and on-demand (F6) / auto-stop diagnostic JSON snapshots
/// of live game state, UI overlay/button states, Pilot detection, and Observation encoding.
/// </summary>
internal static class BugReporter
{
    private const int MaxTimeline = 60;
    private const int MaxStderr = 40;

    private static readonly object LockObj = new();
    private static readonly Queue<object> Timeline = new();
    private static readonly Queue<string> PyStderr = new();
    private static int _counter;

    private static readonly JsonSerializerOptions PrettyJson = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    internal static void RecordEvent(string kind, string detail, object? extra = null)
    {
        lock (LockObj)
        {
            Timeline.Enqueue(new
            {
                ts = DateTime.Now.ToString("HH:mm:ss.fff"),
                kind,
                detail,
                extra,
            });
            while (Timeline.Count > MaxTimeline) Timeline.Dequeue();
        }
    }

    internal static void RecordPyStderr(string line)
    {
        lock (LockObj)
        {
            PyStderr.Enqueue($"[{DateTime.Now:HH:mm:ss.fff}] {line}");
            while (PyStderr.Count > MaxStderr) PyStderr.Dequeue();
        }
    }

    internal static string SaveSnapshot(string trigger, string outputDir, LivePilot? pilot, object pilotStatus)
    {
        Directory.CreateDirectory(outputDir);
        int seq = Interlocked.Increment(ref _counter);
        string safeTrigger = string.Concat(trigger.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '_'));
        if (safeTrigger.Length > 24) safeTrigger = safeTrigger[..24];
        string fileName = $"{DateTime.Now:yyyyMMdd-HHmmss}-{seq:D2}-{safeTrigger}.json";
        string fullPath = Path.Combine(outputDir, fileName);

        object snapshot = BuildSnapshot(trigger, pilot, pilotStatus);
        File.WriteAllText(fullPath, JsonSerializer.Serialize(snapshot, PrettyJson));
        Log.Info($"[NecroPilot] Bug snapshot saved: {fullPath}");
        return fullPath;
    }

    private static object BuildSnapshot(string trigger, LivePilot? pilot, object pilotStatus)
    {
        object[] historyCopy;
        string[] stderrCopy;
        lock (LockObj)
        {
            historyCopy = Timeline.ToArray();
            stderrCopy = PyStderr.ToArray();
        }

        var manager = RunManager.Instance;
        RunState? run = null;
        Player? player = null;
        try
        {
            run = manager?.DebugOnlyGetState();
            player = run == null ? null : LocalContext.GetMe(run.Players);
        }
        catch { }

        object? detectInfo = null;
        if (run != null && player != null)
        {
            try
            {
                var probe = pilot ?? new LivePilot();
                var pending = probe.Detect(run, player, out var waiting);
                string? obsStatus = null;
                int? obsBytes = null;
                if (pending != null && pending.Decision.Choices.Count > 0 && !pending.AutoStep)
                {
                    try
                    {
                        var obs = Observation.Build(probe.Session, pending.Decision, labels: true);
                        obsBytes = obs.Serialize().Length;
                        obsStatus = $"ok (tokens={obs.NumTokens}, cands={obs.NumCands}, fmt={obs.Header[0]})";
                    }
                    catch (Exception ex)
                    {
                        obsStatus = $"ERROR: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}";
                    }
                }
                detectInfo = new
                {
                    waiting_for = waiting,
                    pending = pending == null ? null : new
                    {
                        context = pending.Context,
                        decision_kind = pending.Decision.Kind.ToString(),
                        decision_context = pending.Decision.Context,
                        auto_step = pending.AutoStep,
                        repeatable = pending.Repeatable,
                        selected_count = pending.Decision.Selected,
                        choices = pending.Decision.Choices.Select((c, i) => new
                        {
                            index = i,
                            kind = c.Kind.ToString(),
                            label = c.Label,
                            a = c.A?.GetType().Name,
                            b = c.B?.GetType().Name,
                            key = c.Key,
                            f0 = c.F0,
                            f1 = c.F1,
                            f2 = c.F2,
                        }).ToArray(),
                    },
                    observation_check = obsStatus,
                    observation_bytes = obsBytes,
                    pilot_internals = probe.DebugState(),
                };
            }
            catch (Exception ex)
            {
                detectInfo = new { detect_error = $"{ex.GetType().Name}: {ex.Message}", stack = ex.StackTrace };
            }
        }

        return new
        {
            timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"),
            trigger,
            mod_version = "0.2.1",
            pilot_status = pilotStatus,
            game = SafeRunState(manager, run, player),
            ui = SafeUiState(),
            hooks = new
            {
                active_select_info = AgentCardSelector.ActiveSelectInfo,
                active_cancelable = AgentCardSelector.ActiveCancelable,
                using_potion = AgentCardSelector.UsingPotion?.Id.Entry,
                rewards_pending = RewardsFlow.Pending != null,
                rewards_active_set = RewardsFlow.ActiveSet != null,
                crystal_sphere_active = FakeScreens.CrystalSphere != null,
                vocab_missing = Vocab.Missing.Take(20).ToDictionary(kv => kv.Key, kv => kv.Value),
            },
            detect = detectInfo,
            recent_timeline = historyCopy,
            recent_py_stderr = stderrCopy,
        };
    }

    private static object? SafeRunState(RunManager? manager, RunState? run, Player? player)
    {
        if (run == null || player == null)
            return new { in_run = false, paused = SafePaused(), in_transition = NGame.Instance?.Transition?.InTransition };

        try
        {
            var pcs = player.PlayerCombatState;
            var cs = player.Creature.CombatState;
            bool inCombat = CombatManager.Instance.IsInProgress;
            return new
            {
                in_run = true,
                paused = SafePaused(),
                in_transition = NGame.Instance?.Transition?.InTransition,
                game_mode = run.GameMode.ToString(),
                net_type = manager?.NetService?.Type.ToString(),
                character = player.Character.Id.Entry,
                ascension = run.AscensionLevel,
                act = run.CurrentActIndex + 1,
                act_floor = run.ActFloor,
                total_floor = run.TotalFloor,
                room_type = run.CurrentRoom?.RoomType.ToString(),
                room_class = run.CurrentRoom?.GetType().Name,
                hp = player.Creature.CurrentHp,
                max_hp = player.Creature.MaxHp,
                block = player.Creature.Block,
                gold = player.Gold,
                deck_count = player.Deck.Cards.Count,
                relics = player.Relics.Select(r => r.Id.Entry).ToArray(),
                potions = player.PotionSlots.Select(p => p?.Id.Entry).ToArray(),
                action_queue_empty = manager?.ActionQueueSet.IsEmpty,
                action_executor_running = manager?.ActionExecutor.IsRunning,
                in_combat = inCombat,
                combat = !inCombat ? null : new
                {
                    turn = pcs?.TurnNumber,
                    phase = pcs?.Phase.ToString(),
                    energy = pcs?.Energy,
                    actions_disabled = CombatManager.Instance.PlayerActionsDisabled,
                    is_over_or_ending = CombatManager.Instance.IsOverOrEnding,
                    hand = pcs?.Hand.Cards.Select(c => new { id = c.Id.Entry, up = c.CurrentUpgradeLevel, playable = SafeCanPlay(c) }).ToArray(),
                    enemies = cs?.Enemies.Select(e => new { id = e.Monster?.Id.Entry, hp = e.CurrentHp, max_hp = e.MaxHp, block = e.Block, alive = e.IsAlive }).ToArray(),
                },
            };
        }
        catch (Exception ex)
        {
            return new { error = ex.ToString() };
        }
    }

    private static bool SafePaused()
    {
        try { return ((SceneTree)Engine.GetMainLoop()).Paused; } catch { return false; }
    }

    private static bool? SafeCanPlay(CardModel c)
    {
        try { return c.CanPlay(out _, out _); } catch { return null; }
    }

    private static object SafeUiState()
    {
        try
        {
            var stack = NOverlayStack.Instance;
            object? overlayInfo = null;
            if (stack != null && stack.ScreenCount > 0 && stack.Peek() is Node top)
            {
                object? gridDetail = null;
                if (top is NCardGridSelectionScreen grid)
                {
                    var t = grid.GetType();
                    var sel = (AccessTools.Field(t, "_selectedCards")?.GetValue(grid) as IEnumerable<CardModel>)
                              ?.Select(c => c.Id.Entry).ToArray();
                    var preview = SelectionControls.Preview(grid);
                    var gridConfirm = SelectionControls.GridConfirm(grid);
                    gridDetail = new
                    {
                        selected_cards = sel,
                        preview_open = preview.IsOpen,
                        preview_confirm_ready = preview.Button != null,
                        grid_confirm_ready = gridConfirm != null,
                    };
                }
                overlayInfo = new
                {
                    screen_count = stack.ScreenCount,
                    top_type = top.GetType().Name,
                    top_name = top.Name.ToString(),
                    visible = top is CanvasItem ci && ci.IsVisibleInTree(),
                    grid_detail = gridDetail,
                    buttons = FindButtons(top, 20),
                };
            }

            var map = NMapScreen.Instance;
            var hand = NCombatRoom.Instance?.Ui?.Hand;
            object? handInfo = null;
            if (hand != null && GodotObject.IsInstanceValid(hand))
            {
                var handSel = (AccessTools.Field(typeof(NPlayerHand), "_selectedCards")?.GetValue(hand) as IEnumerable<CardModel>)
                              ?.Select(c => c.Id.Entry).ToArray();
                var handConfirm = AccessTools.Field(typeof(NPlayerHand), "_selectModeConfirmButton")?.GetValue(hand) as NConfirmButton;
                handInfo = new
                {
                    is_in_card_selection = hand.IsInCardSelection,
                    selected_cards = handSel,
                    confirm_enabled = handConfirm?.IsEnabled,
                    confirm_visible = handConfirm?.IsVisibleInTree(),
                };
            }

            var rest = NRestSiteRoom.Instance;
            object? restInfo = null;
            if (rest != null && GodotObject.IsInstanceValid(rest) && rest.IsVisibleInTree())
            {
                restInfo = new
                {
                    proceed_enabled = rest.ProceedButton?.IsEnabled,
                    proceed_visible = rest.ProceedButton?.IsVisibleInTree(),
                    buttons = FindButtons(rest, 12),
                };
            }

            return new
            {
                overlay = overlayInfo,
                map = map == null ? null : new
                {
                    is_open = map.IsOpen,
                    visible = map.IsVisibleInTree(),
                    travel_enabled = map.IsTravelEnabled,
                    is_traveling = map.IsTraveling,
                },
                hand = handInfo,
                rest = restInfo,
            };
        }
        catch (Exception ex)
        {
            return new { error = ex.ToString() };
        }
    }

    private static object[] FindButtons(Node root, int limit)
    {
        var list = new List<object>();
        void Walk(Node n)
        {
            if (list.Count >= limit || !GodotObject.IsInstanceValid(n)) return;
            if (n is NClickableControl btn)
            {
                list.Add(new
                {
                    type = btn.GetType().Name,
                    name = btn.Name.ToString(),
                    visible = btn.IsVisibleInTree(),
                    enabled = btn.IsEnabled,
                    mouse = btn.MouseFilter.ToString(),
                });
            }
            foreach (var child in n.GetChildren()) Walk(child);
        }
        Walk(root);
        return list.ToArray();
    }
}
