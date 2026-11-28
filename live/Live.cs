using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using Sts2Headless;
using Environment = System.Environment;

namespace NecroPilot;

[ModInitializer(nameof(Init))]
public static class Live
{
    private static bool _enabled;
    private static bool _f7Down, _f6Down, _f8Down;
    private static bool _slEnabled;
    private static int _slGeneration, _inferSlGeneration;
    private static object? _slCombat;
    private static readonly Queue<PlanStep> SlPlan = new();
    private static readonly FieldInfo? ReplayField = typeof(CombatReplayWriter)
        .GetField("_replay", BindingFlags.Instance | BindingFlags.NonPublic);
    internal static int Generation { get; private set; }
    internal static bool Active(int generation) => _enabled && generation == Generation;

    private static string _directory = "";
    private static Config? _config;
    private static LivePilot? _pilot;
    private static Process? _worker;
    private static Task<string?>? _readyTask;
    private static bool _workerReady;
    private static string _modelBanner = "";
    private static string? _trainedCharacter;

    private static Task<string?>? _inferTask;
    private static string? _inferFrameKey;
    private static long _inferStartedMs;
    private static int _inferTimeoutMs = 8000;
    private static Task? _actionTask;
    private static long _actionStartedMs;
    private static long _nextPollMs;
    private static long _cooldownUntilMs;

    private static RunState? _run;
    private static string? _lastFrameKey;
    private static long _sameFrameSinceMs;
    private static long? _selectionWaitSinceMs;
    private static string _lastWaiting = "";

    // HUD controls
    private static PanelContainer _panel = null!;
    private static Button _pilotBtn = null!;
    private static Button _bugBtn = null!;
    private static Button _slBtn = null!;
    private static Label _metaLabel = null!;
    private static Label _statusLabel = null!;
    private static Label _valueLabel = null!;
    private static Label _scoresLabel = null!;
    private static Label _bugToastLabel = null!;
    private static long _bugToastUntilMs;

    public static void Init()
    {
        try
        {
            _directory = Path.GetDirectoryName(typeof(Live).Assembly.Location)!;
            Vocab.Load(Path.Combine(_directory, "vocab.txt"));
            RewardsFlow.LoadNoDownside(Path.Combine(_directory, "no_downside_relics.txt"));
            Observation.RouteSummary = true;
            Observation.Privileged = false;
            Observation.FutureContent = false;
            Observation.Oracle = false;
            LiveHooks.Install();

            ((SceneTree)Engine.GetMainLoop()).ProcessFrame += Setup;
            AppDomain.CurrentDomain.ProcessExit += (_, _) => CloseWorker();
            Log.Info($"[NecroPilot] v0.2.2 initialized (vocab={Vocab.Count}, noDownsideRelics={RewardsFlow.NoDownside.Count}).");
        }
        catch (Exception ex)
        {
            Log.Error("[NecroPilot] Init failed: " + ex);
        }
    }

    private static void Setup()
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        tree.ProcessFrame -= Setup;
        try
        {
            BuildHud(tree.Root);
            tree.ProcessFrame += Frame;
            Log.Info("[NecroPilot] HUD loaded (F7=Pilot, F8=Combat SL, F6=Save Bug Snapshot).");
        }
        catch (Exception ex)
        {
            Log.Error("[NecroPilot] HUD setup failed: " + ex);
        }
    }

    private static void BuildHud(Node root)
    {
        var layer = new CanvasLayer { Layer = 123, Name = "NecroPilotHud" };
        root.AddChild(layer);

        _panel = new PanelContainer
        {
            OffsetLeft = 18,
            OffsetTop = 160,
            OffsetRight = 396,
            MouseFilter = Control.MouseFilterEnum.Pass,
        };
        var style = new StyleBoxFlat
        {
            BgColor = new Color(0.06f, 0.08f, 0.12f, 0.84f),
            BorderColor = new Color(0.35f, 0.78f, 0.68f, 0.55f),
            BorderWidthLeft = 1,
            BorderWidthRight = 1,
            BorderWidthTop = 1,
            BorderWidthBottom = 1,
            CornerRadiusTopLeft = 8,
            CornerRadiusTopRight = 8,
            CornerRadiusBottomLeft = 8,
            CornerRadiusBottomRight = 8,
            ContentMarginLeft = 12,
            ContentMarginRight = 12,
            ContentMarginTop = 8,
            ContentMarginBottom = 8,
        };
        _panel.AddThemeStyleboxOverride("panel", style);
        layer.AddChild(_panel);

        var box = new VBoxContainer();
        box.AddThemeConstantOverride("separation", 5);
        _panel.AddChild(box);

        var row = new HBoxContainer();
        row.AddThemeConstantOverride("separation", 6);
        _pilotBtn = new Button { Text = "Pilot: OFF [F7]", FocusMode = Control.FocusModeEnum.None, SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        _pilotBtn.Pressed += TogglePilot;
        _bugBtn = new Button { Text = "记录异常 [F6]", FocusMode = Control.FocusModeEnum.None };
        _bugBtn.Pressed += () => CaptureBugSnapshot("manual_f6");
        row.AddChild(_pilotBtn);
        row.AddChild(_bugBtn);
        _slBtn = new Button { Text = "SL: OFF [F8]", FocusMode = Control.FocusModeEnum.None };
        _slBtn.Pressed += ToggleSl;
        row.AddChild(_slBtn);
        box.AddChild(row);

        _metaLabel = MakeLabel("Idle · v0.114.0 AlphaDefect", 12, new Color("#8ef2cc"));
        _statusLabel = MakeLabel("F7 开启/关闭接管  ·  F6 记录当前异常快照", 13, Colors.White);
        _valueLabel = MakeLabel("", 13, new Color("#ffd479"));
        _scoresLabel = MakeLabel("", 13, new Color("#d8e1f0"));
        _bugToastLabel = MakeLabel("", 12, new Color("#ff9e7d"));
        _bugToastLabel.Visible = false;
        box.AddChild(_metaLabel);
        box.AddChild(_statusLabel);
        box.AddChild(_valueLabel);
        box.AddChild(_scoresLabel);
        box.AddChild(_bugToastLabel);
    }

    private static Label MakeLabel(string text, int fontSize, Color color)
    {
        var label = new Label
        {
            Text = text,
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        label.AddThemeFontSizeOverride("font_size", fontSize);
        label.AddThemeColorOverride("font_color", color);
        return label;
    }

    private static string ResolveBugReportDir()
    {
        try
        {
            _config ??= TryLoadConfig();
            if (!string.IsNullOrWhiteSpace(_config?.BugReportDir))
                return _config.BugReportDir;
        }
        catch { }
        return Path.Combine(_directory, "bug_reports");
    }

    private static void CaptureBugSnapshot(string trigger)
    {
        try
        {
            string dir = ResolveBugReportDir();
            var status = new
            {
                enabled = _enabled,
                worker_ready = _workerReady,
                model_banner = _modelBanner,
                trained_character = _trainedCharacter,
                status_text = _statusLabel?.Text,
                last_waiting = _lastWaiting,
                last_frame_key = _lastFrameKey,
                same_frame_ms = _sameFrameSinceMs > 0 ? Environment.TickCount64 - _sameFrameSinceMs : 0,
                action_task = _actionTask?.Status.ToString(),
                infer_task = _inferTask?.Status.ToString(),
            };
            string path = BugReporter.SaveSnapshot(trigger, dir, _pilot, status);
            string name = Path.GetFileName(path);
            BugReporter.RecordEvent("snapshot", $"{trigger} -> {name}");
            ShowToast($"已保存异常快照: {name}");
        }
        catch (Exception ex)
        {
            Log.Error("[NecroPilot] failed to save bug snapshot: " + ex);
            ShowToast("快照保存失败: " + ex.GetBaseException().Message);
        }
    }

    private static void ShowToast(string message)
    {
        if (_bugToastLabel == null) return;
        _bugToastLabel.Text = message;
        _bugToastLabel.Visible = true;
        _bugToastUntilMs = Environment.TickCount64 + 8000;
    }

    private static void TogglePilot()
    {
        if (_enabled)
        {
            Stop("Manual control", autoDump: false);
            return;
        }
        try
        {
            _config = LoadConfig();
            _pilot ??= new LivePilot();
            CloseWorker();
            _workerReady = false;
            _modelBanner = "Loading PyTorch model...";
            _worker = StartWorker(_config);
            _readyTask = _worker.StandardOutput.ReadLineAsync();
            _enabled = true;
            Generation++;
            ClearInfer();
            _lastFrameKey = null;
            _sameFrameSinceMs = 0;
            _selectionWaitSinceMs = null;
            _cooldownUntilMs = 0;
            _lastWaiting = "";
            _pilotBtn.Text = "Pilot: ON [F7]";
            _metaLabel.Text = "Loading model (first load takes ~5-15s)...";
            _statusLabel.Text = "Pilot ON · waiting for decision";
            BugReporter.RecordEvent("pilot", "enabled");
        }
        catch (Exception ex)
        {
            Log.Error("[NecroPilot] could not start: " + ex);
            Stop("Start failed: " + ex.GetBaseException().Message, autoDump: true);
        }
    }

    private static void Stop(string reason, bool autoDump)
    {
        BugReporter.RecordEvent("stop", reason);
        if (autoDump)
            CaptureBugSnapshot("auto_" + reason);

        _enabled = false;
        SlPlan.Clear();
        Generation++;
        _pilotBtn.Text = "Pilot: OFF [F7]";
        _statusLabel.Text = reason;
        _valueLabel.Text = "";
        _scoresLabel.Text = "";
        ClearInfer();
        CloseWorker();
        _workerReady = false;
        _readyTask = null;
        _lastFrameKey = null;
        _sameFrameSinceMs = 0;
        _selectionWaitSinceMs = null;
    }

    private static Config? TryLoadConfig()
    {
        string path = Path.Combine(_directory, "pilot.cfg");
        if (!File.Exists(path)) return null;
        return JsonSerializer.Deserialize<Config>(File.ReadAllText(path), JsonOpts);
    }

    private static Config LoadConfig()
    {
        string path = Path.Combine(_directory, "pilot.cfg");
        if (!File.Exists(path)) throw new FileNotFoundException("Missing pilot.cfg in NecroPilot mod folder.");
        return JsonSerializer.Deserialize<Config>(File.ReadAllText(path), JsonOpts)
               ?? throw new InvalidDataException("Invalid pilot.cfg");
    }

    private static Process StartWorker(Config cfg)
    {
        if (string.IsNullOrWhiteSpace(cfg.Python) || !File.Exists(cfg.Python))
            throw new FileNotFoundException($"Python not found: {cfg.Python}");
        string script = Path.Combine(_directory, "live_policy.py");
        if (!File.Exists(script))
            throw new FileNotFoundException($"Missing live_policy.py in {_directory}");

        var start = new ProcessStartInfo(cfg.Python)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = _directory,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardInputEncoding = new System.Text.UTF8Encoding(false),
            StandardOutputEncoding = System.Text.Encoding.UTF8,
        };
        start.ArgumentList.Add("-u");
        start.ArgumentList.Add(script);
        start.ArgumentList.Add("--config");
        start.ArgumentList.Add(Path.Combine(_directory, "pilot.cfg"));
        start.Environment["PYTHONIOENCODING"] = "utf-8";

        var pyPaths = new List<string>();
        if (cfg.PythonPath != null) pyPaths.AddRange(cfg.PythonPath.Where(p => !string.IsNullOrWhiteSpace(p)));
        string? existing = Environment.GetEnvironmentVariable("PYTHONPATH");
        if (!string.IsNullOrWhiteSpace(existing)) pyPaths.Add(existing);
        if (pyPaths.Count > 0)
            start.Environment["PYTHONPATH"] = string.Join(Path.PathSeparator, pyPaths);

        var proc = Process.Start(start) ?? throw new IOException("Could not start live_policy.py");
        proc.ErrorDataReceived += (_, args) =>
        {
            if (args.Data != null)
            {
                Log.Info("[NecroPilot/py] " + args.Data);
                BugReporter.RecordPyStderr(args.Data);
            }
        };
        proc.BeginErrorReadLine();
        return proc;
    }

    private static void CloseWorker()
    {
        var w = Interlocked.Exchange(ref _worker, null);
        if (w == null) return;
        try { if (!w.HasExited) w.Kill(entireProcessTree: true); }
        catch (InvalidOperationException) { }
        finally { w.Dispose(); }
    }

    private static bool TypingInInput()
    {
        try
        {
            var focus = ((SceneTree)Engine.GetMainLoop()).Root.GuiGetFocusOwner();
            return focus is LineEdit or TextEdit or CodeEdit;
        }
        catch { return false; }
    }

    private static bool AnyModifierPressed() =>
        Input.IsPhysicalKeyPressed(Key.Ctrl) || Input.IsPhysicalKeyPressed(Key.Alt) || Input.IsPhysicalKeyPressed(Key.Shift);

    private static void ToggleSl()
    {
        _slEnabled = !_slEnabled;
        _slGeneration++;
        SlPlan.Clear();
        _slBtn.Text = _slEnabled ? "SL: ON [F8]" : "SL: OFF [F8]";
    }

    private static string? ReadRoomSave(RunManager manager)
    {
        if (ReplayField?.GetValue(manager.CombatReplayWriter) is not CombatReplay replay ||
            replay.serializableRun is not { } save)
            return null;
        return JsonSerializer.Serialize(save, JsonSerializationUtility.GetTypeInfo(save));
    }

    private static void Frame()
    {
        if (_bugToastLabel.Visible && Environment.TickCount64 > _bugToastUntilMs)
            _bugToastLabel.Visible = false;

        if (!TypingInInput() && !AnyModifierPressed())
        {
            bool f7 = Input.IsPhysicalKeyPressed(Key.F7);
            if (f7 && !_f7Down) TogglePilot();
            _f7Down = f7;

            bool f6 = Input.IsPhysicalKeyPressed(Key.F6);
            if (f6 && !_f6Down) CaptureBugSnapshot("manual_f6");
            _f6Down = f6;

            bool f8 = Input.IsPhysicalKeyPressed(Key.F8);
            if (f8 && !_f8Down) ToggleSl();
            _f8Down = f8;
        }

        if (!_enabled || Environment.TickCount64 < _nextPollMs) return;
        _nextPollMs = Environment.TickCount64 + 50;

        try { TakeStep(); }
        catch (Exception ex)
        {
            Log.Error("[NecroPilot] step error: " + ex);
            BugReporter.RecordEvent("exception", ex.ToString());
            Stop("Paused · " + ex.GetBaseException().Message, autoDump: true);
        }
    }

    private static void TakeStep()
    {
        if (_worker == null || _worker.HasExited)
        {
            Stop("Policy worker exited", autoDump: true);
            return;
        }

        if (!_workerReady)
        {
            if (_readyTask == null || !_readyTask.IsCompleted) return;
            string? line = _readyTask.GetAwaiter().GetResult();
            if (string.IsNullOrWhiteSpace(line))
            {
                Stop("Worker exited before ready", autoDump: true);
                return;
            }
            var ready = JsonSerializer.Deserialize<ReadyReply>(line, JsonOpts);
            if (ready == null || ready.Type != "ready")
            {
                Stop("Worker init error: " + (ready?.Error ?? line), autoDump: true);
                return;
            }
            _workerReady = true;
            _trainedCharacter = ready.Character;
            _modelBanner = $"{ready.Checkpoint} · v{ready.Version} · {ready.Device}{(ready.Ema ? " EMA" : "")}";
            _metaLabel.Text = _modelBanner;
            Log.Info("[NecroPilot] worker ready: " + _modelBanner);
            BugReporter.RecordEvent("ready", _modelBanner);
        }

        var manager = RunManager.Instance;
        var run = manager.DebugOnlyGetState();
        var player = run == null ? null : LocalContext.GetMe(run.Players);
        if (run == null || player == null || run.GameMode != GameMode.Standard ||
            manager.NetService?.Type != NetGameType.Singleplayer)
        {
            if (_run != null) Stop("Run ended", autoDump: false);
            else _statusLabel.Text = "Waiting for a standard single-player run";
            return;
        }

        if (!string.IsNullOrWhiteSpace(_config?.CharacterFilter) &&
            !string.Equals(player.Character.Id.Entry, _config.CharacterFilter, StringComparison.OrdinalIgnoreCase))
        {
            _statusLabel.Text = $"Waiting for {_config.CharacterFilter} (current: {player.Character.Id.Entry})";
            return;
        }

        if (_run != null && !ReferenceEquals(run, _run))
        {
            _pilot!.ResetRunTracking();
            _run = run;
            Stop("New run detected · manual control", autoDump: false);
            return;
        }
        _run = run;

        var combat = player.Creature.CombatState;
        if (!ReferenceEquals(combat, _slCombat))
            SlPlan.Clear();

        if (!string.IsNullOrEmpty(_trainedCharacter) &&
            !string.Equals(_trainedCharacter, player.Character.Id.Entry, StringComparison.OrdinalIgnoreCase))
        {
            _metaLabel.Text = _modelBanner + $"  [⚠ trained for {_trainedCharacter}]";
        }

        if (((SceneTree)Engine.GetMainLoop()).Paused)
        {
            Stop("Game paused · manual control", autoDump: false);
            return;
        }

        if (_actionTask is { IsCompleted: false })
        {
            if (Environment.TickCount64 - _actionStartedMs > 5000)
                Stop("Game action timed out", autoDump: true);
            return;
        }
        if (_actionTask is { IsFaulted: true })
        {
            Log.Error("[NecroPilot] action fault: " + _actionTask.Exception);
            BugReporter.RecordEvent("action_fault", _actionTask.Exception!.ToString());
            Stop("Action failed · " + _actionTask.Exception!.GetBaseException().Message, autoDump: true);
            return;
        }
        _actionTask = null;

        if (Environment.TickCount64 < _cooldownUntilMs)
            return;

        if (_inferTask is { IsCompleted: false })
        {
            if (Environment.TickCount64 - _inferStartedMs > _inferTimeoutMs)
                Stop("Inference timed out", autoDump: true);
            return;
        }

        var pending = _pilot!.Detect(run, player, out var waiting);
        if (pending == null)
        {
            if (waiting != _lastWaiting)
            {
                _lastWaiting = waiting;
                if (!string.IsNullOrEmpty(waiting)) BugReporter.RecordEvent("wait", waiting);
            }
            if (waiting == "run over") Stop("Run over", autoDump: false);
            else if (waiting.StartsWith("your choice", StringComparison.Ordinal)) Stop(waiting, autoDump: true);
            else if (waiting.StartsWith("card selection: ", StringComparison.Ordinal))
            {
                _selectionWaitSinceMs ??= Environment.TickCount64;
                if (Environment.TickCount64 - _selectionWaitSinceMs.Value >= 6000)
                    Stop("Card selection stalled", autoDump: true);
                else
                    _statusLabel.Text = "Waiting for " + waiting;
            }
            else
            {
                _statusLabel.Text = string.IsNullOrEmpty(waiting) ? "Waiting for game decision" : waiting;
            }
            if (!waiting.StartsWith("card selection: ", StringComparison.Ordinal))
                _selectionWaitSinceMs = null;
            return;
        }
        _selectionWaitSinceMs = null;
        _lastWaiting = "";

        var decision = pending.Decision;
        if (decision.Choices.Count == 0) return;

        bool useSl = _slEnabled && combat != null && CombatManager.Instance.IsInProgress &&
            decision.Kind is DecisionKind.Combat or DecisionKind.CardSelect;
        if (pending.AutoStep || (!useSl && decision.Choices.Count == 1))
        {
            var only = decision.Choices[0];
            string autoKey = pending.Context + "|" + only.Label + "|" + run.ActFloor + "|" + player.Gold + "|" + player.Creature.CurrentHp;
            if (!pending.Repeatable && autoKey == _lastFrameKey)
            {
                if (Environment.TickCount64 - _sameFrameSinceMs > 3500)
                    Stop($"Auto step stalled ({pending.Context})", autoDump: true);
                return;
            }
            _lastFrameKey = autoKey;
            _sameFrameSinceMs = Environment.TickCount64;
            _statusLabel.Text = $"Auto · {only.Label}";
            _scoresLabel.Text = "";
            BugReporter.RecordEvent("auto", $"{pending.Context} -> {only.Label}");
            LogRewardChoice(run, decision, only, "auto", player.HasOpenPotionSlots);
            ExecuteChoice(pending, only);
            return;
        }

        byte[] frameBytes;
        try
        {
            var obs = Observation.Build(_pilot.Session, decision, labels: false);
            frameBytes = obs.Serialize();
        }
        catch (Exception ex)
        {
            Log.Info("[NecroPilot] transient observation error (retrying): " + ex.GetBaseException().Message);
            BugReporter.RecordEvent("obs_transient", ex.GetBaseException().Message);
            _statusLabel.Text = "Waiting for stable state...";
            return;
        }

        string frameKey = pending.Context + ":" + Convert.ToHexString(SHA256.HashData(frameBytes));

        if (frameKey == _lastFrameKey && _inferTask == null)
        {
            if (Environment.TickCount64 - _sameFrameSinceMs > 3500)
                Stop($"Action not accepted ({pending.Context})", autoDump: true);
            return;
        }

        if (_inferTask != null && _inferFrameKey != frameKey)
            ClearInfer();

        if (_inferTask == null)
        {
            if (useSl && SlPlan.TryPeek(out var planned))
            {
                if (PlanMatches(planned, frameBytes, decision.Choices.Count))
                {
                    SlPlan.Dequeue();
                    CommitChoice(pending, decision.Choices[planned.Index], frameKey, "SL");
                    return;
                }
                SlPlan.Clear();
                BugReporter.RecordEvent("sl", "Action frame mismatch; switched to policy");
                Log.Info($"[NecroPilot/SL] F{run.TotalFloor} action mismatch; switched to policy");
            }
            _inferFrameKey = frameKey;
            _inferStartedMs = Environment.TickCount64;
            _inferSlGeneration = _slGeneration;
            string? roomSave = null, room = null;
            if (useSl && decision.Kind == DecisionKind.Combat && !ReferenceEquals(combat, _slCombat))
            {
                _slCombat = combat;
                room = run.CurrentRoom?.RoomType.ToString();
                if (room is "Monster" or "Elite" or "Boss")
                {
                    try { roomSave = ReadRoomSave(manager); }
                    catch (Exception ex) { BugReporter.RecordEvent("sl", "Room save unavailable: " + ex.Message); }
                    if (roomSave != null) _statusLabel.Text = "SL · 正在搜索当前战斗";
                }
            }
            string req = JsonSerializer.Serialize(new { type = "infer", frame = Convert.ToBase64String(frameBytes), room_save = roomSave, room });
            _inferTimeoutMs = roomSave == null ? 8000 : 18000;
            _inferTask = Query(_worker, req);
            return;
        }

        string? rawReply = _inferTask.GetAwaiter().GetResult();
        ClearInfer();
        if (string.IsNullOrWhiteSpace(rawReply))
        {
            Stop("Policy worker closed stdout", autoDump: true);
            return;
        }

        var reply = JsonSerializer.Deserialize<InferReply>(rawReply, JsonOpts);
        if (reply == null || reply.Type != "infer")
        {
            Stop("Policy error: " + (reply?.Error ?? rawReply), autoDump: true);
            return;
        }
        if (reply.Index < 0 || reply.Index >= decision.Choices.Count)
        {
            Stop($"Invalid policy index {reply.Index}/{decision.Choices.Count}", autoDump: true);
            return;
        }

        var chosen = decision.Choices[reply.Index];
        LogRewardChoice(run, decision, chosen, "policy", player.HasOpenPotionSlots, reply.Top);
        _valueLabel.Text = $"Value: {reply.Value:+0.00;-0.00;0.00}   ·   {reply.Ms:F1} ms   ·   {decision.Choices.Count} opts";
        _scoresLabel.Text = string.Join("\n", (reply.Top ?? Array.Empty<TopItem>()).Select((t, rank) =>
        {
            string lbl = t.Index >= 0 && t.Index < decision.Choices.Count ? decision.Choices[t.Index].Label : $"#{t.Index}";
            string mark = rank == 0 ? "▶ " : "  ";
            return $"{mark}{t.Prob * 100:F1}%  ({t.Logit:+0.00;-0.00;0.00})  {lbl}";
        }));

        BugReporter.RecordEvent("choose", $"{pending.Context} -> {chosen.Label}", new
        {
            val = Math.Round(reply.Value, 3),
            ms = Math.Round(reply.Ms, 1),
            n = decision.Choices.Count,
            top = (reply.Top ?? Array.Empty<TopItem>()).Select(t => new
            {
                idx = t.Index,
                lbl = t.Index >= 0 && t.Index < decision.Choices.Count ? decision.Choices[t.Index].Label : $"#{t.Index}",
                p = Math.Round(t.Prob, 3),
            }).ToArray(),
        });

        if (reply.SlStatus != null)
        {
            BugReporter.RecordEvent("sl", reply.SlStatus, new { completed = reply.SlCompleted });
            Log.Info($"[NecroPilot/SL] F{run.TotalFloor} {reply.SlStatus} completed={reply.SlCompleted} " +
                $"plan_steps={reply.SlPlan?.Length ?? 0}");
        }
        if (useSl && _inferSlGeneration == _slGeneration && reply.SlPlan is { Length: > 0 } plan &&
            PlanMatches(plan[0], frameBytes, decision.Choices.Count))
        {
            foreach (var step in plan.Skip(1)) SlPlan.Enqueue(step);
            CommitChoice(pending, decision.Choices[plan[0].Index], frameKey, "SL");
            return;
        }
        CommitChoice(pending, chosen, frameKey, "▶");
    }

    private static bool PlanMatches(PlanStep step, byte[] frame, int choiceCount)
    {
        if (step.Index < 0 || step.Index >= choiceCount) return false;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hash.AppendData(frame.AsSpan(0, 48));
        hash.AppendData(frame.AsSpan(60));
        return step.Guard == Convert.ToHexString(hash.GetHashAndReset());
    }

    private static void LogRewardChoice(RunState run, Decision decision, Choice chosen, string source,
        bool openPotionSlot, TopItem[]? top = null)
    {
        if (decision.Kind is not (DecisionKind.Rewards or DecisionKind.CardReward)) return;
        var candidates = string.Join(" | ", decision.Choices.Select((choice, index) => $"{index}:{choice.Label}"));
        var scores = string.Join(" | ", (top ?? Array.Empty<TopItem>()).Select(t => $"{t.Index}:{t.Prob:P1}"));
        Log.Info($"[NecroPilot/reward] F{run.TotalFloor} {run.CurrentRoom?.RoomType} {decision.Kind} " +
            $"source={source} selected={decision.Choices.IndexOf(chosen)}:{chosen.Label} " +
            $"open_potion_slot={openPotionSlot} candidates=[{candidates}] top=[{scores}]");
    }

    private static void CommitChoice(LivePilot.Pending pending, Choice chosen, string frameKey, string prefix)
    {
        _lastFrameKey = frameKey;
        _sameFrameSinceMs = Environment.TickCount64;
        _statusLabel.Text = $"{prefix} {chosen.Label}";
        ExecuteChoice(pending, chosen);
    }

    private static void ExecuteChoice(LivePilot.Pending pending, Choice chosen)
    {
        _pilot!.Session.Steps++;
        _actionStartedMs = Environment.TickCount64;
        int extraDelay = _config?.ActionDelayMs ?? 0;
        _cooldownUntilMs = Environment.TickCount64 + Math.Max(pending.CooldownMs, extraDelay);
        _actionTask = chosen.Run();
    }

    private static void ClearInfer()
    {
        if (_inferTask is { IsFaulted: true }) _ = _inferTask.Exception;
        _inferTask = null;
        _inferFrameKey = null;
    }

    private static async Task<string?> Query(Process worker, string request)
    {
        await worker.StandardInput.WriteLineAsync(request).ConfigureAwait(false);
        await worker.StandardInput.FlushAsync().ConfigureAwait(false);
        return await worker.StandardOutput.ReadLineAsync().ConfigureAwait(false);
    }

    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    private sealed record Config(
        [property: JsonPropertyName("python")] string Python,
        [property: JsonPropertyName("pythonpath")] string[]? PythonPath = null,
        [property: JsonPropertyName("checkpoint")] string? Checkpoint = null,
        [property: JsonPropertyName("device")] string? Device = "cpu",
        [property: JsonPropertyName("use_ema")] bool UseEma = true,
        [property: JsonPropertyName("character_filter")] string? CharacterFilter = null,
        [property: JsonPropertyName("action_delay_ms")] int ActionDelayMs = 150,
        [property: JsonPropertyName("bug_report_dir")] string? BugReportDir = null);

    private sealed record ReadyReply(string Type, string? Checkpoint, int Version, string? Device, bool Ema, string? Character, string? Error);
    private sealed record TopItem(int Index, double Prob, double Logit);
    private sealed record PlanStep(string Guard, int Index);
    private sealed record InferReply(string Type, int Index, double Value, double Ms, TopItem[]? Top, string? Error,
        [property: JsonPropertyName("sl_plan")] PlanStep[]? SlPlan,
        [property: JsonPropertyName("sl_status")] string? SlStatus,
        [property: JsonPropertyName("sl_completed")] int SlCompleted);
}
