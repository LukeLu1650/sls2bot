using System.Diagnostics;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Runs;
using Sts2Ai.Shared.Decisions;
using Sts2Ai.Shared.Encoding;
using Sts2AiMod;
using Environment = System.Environment;

namespace NecroPilot;

[ModInitializer(nameof(Init))]
public static class Live
{
    private static bool _enabled, _pressed;
    internal static int Generation { get; private set; }
    internal static bool Active(int generation) => _enabled && generation == Generation;
    private static string _directory = "";
    private static Label _status = null!;
    private static Label _scores = null!;
    private static Button _toggle = null!;
    private static Pilot? _pilot;
    private static Observation? _observation;
    private static Process? _worker;
    private static Task<string?>? _score;
    private static Task? _action;
    private static RunState? _run;
    private static (int Act, int Floor)? _triggerRoom;
    private static object? _trigger;
    private static (int Act, int Floor)? _removalRoom;
    private static string? _requested, _previous;
    private static long _scoreStarted, _nextPoll, _actionStarted;
    private static long? _selectionWaitSince;

    public static void Init()
    {
        _directory = Path.GetDirectoryName(typeof(Live).Assembly.Location)!;
        ((SceneTree)Engine.GetMainLoop()).ProcessFrame += Setup;
        AppDomain.CurrentDomain.ProcessExit += (_, _) => CloseWorker();
    }

    private static void Setup()
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        tree.ProcessFrame -= Setup;
        var layer = new CanvasLayer { Layer = 123 };
        tree.Root.AddChild(layer);
        var box = new VBoxContainer { OffsetLeft = 24, OffsetTop = 140, OffsetRight = 330 };
        _toggle = new Button { Text = "NecroPilot · OFF [F7]", FocusMode = Control.FocusModeEnum.None };
        _toggle.Pressed += Toggle;
        _status = new Label();
        _status.AddThemeColorOverride("font_outline_color", Colors.Black);
        _status.AddThemeConstantOverride("outline_size", 6);
        box.AddChild(_toggle);
        box.AddChild(_status);
        _scores = new Label { MouseFilter = Control.MouseFilterEnum.Ignore };
        _scores.AddThemeColorOverride("font_outline_color", Colors.Black);
        _scores.AddThemeConstantOverride("outline_size", 6);
        box.AddChild(_scores);
        layer.AddChild(box);
        tree.ProcessFrame += Frame;
        Log.Info("[NecroPilot] takeover controls loaded (off).");
    }

    private static void Toggle()
    {
        if (_enabled) { Stop("Manual control"); return; }
        try
        {
            var config = JsonSerializer.Deserialize<Config>(File.ReadAllText(Path.Combine(_directory, "pilot.cfg")))!;
            _observation ??= new Observation(Path.Combine(_directory, "names.dat"));
            _pilot ??= MakePilot();
            _worker = StartWorker(config.Python);
            _previous = _requested = null;
            _run = null;
            _score = null;
            Generation++;
            _enabled = true;
            _toggle.Text = "NecroPilot · ON [F7]";
            _status.Text = "Waiting for an active standard run";
        }
        catch (Exception ex)
        {
            Stop(ex.Message);
            Log.Error("[NecroPilot] could not start: " + ex);
        }
    }

    private static Pilot MakePilot()
    {
        var vocab = Vocab.Load(Path.Combine(_directory, "vocab.dat"));
        return new Pilot(new DecisionBuilder(new StateEncoder(vocab), vocab, new RunHistorySummary()));
    }

    private sealed record Config(string Python);

    private static Process StartWorker(string executable)
    {
        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = _directory,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
            StandardInputEncoding = new System.Text.UTF8Encoding(false),
            StandardOutputEncoding = System.Text.Encoding.UTF8
        };
        start.ArgumentList.Add("-u");
        start.ArgumentList.Add(Path.Combine(_directory, "live_policy.py"));
        start.Environment["PYTHONIOENCODING"] = "utf-8";
        var process = Process.Start(start) ?? throw new IOException("Could not run the local policy worker.");
        process.ErrorDataReceived += (_, args) =>
        {
            if (args.Data != null) Console.Error.WriteLine("[NecroPilot] " + args.Data);
        };
        process.BeginErrorReadLine();
        return process;
    }

    private static void Stop(string reason)
    {
        _enabled = false;
        Generation++;
        _toggle.Text = "NecroPilot · OFF [F7]";
        _status.Text = reason;
        _scores.Text = "";
        CloseWorker();
        _score = null;
        _requested = _previous = null;
        _triggerRoom = null;
        _trigger = null;
        _removalRoom = null;
        _selectionWaitSince = null;
    }

    private static void CloseWorker()
    {
        var worker = Interlocked.Exchange(ref _worker, null);
        if (worker == null) return;
        try { if (!worker.HasExited) worker.Kill(entireProcessTree: true); }
        catch (InvalidOperationException) { }
        finally { worker.Dispose(); }
    }

    private static void Frame()
    {
        bool down = Input.IsPhysicalKeyPressed(Key.F7);
        if (down && !_pressed) Toggle();
        _pressed = down;
        if (!_enabled || Environment.TickCount64 < _nextPoll) return;
        _nextPoll = Environment.TickCount64 + 45;
        try { TakeStep(); }
        catch (Exception ex)
        {
            Log.Error("[NecroPilot] " + ex);
            Stop("AI paused · " + ex.GetBaseException().Message);
        }
    }

    private static void TakeStep()
    {
        var manager = RunManager.Instance;
        var run = manager.DebugOnlyGetState();
        var player = run == null ? null : LocalContext.GetMe(run.Players);
        if (run == null || player == null || run.GameMode != GameMode.Standard ||
            manager.NetService?.Type != NetGameType.Singleplayer || player.Character.Id.Entry != "NECROBINDER")
        {
            if (_run != null) Stop("Run ended");
            else _status.Text = "Waiting for a standard single-player Necrobinder run";
            return;
        }
        if (_run != null && !ReferenceEquals(run, _run)) { Stop("New run · manual control"); return; }
        _run = run;
        if (_worker!.HasExited) { Stop("Local policy worker exited"); return; }
        if (((SceneTree)Engine.GetMainLoop()).Paused) { Stop("Game paused · manual control"); return; }
        if (_action is { IsCompleted: false })
        {
            if (Environment.TickCount64 - _actionStarted > 4000) Stop("Game action still pending · manual control");
            return;
        }
        if (_action is { IsFaulted: true }) throw _action.Exception!;
        _action = null;

        if (_score != null && !_score.IsCompleted)
        {
            if (Environment.TickCount64 - _scoreStarted > 3500) Stop("Policy response timed out");
            return;
        }
        var pending = _pilot!.Detect(run, player, out var waiting);
        if (pending == null)
        {
            if (waiting == "run over") Stop("Run over");
            else if (waiting.StartsWith("your choice", StringComparison.Ordinal)) Stop(waiting);
            else if (waiting.StartsWith("card selection: ", StringComparison.Ordinal))
            {
                _scores.Text = "";
                _selectionWaitSince ??= Environment.TickCount64;
                if (Environment.TickCount64 - _selectionWaitSince.Value >= 5000)
                {
                    Log.Info("[NecroPilot] No ready action for " + waiting);
                    Stop("Card selection needs manual control");
                }
                else _status.Text = "Waiting for " + waiting;
            }
            else _status.Text = string.IsNullOrEmpty(waiting) ? "Waiting for a game decision" : waiting;
            if (!waiting.StartsWith("card selection: ", StringComparison.Ordinal)) _selectionWaitSince = null;
            ClearScore();
            return;
        }
        _selectionWaitSince = null;
        if (pending.Context.StartsWith("crystal sphere", StringComparison.Ordinal))
        {
            Stop("Crystal Sphere · manual control");
            return;
        }
        var options = pending.Decision.Actions
            .Where(a => a.Type != ActionType.DiscardPotion &&
                        (pending.Decision.Type == DecisionType.Combat || a.Type != ActionType.UsePotion))
            .ToArray();
        if (options.Length == 0 && (pending.Decision.Actions.Count != 0 || !IsForced(pending.Context)))
        {
            Stop("This choice needs manual control");
            return;
        }
        var room = (run.CurrentActIndex, run.ActFloor);
        string request = _observation!.Encode(run, player, pending.Decision, options,
            _triggerRoom == room ? _trigger : null, _removalRoom == room ? "remove_card" : null);
        // State changed after the request, such as when the player took an action manually.
        if (_score != null && !string.Equals(request, _requested, StringComparison.Ordinal)) ClearScore();
        string signature = pending.Context + "\n" + request;
        if (signature == _previous)
        {
            if (Environment.TickCount64 - _actionStarted > 1700) Stop("Game did not accept the last action");
            return;
        }

        int selected = 0;
        if (options.Length > 1)
        {
            if (_score == null)
            {
                _requested = request;
                _scoreStarted = Environment.TickCount64;
                _score = Query(_worker, request);
                return;
            }
            var response = JsonSerializer.Deserialize<Reply>(_score.GetAwaiter().GetResult()!)!;
            if (response.Index < 0 || response.Index >= options.Length) throw new InvalidDataException("Invalid policy answer.");
            selected = response.Index;
            _status.Text = $"{options[selected].Label}  ·  {response.Ms:F0} ms";
            _scores.Text = string.Join("\n", response.Top.Select(score =>
                $"{score.Score:+0.00;-0.00;0.00}  {options[score.Index].Label}"));
            ClearScore();
        }
        else
        {
            _status.Text = options.Length == 0 ? pending.Context : options[0].Label;
            _scores.Text = "";
        }
        _previous = signature;
        _actionStarted = Environment.TickCount64;
        var action = options.Length == 0 ? new LegalAction() : options[selected];
        if (pending.Decision.Type != DecisionType.CardSelect)
            _removalRoom = action.Type == ActionType.RemoveCardService ? room : null;
        if (pending.Decision.Type == DecisionType.Combat)
        {
            _trigger = _observation.CombatTrigger(action);
            _triggerRoom = room;
        }
        _action = pending.Execute(action);
    }

    private static bool IsForced(string context) => context is
        "claim gold" or "open shop" or "card select: confirm" or "hand select: confirm" or "rest site: proceed" or
        "open chest" or "leave treasure room" or "event dialogue" or "event proceed" or
        "fake merchant: close counter" or "fake merchant: leave";

    private static void ClearScore()
    {
        if (_score is { IsFaulted: true }) _ = _score.Exception;
        _score = null;
        _requested = null;
    }

    private sealed record Candidate(int Index, double Score);
    private sealed record Reply(int Index, double Ms, Candidate[] Top);

    private static async Task<string?> Query(Process worker, string request)
    {
        await worker.StandardInput.WriteLineAsync(request).ConfigureAwait(false);
        await worker.StandardInput.FlushAsync().ConfigureAwait(false);
        return await worker.StandardOutput.ReadLineAsync().ConfigureAwait(false);
    }
}
