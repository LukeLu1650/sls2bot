using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Modding;

namespace NecroReplay;

[ModInitializer(nameof(Init))]
public static class Replay
{
    private static readonly Color Ink = new("#0d1018");
    private static readonly Color Panel = new("#191e2c");
    private static readonly Color Mint = new("#8ef2cc");
    private static readonly Color Muted = new("#a8b2ce");
    private static JsonElement _data;
    private static JsonElement _frames;
    private static JsonElement _run;
    private static PanelContainer _panel = null!;
    private static Label _position = null!, _stats = null!, _scene = null!, _enemies = null!, _hand = null!;
    private static Label _selected = null!;
    private static readonly Label[] CandidateLabels = new Label[5];
    private static readonly ProgressBar[] ScoreBars = new ProgressBar[5];
    private static HSlider _timeline = null!;
    private static Button _play = null!;
    private static bool _playing, _wasF8;
    private static ulong _nextFrame;

    public static void Init()
    {
        try
        {
            string dir = Path.GetDirectoryName(typeof(Replay).Assembly.Location)!;
            using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "replay.dat")));
            _data = json.RootElement.Clone();
            if (_data.GetProperty("schema").GetInt32() != 1 || _data.GetProperty("episodes").GetArrayLength() == 0)
                throw new InvalidDataException("Expected a schema 1 replay with at least one episode.");
            var tree = (SceneTree)Engine.GetMainLoop();
            tree.ProcessFrame += Setup;
        }
        catch (Exception ex)
        {
            Log.Error("[NecroReplay] " + ex);
        }
    }

    private static void Setup()
    {
        var tree = (SceneTree)Engine.GetMainLoop();
        tree.ProcessFrame -= Setup;
        try
        {
            Build(tree.Root);
            tree.ProcessFrame += OnFrame;
            Log.Info($"[NecroReplay] loaded {_data.GetProperty("episodes").GetArrayLength()} recorded episodes.");
            if (System.Environment.GetEnvironmentVariable("NECRO_REPLAY_SMOKE") == "1")
            {
                Show(true);
                _timeline.Value = Math.Min(2, _frames.GetArrayLength() - 1);
                Log.Info("[NecroReplay] opened panel and rendered a recorded decision.");
            }
        }
        catch (Exception ex)
        {
            Log.Error("[NecroReplay] setup failed: " + ex);
        }
    }

    private static void Build(Node root)
    {
        var layer = new CanvasLayer { Layer = 125, Name = "NecroReplay" };
        root.AddChild(layer);
        var open = MakeButton("Policy Replay · F8", () => Show(!_panel.Visible));
        open.AnchorLeft = open.AnchorRight = 1;
        open.OffsetLeft = -222;
        open.OffsetRight = -22;
        open.OffsetTop = 98;
        layer.AddChild(open);

        _panel = new PanelContainer { Visible = false };
        layer.AddChild(_panel);
        _panel.SetAnchorsAndOffsetsPreset(Control.LayoutPreset.FullRect);
        _panel.OffsetLeft = _panel.OffsetTop = 42;
        _panel.OffsetRight = _panel.OffsetBottom = -42;
        _panel.AddThemeStyleboxOverride("panel", Box(Ink, 24));
        var stack = new VBoxContainer();
        stack.AddThemeConstantOverride("separation", 12);
        _panel.AddChild(stack);

        var header = new HBoxContainer();
        var title = MakeLabel("Necrobinder  /  Policy Replay", 32, Colors.White);
        title.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        header.AddChild(title);
        header.AddChild(MakeButton("Close · F8", () => Show(false)));
        stack.AddChild(header);
        stack.AddChild(MakeLabel($"STANDARD    ·    {Value(_data, "build")}    ·    {Value(_data, "parameters")} PARAMETERS    ·    ZERO SEARCH", 15, Mint));
        if (_data.TryGetProperty("first_combat", out var firstCombat))
        {
            var before = firstCombat.GetProperty("base");
            var after = firstCombat.GetProperty("student");
            stack.AddChild(MakeLabel($"A0 first combat   {before[0]}/{before[1]} → {after[0]}/{after[1]}" +
                $"     ·     {Value(_data, "demonstrations")} labeled decisions", 18, Muted));
        }
        _position = MakeLabel("", 19, Muted);
        stack.AddChild(_position);

        var rooms = new ScrollContainer { SizeFlagsVertical = Control.SizeFlags.ExpandFill };
        stack.AddChild(rooms);
        var body = new HBoxContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill };
        body.AddThemeConstantOverride("separation", 14);
        rooms.AddChild(body);
        var board = Section(body, "Current decision");
        _scene = MakeLabel("", 22, Mint);
        _stats = MakeLabel("", 18, Colors.White);
        _enemies = MakeLabel("", 18, Colors.White);
        _hand = MakeLabel("", 18, Muted);
        foreach (var item in new[] { _scene, _stats, _enemies, _hand }) board.AddChild(item);
        var actions = Section(body, "Model action scores");
        _selected = MakeLabel("", 20, Mint);
        actions.AddChild(_selected);
        for (int i = 0; i < CandidateLabels.Length; i++)
        {
            CandidateLabels[i] = MakeLabel("", 16, Colors.White);
            ScoreBars[i] = new ProgressBar { MinValue = 0, MaxValue = 1, ShowPercentage = false,
                CustomMinimumSize = new Vector2(0, 7), MouseFilter = Control.MouseFilterEnum.Ignore };
            ScoreBars[i].AddThemeStyleboxOverride("background", Box(Ink, 0));
            ScoreBars[i].AddThemeStyleboxOverride("fill", Box(Mint, 0));
            actions.AddChild(CandidateLabels[i]);
            actions.AddChild(ScoreBars[i]);
        }

        _timeline = new HSlider { MinValue = 0, Step = 1, CustomMinimumSize = new Vector2(0, 22) };
        _timeline.ValueChanged += _ => Render();
        stack.AddChild(_timeline);
        var controls = new HBoxContainer();
        controls.AddThemeConstantOverride("separation", 10);
        var picker = new OptionButton { FocusMode = Control.FocusModeEnum.None };
        int index = 1;
        foreach (var run in _data.GetProperty("episodes").EnumerateArray())
            picker.AddItem($"Run {index++}  ·  A{Value(run, "ascension")}  ·  {Value(run, "seed")}");
        picker.ItemSelected += i => LoadRun((int)i);
        controls.AddChild(picker);
        controls.AddChild(MakeButton("Previous", () => Move(-1)));
        _play = MakeButton("Play", TogglePlay);
        controls.AddChild(_play);
        controls.AddChild(MakeButton("Next", () => Move(1)));
        stack.AddChild(controls);
        LoadRun(0);
    }

    private static VBoxContainer Section(HBoxContainer parent, string title)
    {
        var card = new PanelContainer { SizeFlagsHorizontal = Control.SizeFlags.ExpandFill,
            CustomMinimumSize = new Vector2(320, 0) };
        card.AddThemeStyleboxOverride("panel", Box(Panel, 18));
        parent.AddChild(card);
        var body = new VBoxContainer();
        body.AddThemeConstantOverride("separation", 10);
        card.AddChild(body);
        body.AddChild(MakeLabel(title, 17, Muted));
        return body;
    }

    private static void LoadRun(int index)
    {
        Pause();
        _run = _data.GetProperty("episodes")[index];
        _frames = _run.GetProperty("frames");
        _timeline.MaxValue = _frames.GetArrayLength() - 1;
        _timeline.Value = 0;
        Render();
    }

    private static void Render()
    {
        if (_frames.ValueKind != JsonValueKind.Array || _frames.GetArrayLength() == 0) return;
        int i = (int)_timeline.Value;
        var frame = _frames[i];
        _position.Text = $"Decision {i + 1} / {_frames.GetArrayLength()}     ·     Recorded run: {Value(_run, "result").Replace('_', ' ')}";
        _scene.Text = Value(frame, "scene").Replace('_', ' ');
        _stats.Text = $"Floor {Value(frame, "floor")}   ·   Turn {Value(frame, "round")}\n" +
                      $"HP {Value(frame, "hp")} / {Value(frame, "max_hp")}   ·   Block {Value(frame, "block")}\n" +
                      $"Energy {Value(frame, "energy")}   ·   Osty HP {Value(frame, "osty")}";
        _enemies.Text = string.Join("\n", frame.GetProperty("enemies").EnumerateArray().Select(enemy =>
            $"{Value(enemy, "name")}  {Value(enemy, "hp")} / {Value(enemy, "max_hp")} HP   ·   " +
            string.Join(", ", enemy.GetProperty("intents").EnumerateArray().Select(intent =>
                $"{Value(intent, "kind")} {Value(intent, "damage")}"))));
        _hand.Text = "Hand  ·  " + string.Join("   ", frame.GetProperty("hand").EnumerateArray()
            .Select(card => $"{Value(card, "name")} [{Value(card, "cost")}]"));
        _selected.Text = "Selected  ·  " + Value(frame, "selected");
        var scores = frame.GetProperty("scores");
        double maximum = scores[0].GetProperty("score").GetDouble();
        double minimum = scores[scores.GetArrayLength() - 1].GetProperty("score").GetDouble();
        for (int j = 0; j < CandidateLabels.Length; j++)
        {
            bool visible = j < scores.GetArrayLength();
            CandidateLabels[j].Visible = ScoreBars[j].Visible = visible;
            if (!visible) continue;
            var item = scores[j];
            double score = item.GetProperty("score").GetDouble();
            CandidateLabels[j].Text = $"{score:+0.00;-0.00;0.00}   {Value(item, "label")}";
            ScoreBars[j].Value = maximum == minimum ? 1 : 0.1 + 0.9 * (score - minimum) / (maximum - minimum);
        }
    }

    private static string Value(JsonElement item, string key) =>
        item.TryGetProperty(key, out var value) && value.ValueKind != JsonValueKind.Null ? value.ToString() : "—";

    private static StyleBoxFlat Box(Color color, int margin) => new()
    {
        BgColor = color, CornerRadiusBottomLeft = 10, CornerRadiusBottomRight = 10,
        CornerRadiusTopLeft = 10, CornerRadiusTopRight = 10,
        ContentMarginLeft = margin, ContentMarginTop = margin, ContentMarginRight = margin, ContentMarginBottom = margin
    };

    private static Label MakeLabel(string text, int size, Color color)
    {
        var label = new Label { Text = text, AutowrapMode = TextServer.AutowrapMode.WordSmart };
        label.AddThemeFontSizeOverride("font_size", size);
        label.AddThemeColorOverride("font_color", color);
        return label;
    }

    private static Button MakeButton(string text, Action pressed)
    {
        var button = new Button { Text = text, FocusMode = Control.FocusModeEnum.None };
        button.Pressed += pressed;
        return button;
    }

    private static void Show(bool visible)
    {
        _panel.Visible = visible;
        if (!visible) Pause();
    }

    private static void Pause()
    {
        _playing = false;
        _play.Text = "Play";
    }

    private static void TogglePlay()
    {
        if (_playing) { Pause(); return; }
        if (_timeline.Value >= _timeline.MaxValue) _timeline.Value = 0;
        _playing = true;
        _play.Text = "Pause";
        _nextFrame = Time.GetTicksMsec() + 600;
    }

    private static void Move(int offset)
    {
        Pause();
        _timeline.Value = Math.Clamp(_timeline.Value + offset, 0, _timeline.MaxValue);
    }

    private static void OnFrame()
    {
        bool down = Input.IsPhysicalKeyPressed(Key.F8);
        if (down && !_wasF8) Show(!_panel.Visible);
        _wasF8 = down;
        if (!_playing || Time.GetTicksMsec() < _nextFrame) return;
        _nextFrame = Time.GetTicksMsec() + 600;
        _timeline.Value++;
        if (_timeline.Value >= _timeline.MaxValue) Pause();
    }
}
