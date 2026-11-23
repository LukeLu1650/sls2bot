# 本地模型回放 Mod

`NecroReplay` 在 v0.113.1 的游戏窗口展示此次模型的三段录制轨迹。包含手牌、敌人意图、生命、Osty、选中动作和得分最高的五个候选。可以切换对局、自动播放、前后步进或拖动时间轴。画面是已录制决策回放，当前实机对局由玩家照常控制。

安装目录为 Steam 游戏的 `mods/NecroReplay/`，需要同目录的 `NecroReplay.dll`、`NecroReplay.json` 和 `replay.dat`。启动游戏并确认模组已启用，在菜单点击 **Policy Replay** 或按 **F8**。可从本地 ZIP 将其中的 `NecroReplay` 文件夹解压到另一份匹配的游戏安装的 `mods/` 下。

本机已安装三个文件，使用 v0.113.1 程序集编译通过。直接启动的自动验证在 Steam 登录初始化时报 `ConnectToGlobalUser failed`，运行时画面须从已登录的 Steam 中验证。

用已有 Windows .NET 9 SDK 重新打包时，从仓库根目录执行。源目录也可以换成新训练输出，其中应包含 `report.json` 和 `replay.json`。

```powershell
$py = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $py scripts/package_replay.py results/windows-v0.113.1/standard-continuation --out artifacts/NecroReplay-next.zip
```

展板的评分为模型原始分数，柱长仅反映当前候选的相对位置。关卡和结局直接取自引擎记录。若遇到引擎错误，回放在最后一个已送出的动作处结束，并显示该局的错误状态。[当前模型和动作说明](current-policy.md) · [Mod 源码](../mod/Replay.cs) · [游戏 Mod 接口参考](https://github.com/fresh-milkshake/Modding-Tutorial)
