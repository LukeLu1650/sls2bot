# 实机接管与建议 Mod（NecroPilot v0.2.0 · v0.114.0）

`NecroPilot v0.2.0` 面向 Steam **`v0.114.0`**，直接复用路线 A（`external/AlphaDefect-Public/game-v0.114.0`）的原生 C# 观测编码器（`Observation` + `Vocab` + `RawPreview` + `DrawTop`）与 PyTorch `PolicyValueNet` 检查点，实现与无头训练 **100% 同源** 的实机决策。

> 原 `NecroReplay` 回放 Mod 因仅能展示静态文本日志且无法稳定可视化复刻到局内种子，现已退役移除。若要直观查看模型在特定种子上的表现，请直接在实机内输入该种子并开启 `NecroPilot`。

## 功能与快捷键

| 快捷键 / 按钮 | 模式 | 说明 |
| --- | --- | --- |
| **F7** / `Pilot` | 自动接管 | 自动执行模型选择的动作（含战斗、选牌、卡牌奖励、战后奖励、宝箱、商店、休息处、事件、水晶球、地图） |
| **F6** / `记录异常` | 异常快照 | 无论 F7 开启与否均可随时按下，立即把当前局面、UI 覆盖层/按钮状态、检测到的候选动作、观测编码结果及最近 60 步历史与 Python 日志写入 `runs/live-bug-reports/*.json`；遇到卡死或报错停机时也会自动落盘一份 |

- 在输入框打字或按住 `Ctrl`/`Alt`/`Shift` 时自动忽略 F6/F7。
- 首次打开会异步加载 PyTorch 检查点（冷启动约 5–15 秒，HUD 显示加载进度），加载完成后单步推理通常在 5–20 ms。
- 默认 `device: "cpu"`，即使本机正在用 GPU 跑训练也不会抢占显存；无训练任务时可在 `pilot.cfg` 中改为 `"cuda"`。

## 打包与安装

```powershell
$py = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py scripts/package_live.py `
    --checkpoint runs/baseline-ad-necro-v1140-a10-s0/latest.pt `
    --device cpu `
    --out artifacts/NecroPilot-v0.2.0-v1140.zip
```

将 ZIP 内的 `NecroPilot/` 文件夹放到 Steam 游戏目录的 `mods/` 下即可。

## 热切换检查点（无需重编译 DLL）

直接修改 Steam `mods/NecroPilot/pilot.cfg` 的 `checkpoint` / `device` / `use_ema`，或运行：

```powershell
& $py scripts/update_live_weights.py runs/baseline-ad-necro-v1140-a10-s0/milestones/v001800.pt --device cpu
```

在游戏内关闭再重新打开 F7/F6 即可加载新权重。
