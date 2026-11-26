# 杀戮尖塔 2 · 骨人零搜索策略

本项目面向《杀戮尖塔 2》（Slay the Spire 2）**标准模式 · 死灵契约师（Necrobinder / 骨人）**，目标是**零真人数据、严格 No-SL、纯从零强化学习**攻克标准 **Ascension 10（A10）**。每次决策读取全部玩家可见局面与候选动作，一次前向推理直接出牌或做局外构筑选择，并提供同名实机 AI Mod（`NecroPilot v0.2.1`）可在 Steam 正版游戏中实时接管或诊断。

## 当前主线进展（`v0.114.0` · A10 · 原生 Windows 高吞吐底座）

当前主线基于 MIT 开源常驻 `.NET 9` 无头引擎栈（[AlphaDefect-Public](https://github.com/hmty-CHEN/AlphaDefect-Public) `game-v0.114.0`，配套本仓库 Windows 异步与里程碑补丁 [patches/alphadefect-v1140-win.patch](patches/alphadefect-v1140-win.patch)），在单张 **RTX 5070 Ti 移动版（12 GB）** 上达到 **`945–1,032 dec/s`** 稳态吞吐。

### 1. 已完成：1 亿步 A10 锚点基线（`v3050`）
- **检查点下载**：[GitHub Release · `v3050/latest.pt`](https://github.com/LukeLu1650/sls2bot/releases/download/v3050/latest.pt)（`279.6 MB`，含完整 `state / ema / Adam / Muon` 状态，支持直接续训或接入实机 Mod）
- **128 锁定未见 A10 种子评测（`A10EVAL0000..0127` · 贪心 · 0 卡局）**：

| 检查点 | 累计步数 | 权重 | 均层 | 进 Act 2 | 进 Act 3 | A1 Boss (`杀/到`) | A2 Boss (`杀/到`) | A3 Boss 1 (`杀/到`) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `v300` | `9.8M` | EMA | `15.12` | `13/128 (10.2%)` | `0/128 (0.0%)` | `13/73 (17.8%)` | `0/1 (0.0%)` | `0/0` |
| `v1200` | `39.4M` | EMA | `23.01` | `64/128 (50.0%)` | `3/128 (2.3%)` | `64/123 (52.0%)` | `3/24 (12.5%)` | `0/1 (0.0%)` |
| `v1500` | `49.2M` | EMA | `26.49` | `85/128 (66.4%)` | `7/128 (5.5%)` | `85/125 (68.0%)` | `7/47 (14.9%)` | `0/3 (0.0%)` |
| `v2400` | `78.7M` | EMA | `28.78` | `110/128 (85.9%)` | `12/128 (9.4%)` | `110/126 (87.3%)` | `12/54 (22.2%)` | `1/3 (33.3%)` |
| `v3000` | `98.4M` | EMA | `30.13` | `114/128 (89.1%)` | `18/128 (14.1%)` | `114/126 (90.5%)` | `18/62 (29.0%)` | `0/6 (0.0%)` |
| **`v3050`** | **`100.0M`** | **EMA** | **`31.00`** | **`112/128 (87.5%)`** | **`25/128 (19.5%)`** | **`112/125 (89.6%)`** | **`25/67 (37.3%)`** | `0/13 (0.0%)` |
| `v3050` (diag) | `100.0M` | raw | `28.02` | `99/128 (77.3%)` | `15/128 (11.7%)` | `99/122 (81.1%)` | `15/50 (30.0%)` | `2/8 (25.0%)` |

### 2. 进行中与下一阶段实验
- **方案 C（`v3050` 后期整局占比分叉，`100M → 130M`）**：直接从 `v3050` 分叉 `+3,000 万步`，对比 `run_fraction = 0.50`（主机 5070 Ti，`2×20` 路 = 精确 `50.0%`）与 `run_fraction = 0.75`（协作机 4070，`2×12` 路 = 精确 `75.0%`），配合 [eval_handover_probe.py](scripts/eval_handover_probe.py) 冻结战斗策略接手探针解耦构筑准备与战斗执行。
- **Step 1（后期求解器错题专项）**：已通过 [verify_boss_solver.py](scripts/verify_boss_solver.py) 在独立 `A10MINE` 种子上完成小规模验证（同初态固定种子下，策略败局中搜出并经干净引擎 100% 复验通过 `7/29 = 24.1%`，单核产率 `71.6` 题/小时，`verify_failed = 0`），将在方案 C 评完后无缝接入。

---

## 快速复现（协作训练 / 评测 / 实机 Mod）

```powershell
# 1. 克隆本仓库并一键恢复固定版本的 AlphaDefect v0.114.0 源码、Windows 补丁与 4070 配置
git clone https://github.com/LukeLu1650/sls2bot.git D:\sts2-bot
cd D:\sts2-bot
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_4070.ps1
```

- **参与 4070 协作训练**：下载 [v3050/latest.pt](https://github.com/LukeLu1650/sls2bot/releases/download/v3050/latest.pt)，按 [docs/4070-guide.md](docs/4070-guide.md) 放好断点并编译本地正版 `v0.114.0` 无头引擎即可开跑（`+3,000 万步` 约 `12–16 小时`）。
- **安装实机 AI Mod（`NecroPilot v0.2.1`）**：详见 [docs/live-pilot.md](docs/live-pilot.md)，支持游戏内 **F7 自动接管**、**F6 一键记录异常快照**，并可通过 [scripts/update_live_weights.py](scripts/update_live_weights.py) 免重编热切换 `v3050` 等检查点。
- **统一评测与诊断工具**：
  - [scripts/eval_v1140.py](scripts/eval_v1140.py)：锁定 A10 种子批量 `EMA / raw` 评测（内置防 `A10EVAL` 种子泄漏保护）
  - [scripts/eval_handover_probe.py](scripts/eval_handover_probe.py)：冻结战斗策略 Boss 接手探针
  - [scripts/verify_boss_solver.py](scripts/verify_boss_solver.py)：同初态固定战斗种子 Boss 错题搜寻与干净引擎复验器

---

## 文档导航

### 主线文档（`v0.114.0` · 当前推荐）
| 入口 | 内容 |
| --- | --- |
| [RTX 4070 协作训练指南](docs/4070-guide.md) | 零重练从 `v3050` 分叉续训 `rf=0.75` 的完整环境搭建、编译自检、监控与评测步骤 |
| [实机接管与诊断 Mod（NecroPilot v0.2.1）](docs/live-pilot.md) | 面向 Steam `v0.114.0`，复用原生观测编码器与 `.pt` 检查点（F7 接管 / F6 异常快照 / 权重热切换） |
| [实验台账与工程决策日志](docs/experiment-ledger.md) | 按时间顺序记录的改动、版本指纹、吞吐基准与 128 种子评测汇总 |

### 历史阶段与归档文档（`v0.113.1` PPO v2 / 早期探索）
| 入口 | 内容 |
| --- | --- |
| [PPO v2 端到端训练（v0.113.1 历史主线）](docs/ppo-v2-training.md) | 7.27M 轻量化全可训 6 层 Transformer、房间边界 GAE 与 S1–S6 阶段训练记录 |
| [Windows 骨头人基线](docs/windows-baseline.md) | 早期 `v0.113.1` 本机构建、复测命令与实测结果 |
| [当前模型与可选动作](docs/current-policy.md) | 标准模式动作空间与观测格式说明 |
| [骨头人训练设计](docs/necrobinder.md) | 角色机制、玩家可见特征与课程设计 |
| [标准 A0 战斗训练](docs/combat-training.md) | 早期联合训练、难战课程与有界搜索器 |
| [自主训练路线与构筑](docs/strategy-training.md) | 早期构筑对照实验与续训脚本 |
| [游戏版本迁移](docs/versions-and-modes.md) | 标准模式构建验证与版本迁移资料 |
| [交接说明](docs/sol-handoff.md) | 早期独立开发与验证任务记录 |
| [旧版摄政王实验（v0.107.1）](docs/regent.md) | 早期 Linux `v0.107.1` 历史复现资料 |

---

## 目录说明与致谢

本仓库仅跟踪项目源码、Windows 补丁、评测/Mod 脚本与文档；上游检出、正版游戏程序集与训练检查点分别位于 `.gitignore` 忽略的 `external/`、`artifacts/` 与 `runs/` 目录，均可通过 `scripts/setup_4070.ps1` 与本地正版 Steam 安装复现。

致谢开源社区与前期参考工作：[AlphaDefect-Public（MIT）](https://github.com/hmty-CHEN/AlphaDefect-Public) · [原视频](https://www.bilibili.com/video/BV19Bxq6wE9k/) · [论文（v0.107.1）](https://arxiv.org/abs/2611.12088) · [v0.113.1 宿主适配](https://github.com/wuhao21/sts2-cli/pull/107)
