# 实验台账与工程决策日志（Experiment Ledger & Decision Log）

> 本文档按时间顺序完整记录本项目的**代码改动、宿主修复、超参数决策、评测口径、硬件吞吐实测与阶段成绩**，用于论文复现、消融归因与版本溯源。此后**每一次改动与实验都必须在此追加一条带日期、代码范围、动机和实测结果的记录**。

---

## 一、固定环境与版本指纹（Provenance）

| 项目 | 固定值 | 备注 |
| --- | --- | --- |
| **目标游戏与角色** | Slay the Spire 2 · Standard · **Necrobinder（骨头人）** | 零真人数据、零模仿预训练、严格 No-SL |
| **主基线游戏版本** | **`v0.113.1`**（commit `d09d48e5`） | `sts2.dll` SHA-256: `62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3`（与 arXiv:2611.31076 附录 D 逐字节一致） |
| **已核对最新公测版** | `v0.114.0`（2026-11-25，commit `a7b9a9c0`） | 反编译核对：骨头人专属牌/机制 0 改动；仅 4 项全角色通用微调（`HardToKill`、`Dampen`、`MyShinies`、预览口径） |
| **本机主力硬件** | AMD Ryzen 9 8940HX（16C/32T）+ NVIDIA RTX 5070 Ti Laptop GPU（12 GB VRAM，BF16） | Windows 11 原生环境，.NET SDK `9.0.312` |
| **备用协作硬件** | RTX 4070 Laptop（8 GB VRAM）+ 32 GB RAM（Windows 游戏本） | 暂未启动，待本机定稿研究主线后分配受控对照任务 |
| **PPO v2 固定盲测面板** | `runs/benchmark-a0-128.json`（128 个 A0 种子） | 自动排除 S1–S5 历史全部训练/开发/测试种子，与 S5/S6 训练种子零重合 |
| **Steam 进度快照** | `runs/aligned-ppo-v1-input/progress.save` | SHA-256: `e49f66aa900ccccc0fcfe74776d10333ed430b3f4034c05eac53a717ebf61677` |

---

## 二、决策与代码改动日志（Chronological Decision Log）

| 编号 | 阶段 | 改动模块 | 改动内容与动机 | 验证结果 |
| :---: | --- | --- | --- | --- |
| **DL-01** | S1 | `scripts/ppo_v2_policy.py`, `scripts/train_ppo_v2.py` | 重构 PPO v2：7.27M 100% 全可训 Pre-LN Transformer（`6×256×4×1024`）、实体观测 token、房间边界 GAE（`gamma_step=0.9995, gamma_room=0.99`）、战斗/战略分离优势归一化、余弦学习率、KL 早停 | S1（64 轮）64 种子终评过 A1 `8/64 (12.5%)`，1 次 `engine_error` |
| **DL-02** | S1–S2 | `patches/HeadlessEventPresentation.cs`, `scripts/beta_build.py`, `scripts/beta_env.py` | 宿主可靠性修复 #1–#4：① `PunchOff.Nab` 跳过无头震屏；② `DoPlayCard` 用 `CardPlaysStarted` 修复回手牌误报；③ `patch_shop_removal` + 房间级跟踪修复商店删牌 `step_limit` 死循环；④ `GodotStubs/Core.cs` 补齐 `Variant GodotObject.Call/CallDeferred` 修复 `RollingBoulderPower` 回合超时 | 原 64 种子评测 `engine_error` 由 `3 → 0` |
| **DL-03** | S2 | `scripts/ppo_v2_policy.py`, `scripts/train_ppo_v2.py` | 推理提速：新增无锁动态微批推理线程 `InferenceBatcher` + CUDA BF16 混合精度 + 终评去重 | 14 workers 端到端实测提速 **1.53×** |
| **DL-04** | S3 | `scripts/ppo_v2_policy.py` | 升级至 `OBSERVATION_VERSION = "v2.4-deck-event-relic-shop"`（全量玩家可见信息：牌组构成/12 体系占比/完整卡牌清单、事件选项文案、致死击杀减伤预览 `kill_attacker_dmg`、遗物精确计数器 `counter_exact` + 描述词、商店打折/稀有度/余钱比例） | 128 锁定盲测 `S3-R48` 进 A2 Boss 达 **`12/34 (35.3%)`**、首次评测过 A2 达 `A3F15` |
| **DL-05** | S4 | `scripts/ppo_v2_policy.py`, `scripts/train_ppo_v2.py`, `patches/HeadlessEventPresentation.cs` | ① 引入 DreamerV3 Symlog Two-Hot 41-bin 分布式价值头（`[-12.46, +12.46]`，CE 除以 `ln(K)` 归一化 + `eff_vf_coef=0.5*vf_coef` + 99.86% 权重兼容迁移 + 2 轮 critic warmup）；② `--init` 自动启用中低温续训调度（`0.55->0.30 / 0.65->0.40`）并支持 `--focus-cases 0`；③ 静态扫描全游戏 73 个 `EventModel` 并修复最后一个未判空事件 Act 3 `The Trial` | `S4-R160` 128 盲测过 A1 升至 `41/128 (32.0%)`；训练期出现 6 次过 A2；发现 `Sozu` 买药水死循环 |
| **DL-06** | S5 | `scripts/beta_env.py`, `scripts/train_ppo_v2.py`, `scripts/ppo_v2_policy.py` | ① 通用商店空操作防护（`_shop_snapshot` 根治 `Sozu` `buy_potion` 1,291 步死循环）；② 三项协议加固（阶段独立训练种子命名空间、非终局屏蔽 `v_floor` 损失、严格 `--resume` 校验）；③ 64 种子 dev + 平滑 `dev_rank`（要求 100% 正常结算）；④ 全非终局异常去重诊断落盘 + `--max-error-rate` 熔断；⑤ 分幕推进/Boss 加权（Act1/2/3 推进 `+0.01/+0.06/+0.10`，克制精英 `+0.35/+0.25/+0.20`，Boss 伤害 `+0.40/+0.80/+1.00`） | S5（180 轮，8,640 局）全程 **0 异常**；**历史首次整局通关**（`R159 1LDA8S3ES3J1` `A3F15 victory`）；128 盲测 `S5-R16` 过 A1 `59/128 (46.1%)`、`S5-R180` A2 Boss 均伤 `61.0%`、过 A2 `2/128` |
| **DL-07** | S5→S6 | `scripts/eval_compare.py`, `scripts/train_ppo_v2.py`, `scripts/train_curriculum.py`, `tests/test_ppo_v2.py` | ① `eval_compare.py` 支持 `--seeds-from` 锁定面板、逐模型温度后缀 `@c,s`、Act 2 普通/精英/Boss 死亡分项、进 Act 2 构筑快照 `act2_entry`；② `train_ppo_v2.py` 增加 **R0 贪心基线**（`round-0000.json`）、**Top-K 候选权重自动保留与终评复测**（默认 `--top-k 3`）、**`--focus-acts` 幕数筛选开关**（支持收集 Act 2/3 普/精/Boss 战）；③ 补齐 29/29 单元测试 | 29/29 单测全过；S6 从 `S5-R180` 以 `lr=4e-5 / target_kl=0.015` 温和续训至 R111，128 盲测 **`S6-R88` 过 A1 `70/128 (54.7%)`、进 A2 Boss `29/70 (41.4%)`、过 A2 `3/128`、`mProg=23.57`**，**`S6-R111` 过 A2 `4/128`、A2 Boss 均伤 `61.9%`** |
| **DL-08** | 吞吐线 | `external/AlphaDefect-Public/`（`v0.113.1` 与 `game-v0.114.0/`） | 引入开源高吞吐参考栈（arXiv:2611.31076，MIT）并完成原生 Windows 适配：① `env.py` 自动定位 `dotnet` + Windows 每槽位守护读线程异步 `VecEnv.recv_ready()`；② `actor.py` 关闭 Windows `mmap` + `os.nice()` 保护；③ `evaluator.py` `os.nice()` 保护；④ `train.py` `publish()` 覆盖重试；⑤ `resources.py` Windows GPU 采样兼容；⑥ `VirtualFs.cs` 支持 `STS2_USER_DIR`；⑦ 新增三档测速工具 `tools/win_bench.py` | 原生 Windows 0 异常跑通全三档：纯采集 **`2,616 dec/s`**（18.4× S6）、完整多 Actor PPO **`~940 dec/s`**（6.6× S6，峰值显存 `3.95 GB`）；定位到 Windows WDDM 下 `actors=2` 显著优于 `actors=4`（避免 5 进程 CUDA 调度竞争） |

---

## 三、统一 128 锁定未见种子横向对比表（`runs/benchmark-a0-128.json` · A0 · 骨头人 · 修复后同宿主）

> 全部检查点在**同一批 128 个从未参与训练/开发的 A0 种子**与同一修复后宿主上贪心评测（0 任何异常）。数据文件：[benchmark-a0-128-s5.json](../runs/benchmark-a0-128-s5.json)、[benchmark-a0-128-s6.json](../runs/benchmark-a0-128-s6.json)。

| 检查点 | 价值头 | 采样 | 到 A1 Boss | A1 Boss 击杀率 | 过 Act 1 | A2 死亡 (`普/精/B`) | 进 A2 Boss（占过 A1） | A2 Boss 均伤 | 过 Act 2（进 A3 Boss） | `mProg` | 进 A2 牌组均张 / 升级中位 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `S3-R48` | Scalar | greedy | 96/128 | `35.4%` | `34 (26.6%)` | `11 / 11 / 11` | `12/34 (35.3%)` | `45.3%` | `1 (1)` | `18.82` | `19.85 / 2.0` |
| `S4-R160` | Two-Hot 41 | greedy | 90/128 | `45.6%` | `41 (32.0%)` | `23 / 11 / 7` | `7/41 (17.1%)` | `39.8%` | `0 (0)` | `18.32` | `21.10 / 1.0` |
| `S5-R16` | Two-Hot 41 | greedy | 105/128 | `56.2%` | `59 (46.1%)` | `27 / 14 / 18` | `18/59 (30.5%)` | `39.2%` | `0 (0)` | `21.01` | `20.64 / 1.0` |
| `S5-R180` | Two-Hot 41 | greedy | 97/128 | `50.5%` | `49 (38.3%)` | `20 / 14 / 13` | `15/49 (30.6%)` | `61.0%` | `2 (1)` | `20.10` | `20.73 / 1.0` |
| `S5-R16@lowT` | Two-Hot 41 | `0.32/0.42` | 97/128 | `45.4%` | `44 (34.4%)` | `22 / 11 / 10` | `11/44 (25.0%)` | `55.6%` | `1 (1)` | `19.38` | `20.70 / 1.0` |
| `S5-R180@lowT` | Two-Hot 41 | `0.32/0.42` | 93/128 | `44.1%` | `41 (32.0%)` | `24 / 6 / 10` (+1 他) | `10/41 (24.4%)` | `60.2%` | `0 (0)` | `18.92` | `20.71 / 1.0` |
| `S6-R64` | Two-Hot 41 | greedy | 110/128 | **`61.8%`** | `68 (53.1%)` | `24 / 17 / 27` | `27/68 (39.7%)` | `52.7%` | `0 (0)` | `22.77` | `20.84 / 1.5` |
| **`S6-R88`** | Two-Hot 41 | greedy | **116/128** | `60.3%` | **`70 (54.7%)`** | `20 / 21 / 26` | **`29/70 (41.4%)`** | `57.5%` | `3 (1)` | **`23.57`** | `20.70 / 1.5` |
| **`S6-R111`** | Two-Hot 41 | greedy | **116/128** | `56.9%` | `66 (51.6%)` | `23 / 15 / 24` | **`28/66 (42.4%)`** | **`61.9%`** | **`4 (0)`** | `23.18` | `20.73 / 2.0` |

---

## 四、本机原生 Windows 吞吐基准记录（2026-11-25，RTX 5070 Ti 移动版 + Ryzen 9 8940HX）

| 系统 / 档位 | 配置 | 稳态吞吐 (`dec/s`) | 显存峰值 / 温度 | 关键耗时与诊断 |
| --- | --- | :---: | :---: | --- |
| **PPO v2（S6 实测）** | 12–14 同步 worker + JSON stdio + 逐局重起宿主 | `120–142` | ~2 GB | 1 亿步需 ~8.6–9.6 天 |
| **AlphaDefect · Tier 1A** | 1 进程 × 16 路常驻引擎（二进制帧，随机动作） | **`2,615.9`** | — | `wait=41.3%`，20s 完成 750 局、3,361 快照，0 错 |
| **AlphaDefect · Tier 1B** | 1 进程 × 32 路常驻引擎 | `2,266.7` | — | 单进程 GIL 饱和，`wait=15.7%` |
| **AlphaDefect · Tier 2A** | 1 Actor × 16 路 + 19.4M BF16 CUDA 推理 | `391.0` | — | `avg_batch=7.18`，`fwd=84.1%` |
| **AlphaDefect · Tier 2B** | 1 Actor × 32 路 + 19.4M BF16 CUDA 推理 | `667.9` | — | `avg_batch=14.47`，`fwd=74.9%`，`wait=0.9%` |
| **AlphaDefect · Tier 3A** | **2 Actors × 16 路 = 32 路**，完整 PPO（`batch=16k, mb=1024×2`） | **`939.3`**（`948/939/931`） | **`3.95 GB` / `68°C`** | `t_collect=12.3s`, `t_learn=4.78s`, `dropped=0, lag=0.23, stuck=0`；每批含 ~246 场 Boss 专项练习 |
| **AlphaDefect · Tier 3B** | 4 Actors × 16 路 = 64 路，完整 PPO（`batch=32k, mb=1024×4`） | `509–732` | `5.20 GB` / `72°C` | Windows WDDM 下 5 进程 CUDA 竞争导致 `t_learn` 暴涨至 `22.7–26.7s` |
