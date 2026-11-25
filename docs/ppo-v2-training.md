# 骨头人 PPO v2 · 零真人数据端到端从零训练（目标 A10）

`ppo_v2_policy.py` 与 `train_ppo_v2.py` 面向**不借助任何真人数据、不做模仿预训练、纯自博弈 RL 冲击标准 A10 连胜**的目标，系统性重构了上一代 `ppo_policy.py` + `train_aligned.py` 的架构、观测、回报信用分配与训练稳定性问题。

---

## 1. 核心改进总览（vs 旧版 `train_aligned.py`）

| 模块 | 旧版（`ppo_policy.py` + `train_aligned.py`） | 新版 PPO v2（`ppo_v2_policy.py` + `train_ppo_v2.py`） |
| --- | --- | --- |
| **骨干架构** | 18.1M 参数：前 5 层 + 嵌入共 15.26M **未预训练随机冻结**，仅训第 6 层（畸形 `ffn_width=5024`） | 轻量化 **100% 全参数端到端可训练** Pre-LN Transformer（融合 SDPA + GELU）：`base` 为 **6 层 / 宽度 256 / 4 头 / FFN 1024（7.27M 全可训）**，另提供 `compact`（4 层 2.9M）与 `wide`（6 层 8 头 7.27M） |
| **实体观测（`v2.4-deck-event-relic-shop`）** | Transformer 看不到遗物、药水、牌组摘要、地图前瞻，也看不到手牌关键词/附魔/状态/预览数值 | 全部玩家可见信息直接编码为独立实体 token：**已持有遗物（精确计数器 `counter_exact` + 描述词 + 变量）、药水、牌组构成摘要（基础 Strike/Defend 与诅咒数量/占比、体型分桶、12 类体系关键词占比）+ `token/deck_cards` 完整牌组卡牌清单计数、未来 4 层地图前瞻、Boss ID、手牌关键词/附魔/诅咒/数值与致死击杀减伤预览（`kill_attacker_dmg`）、事件/营火选项完整文案与后果提示（进战斗/加诅咒/满血回血）、商店 `on_sale`/稀有度/余钱比例与遗物/药水详情、跳牌膨胀信号与牌库已有张数**（严格隔离种子、抽牌顺序与隐藏内部状态） |
| **策略头与价值头** | 价值头仅 `Linear(256, 1)` 读取 `hidden[0]`，对牌组/遗物/药水/地图完全失明 | 注意力池化全局摘要 `[token_0, attn_pool(all_tokens)]`（512 维）→ **2 层 MLP 策略打分头** + **2 层 MLP 价值主干**，支持标量裁剪 Huber 或 **DreamerV3 风格 Symlog Two-Hot 分布式价值头**（`--value-bins 41`，交叉熵按 `/ ln(K)` 归一化且 `eff_vf_coef = vf_coef * 0.5`，自动 99.86% 兼容迁移旧权重 + `--critic-warmup 2` 价值头热身），并附带纯自博弈辅助头 `v_floor`（非终局中断自动屏蔽损失）与 `v_hp` |
| **信用分配（GAE）** | 每打一张牌都乘 `gamma=0.995, lam=0.97`（100 步后衰减至 2.8%，240 步几乎归零） | **房间边界折扣**：房间内 `gamma_step=0.9995, lam=1.0`，跨房间才乘 `gamma_room=0.99` 与 `lam_combat=0.95 / lam_strategic=0.98`，1 层构筑决策可保留 Boss 胜负 >40% 信用 |
| **奖励塑形** | 偏重眼前普通战保血（`0.65 × ΔHP`），Boss 战败全为 `-1` 缺乏进度梯度，无精英奖励 | 平衡普通战保血（`0.45 × ΔHP`）+ **分幕推进/Boss 加权与克制精英奖**（房间推进 Act1 `+0.01` / Act2 `+0.06` / Act3 `+0.10`；精英击杀 `+0.35/+0.25/+0.20`；Boss 伤害进度 `+0.40/+0.80/+1.00 × dmg_frac`）+ 过幕 `+1.5` + 通关 `+3.0` / 战败 `-1.0` |
| **PPO 稳定性、协议与吞吐** | `minibatch=24`（每轮 33 局更新 ~540 次）、恒定 LR/温度、战斗与战略混算优势、`--focus-rollouts 1` 使组内基线失效、无开发集选优、单样本持锁 FP32 推理 | 默认 `minibatch=128`、**动态微批推理线程（`InferenceBatcher`）+ CUDA BF16 混合精度（14w 实测提速 1.53×）**、**战斗/战略分离优势归一化**、`--init` 自动启用中低温续训调度（`0.55->0.30 / 0.65->0.40` 并持久化至 `protocol.json`）、支持 `--focus-cases 0` 纯新种子采样、阶段独立种子空间（`train_seed_namespace` + `seed_offset` 杜绝跨阶段重抽）、严格 `--resume` 一致性校验（模型/宿主/进度摘要/观测版本/开发集种子）、Value Clipping（`0.2`）、KL 早停（`0.025`）、**余弦学习率**、平滑综合 `dev_rank` 自动保存 **`best-checkpoint.pt`** 与终评去重 |
| **可选搜索引导（P2）** | 未接入 | 通过 `solver_bc.py` 支持可选 `--solver-demos` + `--bc-coef` 接入本地 `CombatSolver` 离线搜索示范（零真人数据），默认关闭（`--bc-coef 0`） |

---

## 2. 宿主可靠性修复（`engine_error: 3 -> 0`）

配合本轮升级同步修复了导致 `aligned-ppo-15x2-3x1` 评测出现 `engine_error: 3` 的两个无头宿主缺陷，已通过 `beta_build.py` 与 `beta_build.py --steam-parity` 重编译校验：

1. **`PunchOff.Nab()` 空引用崩溃**：在 [HeadlessEventPresentation.cs](../patches/HeadlessEventPresentation.cs) 中为 `PunchOff.Nab` 状态机跳过无画面宿主的 `NGame.Instance.ScreenShakeTrauma` 调用，完整保留添加 `Injury` 诅咒与获取随机遗物的游戏逻辑。
2. **`DoPlayCard` 回手牌误报**：在 [beta_build.py](../scripts/beta_build.py) 新增 `patch_play_card_check`，通过比对 `CombatManager.Instance.History.CardPlaysStarted` 计数，消除卡牌合法打出后返回手牌原位（如 `FeralPower` 下的 0 费攻击牌或单卡重抽回手）被误判为 `Card could not be played` 的问题。
3. **商店删牌重复触发导致 `step_limit` 死循环**：原 `RunSimulator.cs` 的 `ShopState` 仅上报 `removal?.Cost` 而未检查 `removal.IsStocked && !removal.Used`；模型学会高价值删牌后，在单次商店删完 1 张牌且剩余金币仍 ≥ 新价格时，会不断重复 `remove_card`（宿主空操作）直至 1,500 步 `step_limit`。已在 [beta_build.py](../scripts/beta_build.py)（`patch_shop_removal` 检查 `IsStocked && !Used` 并导出 `card_removal_stocked`）与 [beta_env.py](../scripts/beta_env.py)（按商店房间追踪已删牌状态与 `card_removal_stocked` 过滤）双层修复。
4. **`GodotObject.Call` / `CallDeferred` 桩签名缺失导致的 `combat_turn_timeout`**：`GodotStubs/Core.cs` 原先在 `Node` 上定义了返回 `void` 的 `CallDeferred`，而未在 `GodotObject` 上提供返回 `Variant` 的 `Call` 与 `CallDeferred`，导致 `RollingBoulderPower.AfterPlayerTurnStart` 等回调抛 `MissingMethodException` 卡死回合。已在 [beta_build.py](../scripts/beta_build.py) 修正并重编译校验，原诊断包 `err-22949d2669e2.json` 与 `P4BQS35EE3F7` 均已验证清零。
5. **全游戏 73 个事件静态扫描与 Act 3 `The Trial` 空引用根治**：静态扫描 `sts2.dll` 全部 73 个 `EventModel` 及其异步状态机后确认，除已由 `HeadlessSteamRun.cs` 与 `HeadlessEventPresentation.cs` 覆盖的 `DenseVegetation` / `PunchOff` / `Amalgamator` / `WaterfallOfObviousSecrets` 外，Act 3 事件 `Trial`（`Accept` / `AddVfxAnchoredToPortrait` / `DoubleDown`）是全游戏最后一个未判空 UI 的事件（终局事件 `TheArchitect` 原生已全判空）。已在 [HeadlessEventPresentation.cs](../patches/HeadlessEventPresentation.cs) 补齐并在 [beta_env.py](../scripts/beta_env.py) 过滤放弃弹窗选项 `DOUBLE_DOWN`。
6. **通用商店空操作防护（根治 `Sozu` 买药水 1,291 步死循环）**：在 128 种子盲测中发现持有 `Sozu`（禁药水遗物）时反复点击 `buy_potion` 会触发宿主静默空操作直至 `step_limit`。在 [beta_env.py](../scripts/beta_env.py) 中新增 `_shop_snapshot` 通用商店状态快照比对与 `shop_blocked_actions` 屏蔽机制：任意商店非离开动作若未改变金币/牌组/遗物/药水/商品存货，立即在本商店房间屏蔽该动作，一劳永逸封堵同类遗物交互空转。

修复后 S5 全程 **8,640 局训练 + 128 局留出终评均实现 0 任何异常**（`diagnostics/` 为空，100% 正常结算）。

---

## 3. 模型预设与参数规模

| 预设 (`--preset`) | 层数 | 宽度 | 注意力头 | FFN 宽度 | 词表桶数 | 总参数（100% 可训练） | 适用场景 |
| --- | :---: | :---: | :---: | :---: | :---: | ---: | --- |
| `compact` | 4 | 192 | 4 | 768 | 4,096 | ~2.9M | 快速实验、CPU/小显存验证 |
| `base`（默认） | 6 | 256 | 4 | 1,024 | 8,192 | **7,265,284** | 主力标准训练（与 6 层架构对齐且轻量化） |
| `wide` | 6 | 256 | 8 | 1,024 | 8,192 | **7,265,284** | 8 头注意力变体对照 |

---

## 4. 启动与阶梯训练命令（A0 → A5 → A10）

在仓库根目录 PowerShell 运行（确保已按 `beta_build.py --steam-parity` 构建最新宿主）：

```powershell
$py = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = (Resolve-Path 'artifacts/ppo-cuda').Path

# 阶段 1：A0 纯从零训练（64 轮，每轮 12×2 新局 + 3×3 难战 = 33 局，每 4 轮在 24 个固定开发种子选优）
& $py -u scripts/train_ppo_v2.py `
    --progress runs/aligned-ppo-v1-input/progress.save `
    --out runs/ppo-v2-a0 `
    --preset base --ascension 0 `
    --rounds 64 --fresh-seeds 12 --rollouts-per-seed 2 `
    --focus-cases 3 --focus-rollouts 3 `
    --dev-games 24 --eval-every 4 --eval-games 64 `
    --minibatch 128 --epochs 3 --lr 3e-4 `
    --workers 4 --device cuda

# 中断后原目录续训（--resume 自动恢复模型、优化器、难例池与最佳记录）
& $py -u scripts/train_ppo_v2.py `
    --progress runs/aligned-ppo-v1-input/progress.save `
    --out runs/ppo-v2-a0 `
    --preset base --ascension 0 `
    --rounds 128 --workers 4 --device cuda --resume
```

### 进阶迁移到 A5 / A10（或开启自动升阶）

```powershell
# 从 A0 最佳检查点迁移至 A5
& $py -u scripts/train_ppo_v2.py `
    --progress runs/aligned-ppo-v1-input/progress.save `
    --init runs/ppo-v2-a0/best-checkpoint.pt `
    --out runs/ppo-v2-a5 `
    --preset base --ascension 5 --rounds 64 `
    --fresh-seeds 16 --rollouts-per-seed 2 --focus-cases 4 --focus-rollouts 3 `
    --minibatch 256 --lr 1.5e-4 --dev-games 24 --eval-every 4 `
    --workers 4 --device cuda

# 从 A5 迁移至 A10（也可加 --asc-max 10 --asc-promote-rate 0.5 让脚本按开发胜率自动升阶）
& $py -u scripts/train_ppo_v2.py `
    --progress runs/aligned-ppo-v1-input/progress.save `
    --init runs/ppo-v2-a5/best-checkpoint.pt `
    --out runs/ppo-v2-a10 `
    --preset base --ascension 10 --rounds 128 `
    --fresh-seeds 16 --rollouts-per-seed 2 --focus-cases 4 --focus-rollouts 3 `
    --minibatch 256 --lr 1e-4 --dev-games 32 --eval-every 4 --eval-games 128 `
    --workers 4 --device cuda
```

### 可选 P2：接入 `CombatSolver` 离线搜索引导（零真人数据）

若要利用本地 [solve_bounded.py](../scripts/solve_bounded.py) 离线求解器为精英/Boss 难战提供正则示范，可将录制好的 `{state, actions, chosen}` JSONL 示范目录传入 `--solver-demos` 并设置 `--bc-coef`（建议 `0.1–0.2`，前期加速收敛、后期关闭或降至 `0`）：

```powershell
& $py -u scripts/train_ppo_v2.py `
    --progress runs/aligned-ppo-v1-input/progress.save `
    --out runs/ppo-v2-a0-solver `
    --solver-demos runs/solver-demos --bc-coef 0.15 `
    --preset base --ascension 0 --rounds 64 --workers 4 --device cuda
```

---

## 5. 输出产物说明

- `protocol.json`：完整模型配置、观测编码版本（`observation_version`）、参数量、宿主指纹、阶段训练种子命名空间与轮次偏移（`train_seed_namespace` / `seed_offset`，`--init` 自动与历史阶段隔离）、开发集与留出测试种子。
- `round-0000.json`：训练第 1 轮前自动跑出的 **R0 贪心开发集基线**（并同步写入 `protocol.json` 的 `r0_baseline`，方便追踪首幕是否回撤；传 `--skip-r0-eval` 可关闭）。
- `round-NNNN.json`：每轮学习率/温度/熵调度、训练统计、PPO 更新指标（`actor_loss` / `value_loss` / `aux_loss` / `bc_loss` / `mean_kl` / `entropy_combat` / `entropy_strategic`）、进 Act 2 快照（`act2_entry`）及开发集评测摘要。
- `best-checkpoint.pt` 与 `top-k/`：按平滑综合分（`24*win_rate + 10*act2_rate + 5*act1_rate + 1*boss_rate + 0.12*mean_progress`）自动保存最优权重及 **Top-K 候选权重**（默认 `--top-k 3`，仅存模型权重、自动清理淘汰文件；要求全部开发局正常终局结算）。
- `ppo.pt`：最新轮次的完整可续训检查点（含优化器、难例池与 Top-K 列表）。
- `diagnostics/err-<sig>.json`：任意对局发生非终局异常时自动按签名去重落盘的离线复现包。
- `evaluation.json`：训练结束后在独立留出种子上对 `best-checkpoint.pt`、最终轮次及全部 `top-k/` 候选权重的贪心对比评测。

---

## 6. 遇到意外中断（`engine_error` / `step_limit` / `timeout` 等）的自动保护与一键复现

- **全异常自动去重落盘**：训练/开发/评测过程中只要遇到任何非终局意外（`engine_error`、`step_limit`、`timeout`、`unsupported_decision`、`policy_error`），立即在 `<out>/diagnostics/` 按签名去重写入复现包（`err-<sig>.json`、`step_limit-<sig>.json`、`timeout-<sig>.json` 等，附带动作计数与最后 30 步 `recent_actions`）。
- **异常率熔断（`--max-error-rate`，默认 `0.25`）**：若单轮所有非终局异常总占比超过阈值，脚本会先完整保存当前 `round-NNNN.json` 和 `ppo.pt`，打印按异常类型与签名聚合的统计后立即中止（退出码 2），避免高频新 bug 或死循环导致挂机空转（设 `0` 可关闭）。
- **1 秒离线复现（无需加载任何模型权重）**：

```powershell
& $py -u scripts/train_ppo_v2.py --replay-diagnostic runs/ppo-v2-a0/diagnostics/err-<sig>.json
```

打完宿主补丁重编译后重跑同一条命令，输出 `RESOLVED` 即确认修复通过。

---

## 7. 跨检查点严格盲测与迭代改动日志（S1 → S5）

### 多检查点严格未见种子盲测脚本

使用 [eval_compare.py](../scripts/eval_compare.py) 自动收集历史全部 `runs/` 中的训练、开发与测试种子并严格排除，在同一批全新未见种子上横向比较多个检查点（拆分报告 **到首幕 Boss / 首幕 Boss 击杀率 / 过首幕 / 进第二幕 Boss / 第二幕普通战掉血中位数 / 过第二幕 / 通关**）：

```powershell
& $py -u scripts/eval_compare.py `
    --progress runs/aligned-ppo-v1-input/progress.save `
    --out runs/benchmark-a0-128.json `
    --games 128 --workers 14 --device cuda `
    S3-R48=runs/ppo-v2-a0-s3/best-checkpoint.pt `
    S4-R160=runs/ppo-v2-a0-s4/ppo.pt `
    S5-R16=runs/ppo-v2-a0-s5/best-checkpoint.pt `
    S5-R180=runs/ppo-v2-a0-s5/ppo.pt
```

### 阶段迭代日志与核心里程碑（全部 A0 · Necrobinder · 零真人数据）

> 注意：不同阶段 `evaluation.json` 因 `dev_games` 变化（如 S4 为 32、S5 为 64）导致留出 64 种子并非完全相同（S4 与 S5 仅重合 50 局）；严格跨阶段对比请以同一固定盲测面板（`eval_compare.py`）为准。

| 阶段 / 目录 | 核心代码与机制改动 | 训练规模 & 关键信号 | 各自 64 种子留出终评 / 128 种子盲测 |
| --- | --- | --- | --- |
| **S1** (`runs/ppo-v2-a0`) | PPO v2 初始版本：7.27M 全可训 Transformer、标量价值头、房间边界 GAE、基础遗物/药水/牌组/地图 token、修复 `PunchOff` 与 `DoPlayCard` | 64 轮（从零起步） | `best` R64: Act1 过幕 `8/64 (12.5%)`，1 次 `engine_error` |
| **S2** (`runs/ppo-v2-a0-s2`) | 修复商店删牌死循环与 `GodotObject.Call/CallDeferred` 桩；加入 `InferenceBatcher` + CUDA BF16 提速 1.53×；扩展完整牌组/事件/遗物/商店特征 | 160 轮续训（标量价值头），宿主 0 崩溃 | `final` R160: 终评 Act1 过幕 `22/64 (34.4%)`；128 盲测 `32/128 (25.0%)`、进 Act2 Boss `3/32` |
| **S3** (`runs/ppo-v2-a0-s3`) | 引入 `v2.4-deck-event-relic-shop` 全量玩家可见特征（仍为标量价值头，`best` 在 R48、`ppo.pt` 跑至 R54） | 54 轮；128 盲测中展现出最强 Act2 生存力（进 Act2 Boss `13/33 = 39.4%`、首次评测过 Act2 抵达 `A3F15`） | 128 盲测（`S3-R48`）: Act1 过幕 `33/128 (25.8%)`，进 Act2 Boss `13/33 (39.4%)`，过 Act2 `1/128` |
| **S4** (`runs/ppo-v2-a0-s4`) | **首次引入 41-bin Two-Hot 分布式价值头**（CE `/ ln(K)` 归一化 + `eff_vf_coef=0.25`）、`--init` 自动中低温调度、`--focus-cases 0` 纯新种子；全 73 事件静态扫描并修复 Act3 `The Trial` | 160 轮（7,680 局）：训练出现 6 次过 Act2、134 次见 Act2 Boss；发现 32 种子 dev 选优反序与 `Sozu` 空转 | 128 盲测（`S4-R160`）：Act1 Boss 击杀 `46/93 (49.5%)`、过 Act1 `46/128 (35.9%)`，但进 Act2 Boss 仅 `6/46 (13.0%)` |
| **S5** (`runs/ppo-v2-a0-s5`) | **三项训练协议加固**（独立阶段种子命名空间、非终局屏蔽 `v_floor`、严格 `--resume` 校验）+ **平滑 `dev_rank` + 64 种子 dev** + **全异常诊断落盘与熔断** + **分幕推进/Boss 塑形（克制 Act2/3 精英奖）** + **`Sozu` 通用商店防空转** | 180 轮（8,640 局）：**0 异常**；**历史首次整局通关**（`R159` 种子 `1LDA8S3ES3J1` `victory` @ `A3F15`）；训练 **19 次过 Act2**、**387 次见 Act2 Boss** | **128 锁定盲测（[benchmark-a0-128-s5.json](../runs/benchmark-a0-128-s5.json)，0 异常）**：`S5-R16` 过 Act1 **`59/128 (46.1%)`**、进 A2 Boss **`18/59 (30.5%)`**、A2 Boss 均伤 `39.2%`、**`mProg=21.01`**；`S5-R180` 过 Act1 `49/128 (38.3%)`、进 A2 Boss `15/49 (30.6%)`、**A2 Boss 均伤 `61.0%`、过 Act2 `2/128`**、`mProg=20.10` |
| **S6** (`runs/ppo-v2-a0-s6-r180`) | 从 `S5-R180` 起步，**降学习率（`4e-5`）+ 收紧 KL（`0.015`）**；新增 **R0 贪心基线**、`--top-k 3` 自动保留、`--focus-acts` 幕数筛选、进 Act 2 构筑快照 `act2_entry` | 111 轮（开发集 0 中段回撤，`R88` 达 `45/64 = 70.3%` 过 A1、`mProg=25.69`） | **128 锁定盲测（[benchmark-a0-128-s6.json](../runs/benchmark-a0-128-s6.json)，0 异常）**：**`S6-R88`（best）** 到 A1 Boss **`116/128 (90.6%)`**、过 Act1 **`70/128 (54.7%)`**、进 A2 Boss **`29/70 (41.4%)`**、过 Act2 **`3/128`**、**`mProg=23.57`**；**`S6-R111`** 过 Act1 `66/128 (51.6%)`、进 A2 Boss `28/66 (42.4%)`、**A2 Boss 均伤 `61.9%`、过 Act2 `4/128`** |

> 完整论文级实验台账、决策日志与高吞吐原生栈对比见 [docs/experiment-ledger.md](experiment-ledger.md)。


> 注：当前 v2 相关脚本、测试与文档已全部落盘（28/28 单元测试通过），如需在 Git 历史中固定此里程碑快照，可执行一次 `git commit`。
