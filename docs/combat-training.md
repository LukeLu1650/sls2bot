# 标准 A0 骨头人战斗训练

> 新版纯从零端到端训练（全参数可训练轻量化 6 层 Transformer、补齐遗物/药水/牌组/地图观测、房间边界信用分配、精英与 Boss 进度奖励及自动选优）请见 [PPO v2 端到端训练](ppo-v2-training.md)。下文保留旧版联合评分与 PPO v1 对照实验记录。

## 同一局同时训练战斗和路线构筑

使用 `train_joint.py` 从 standard-continuation 基座开始。每次训练尝试只开一局。战斗中的打牌、用药、结束回合和弹窗选牌共享 768 个可训练参数。地图、战后卡牌奖励、商店、火堆和事件等选择使用完整路线与牌组特征的 4,096 个参数。两个评分层共用同一个冻结的 Transformer，同一批对局完成后一起更新并写入一个检查点。

战斗层使用下文的逐场回报，包括净失血和后续结局。战略层沿用楼层、过幕、通关回报和 12 轮 warmup。因而同一局的过幕结果能影响先前的选路和出牌。每轮报告中的 `combat_step_l2` 和 `strategy_update.step_l2` 分别给出实际参数变化量。异常和超时均从两组更新样本中排除。评测使用固定种子，两层均取当前最高分动作。

```powershell
$py = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
$env:OPENBLAS_NUM_THREADS = '1'
& $py scripts/beta_build.py
$joint = @('--out','runs/joint-a0-selection','--games','8','--rollouts-per-seed','2','--eval-games','8')
& $py scripts/train_joint.py @joint --rounds 4
# 续到总计 12 轮
& $py scripts/train_joint.py @joint --rounds 12 --resume
```

选牌观察和动作格式改变后，需要用新目录开训。用 `package_live.py --combat-checkpoint` 可以把联合检查点的战斗评分层带到 F7。

### 随机完整对局与难局补训

`train_curriculum.py` 每轮预留 12 次新的随机完整对局和 4 次难局采样。困难样本从随机训练对局自动收集，轮换优先抽取以下场景。

- 至少半血进场，战败前最后一次战斗决策仍有可用的主动药水。
- 至少 85% HP 进场，死在这场战斗。
- 第一幕 Boss。
- 战败前有直接击杀来袭敌人的机会但错过，观察到敌人的力量增加，或通过可见预览估计力量药水或伤害药水能让一个目标达到击杀线。

宿主按种子和保存的合法动作重放到该场战斗开场。脚本核对开场观察与原记录一致，再由当前模型采样后续动作。难局只给目标战斗的动作更新梯度，并保留后续对局对这些动作的回报。随机完整对局同时更新战斗层和局外层。每轮输出实际随机和难局次数，难局同一轮尽量使用不同种子。难例不足或复位失败时，用新的随机种子补齐，并记录失败。这个版本会先在宿主复现前缀，因此仍有重放时间。

```powershell
$curriculum = @('--out','runs/curriculum-a0','--attempts','16','--eval-games','8')
& $py scripts/train_curriculum.py @curriculum --start-from runs/joint-a0-selection/checkpoint.json --rounds 8
# 继续至本阶段累计 24 轮
& $py scripts/train_curriculum.py @curriculum --rounds 24 --resume
# 完成训练后，对自动选出的检查点运行一次保留种子评测
& $py scripts/train_curriculum.py @curriculum --test 32
```

`--start-from` 读取当前已保存的联合训练权重，按原进度计算探索温度与学习率。每轮的评测包括 `Q8QQWKP240JZ` 和 `YS2C9ALDRA2Y`，并记录错过的击杀来袭敌人的机会数、战斗最后阶段仍可用的药水，以及该场战斗用药次数。这两个种子和固定开发种子都排除在训练和难例池之外。

### 每个新种子跑两遍，同一难战开场跑三遍

从第 263 轮暂停的检查点开新阶段，再跑 249 轮即可累计完成 512 轮。新阶段读取检查点的权重和合规难例，继续原优化进度，每轮生成 12 个新种子，各用独立策略采样跑两遍。然后选一个难例，重放相同的开场动作，核对战斗开场状态，再用独立策略采样跑三遍。难例优先类型在未用药战败、高血量进场战败和第一幕 Boss 之间轮换。只将正常结束且成功到达开场的难例用于更新。若该轮没有可用开场，日志会显示 `focus 0/3`。

```powershell
$repeat = @('--out','runs/curriculum-a0-repeats','--fresh-seeds','12',
            '--rollouts-per-seed','2','--focus-rollouts','3','--eval-games','8')
& $py scripts/train_curriculum.py @repeat --start-from runs/curriculum-a0/checkpoint.json --rounds 249
# 中断后，沿用新阶段的参数续训
& $py scripts/train_curriculum.py @repeat --resume --rounds 249
```

### Steam 开局上的 PPO 对照

`train_aligned.py` 用同一份 Steam 进度快照、起始权重、每轮 12 个新种子和 3 个难战开场比较两种更新方式。随机种子各跑两遍，每个难战开场各跑三遍。两个训练组各跑 33 局，每轮固定开发集各跑 8 局。难例来自同轮或此前对照组在此宿主上的轨迹，两组重放同一开场并校验状态摘要。最终在相同的 32 个新种子上各贪心评测一次。

```powershell
$env:PYTHONPATH = (Resolve-Path 'artifacts/ppo-cuda').Path
$experiment = @('--progress','runs/aligned-ppo-v1-input/progress.save',
                '--checkpoint','runs/aligned-ppo-v1-input/start-checkpoint.json',
                '--out','runs/aligned-ppo-32','--rounds','32','--eval-games','32',
                '--workers','4','--device','cuda')
& $py -u scripts/train_aligned.py @experiment --resume
```

对照组沿用 NumPy 评分更新。PPO 组训练第六层 Transformer、战斗和局外评分权重以及状态价值头。同批数据最多训练三个小批量遍历，保留采样时的动作概率以计算比率和 KL。每个难战开场以其他已完成尝试的平均战斗回报为基线，辅助给该次开场的动作分配梯度。[PPO 原论文](https://arxiv.org/abs/1707.06347)

`round-*.json` 保存训练对局、开发集和更新指标。难例记录存活、净失血、用药次数和可从连续观察推断的敌人退场顺序。`ppo.pt` 保存两组最新权重、PPO 优化器和共有难例池。`comparison.json` 保存终点评测，包括确认过幕、异常前未判定的对局、Boss 进场血量和普通战净失血。宿主出错时保留已观察到的过幕里程碑，PPO 使用有效状态的价值估计处理截断。

### 从 PPO 检查点续训，增加随机种子覆盖

每轮 15 个新种子跑两遍，加 3 个不同难战开场各跑一遍，共 33 局训练。难例分别优先覆盖未用药战败、高血量进场战败和第一幕 Boss。`--init-ppo` 读取上一阶段学到的权重、优化器和已对齐的难例池。单次难例使用状态价值头提供的回报基线。训练 32 轮后，在同一批 64 个新种子上用直接取最高分动作的方式对比起点和终点。局外训练标签来自模型探索和引擎记录的实际结果。

```powershell
$env:PYTHONPATH = (Resolve-Path 'artifacts/ppo-cuda').Path
$next = @('--progress','runs/aligned-ppo-v1-input/progress.save',
          '--checkpoint','runs/aligned-ppo-v1-input/start-checkpoint.json',
          '--init-ppo','runs/aligned-ppo-32/ppo.pt',
          '--out','runs/aligned-ppo-15x2-3x1','--fresh-seeds','15','--focus-rollouts','1',
          '--rounds','32','--eval-games','64','--workers','4','--device','cuda')
& $py -u scripts/train_aligned.py @next
# 中断后使用同一组参数和 --resume
```

### 限制离线战斗求解器的算力

本机验证使用 CombatSolver 1.0.1 源码提交 `d72c52a0252248ec12ab027e335af1c92a48ccb5` 的 `OfflineSearchHarness`，对应 STS2 v0.113.1。`solve_bounded.py` 默认每次启动一个求解进程，搜索并行度为 1，预算为 1 秒、10,000 展开节点和 32 的 beam 宽度。Windows Job Object 将该作业的提交内存限定为 1 GiB。整个进程超过 15 秒就回收，时间包含启动。每次调用写入 `worker.json`。求解器的指标在 `result.json` 中，候选路线在 `route.json` 中。[CombatSolver 源码](https://github.com/Torch1230/CombatSolver) · [Windows 作业内存限制](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information)

```powershell
# 在仓库根目录执行，使用已编译的离线宿主
& $py scripts/solve_bounded.py --out runs/solver-new-fight
& $py scripts/solve_bounded.py --out runs/solver-new-boss `
    --generated-scenario external/CombatSolver/coverage/corpora/novelty/dev-09-necrobinder-boss.json `
    --milliseconds 2000 --nodes 20000
```

`--request` 接受离线宿主的请求 JSON，`--generated-scenario` 接受上游生成式战斗夹具。两者均面向当前离线求解器的格式。当前训练宿主的战斗快照接入需要单独验证。导出标签时保留求解预算、停止原因和路线的预计胜负，超时且没有找到胜利路线的样本可用于补搜。局外训练使用模型探索和分支对局产生的回报。

每轮自动更新参数，同时把开发集表现最好的可用权重保存为 `best-checkpoint.json`。按第一幕通过数、Boss 到达数、第一幕累计楼层优先排序。两条回归种子单独观察。评分相同时保留较早的权重，评测未完整结束的轮次跳过自动替换。`checkpoint.json` 保存最新训练进度供续跑。对已有实验，首次启用选优时比较仍可恢复的阶段起点和最近一轮。

保留种子独立于开发种子。新的 `holdout.json` 比较本阶段起点和自动选好的权重，`selected_round` 记录选中的轮次。报告含第一幕通过数、总尝试数、95% Wilson 区间、Boss 到达数和入场 HP。

## 单独训练战斗作对照

在 Windows v0.113.1 项目根目录运行。这个阶段固定局外决策为 standard-continuation 的基础策略，便于测量出牌与用药本身的影响。每局都从开局打起。每个训练种子默认采样两次，评测在固定种子上直接选最高分，全程无需搜索。

```powershell
$py = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
$env:OPENBLAS_NUM_THREADS = '1'
$combat = @('--out','runs/combat-a0-selection','--games','8','--rollouts-per-seed','2','--eval-games','8')
& $py scripts/train_combat.py @combat --rounds 4
# 同样的采样设置，续到总计 12 轮
& $py scripts/train_combat.py @combat --rounds 12 --resume
```

每轮 16 次训练尝试、一次批量更新、8 个固定种子评测。首次额外跑一次 8 局基础对照。检查点按整轮保存。可以先跑 4 轮查看变化，这个轮数仅用于初筛。检查点记录基座、宿主、训练代码和配置的指纹。

## 动作与回报

战斗中，256 个可训练读出系数直接作用在每个合法候选的 Transformer 表示上，另有 512 个哈希特征系数，初始全部为零，分数加在原模型 logits 上。特征含当前敌人意图、格挡、Osty HP、可见伤害预览、药水种类、房间类型和栏位占用。实际结算以引擎返回的 HP 为准。只有一个合法候选时无需计算策略梯度。

战中卡牌和药水引发的选择按 F7 的操作粒度打分。模型逐张从剩余候选里选，到最少数量才可以确认，达到最多数量后提交。模型看到候选的牌名、费用、升级、卡面效果、来源区域、当前战况、触发动作和已经选了哪些牌。抽牌堆只呈现弹窗列出的候选。无画面宿主收到一次合法提交，F7 则逐张点击游戏控件。战中生成的牌组和卡牌奖励也进入战斗评分层。

每场从第一条战中观察读入场 HP，到战后奖励或其他离开战斗的可观察状态读出场 HP，战斗中死亡记 0。该差值可以包含宿主已经结算的战后遗物治疗。普通战后奖励和商店、事件等局外选牌各按原入口处理。回报在战斗边界计算一次。

```text
本场回报 = 0.18 × 本场存活 + 0.65 × (出场 HP − 入场 HP) / 入场最大 HP
         + 1.5 × 进入下一幕
本场用于训练的回报 = 本场回报 + 0.94 × 下一场用于训练的回报
整局终点值 = 胜利 +2，死亡 −1
```

每个多候选战斗动作都参与本场及未来回报的策略梯度。即使用药立即降低本场失血，留药可能带来的后续精英或 Boss 收益也在优化目标内。两者的取舍需要跨多条实际轨迹学得。打出防御、服药、保留药水及进 Boss 的血量都没有单独奖励。整局出现宿主错误、超时或步数耗尽时，报告会保留状态，该局不参与更新。

一批全是败局时，整局胜利的 `+2` 在该批没有样本。败局的 `−1` 对最后一场贡献 `−0.94`，向前一场再乘一次 `0.94`，因此越早结束的轨迹越少得到后续存活奖励，且终局惩罚折扣越少。更新会减去批次的平均回报再标准化。相同终点值和相同剩余场数带来的常量会在中心化时抵消，剩余场数不同时仍有差异。首幕阶段可以用同种子对照比较死亡终点值 `−1` 和 `0`，同时保留战死时出场 HP 为 0 的信号，并以第一幕通过率为主指标。目前这两套终点值还没有做对照。

## 联合模型评分时读取的输入

| 输入 | 实际进入评分的内容 |
| --- | --- |
| 局面摘要 | 当前决策类型、幕、层、房间类型、战斗回合、剩余能量、角色当前与最大 HP、当前格挡、金币、牌组张数、抽牌堆与弃牌堆张数。Transformer 将类别映射到哈希词嵌入，将可用数值做带符号的 `log1p` 缩放。 |
| 角色 Buff、Debuff 与 Osty | 角色状态名称和层数分别编码。Osty 编码名字、存活、HP、最大 HP 和格挡。战斗层估算暴露伤害时，用预计来袭伤害减去角色格挡与 Osty 当前 HP。 |
| 敌人 | 每个活着的敌人有自己的表示，含名称、当前与最大 HP、格挡、攻击意图标志、Buff 与 Debuff 名称和层数。意图有类型。攻击意图有单次伤害和多段次数。总伤害用于战斗层的来袭估算。 |
| 手牌和合法动作 | 每张手牌的 ID、名称、类型、费用、目标类型和可打标志。每个合法打牌、用药、结束回合候选独立打分。卡牌有目标时，目标是候选的一部分。战斗层用宿主预览及卡面数值估算伤害、击杀、有效格挡、召唤、灾厄、抽牌和最后一点能量的机会成本。用药特征包含当前可用药的身份、栏位是否已满、房间类型和来袭压力。 |
| 战中选牌 | 所见候选、已选牌、最少与最多数量、可确认状态、来源区域和触发牌或药水。候选特征包括身份、费用、升级、关键词、卡面英文词及数值。 |
| 地图与构筑 | 基础评分读入场景、候选种类、动作参数和匹配的卡牌或选项。战略评分再用当前和最大 HP、HP 比例、幕、公开 Boss ID，以及沿地图候选未来四层的房型、宽度、分叉和房型与血量的交互。牌组吸收单体和范围伤害、格挡、抽牌、能量、费用、Osty、灾厄、诅咒或状态牌和升级比例，并估算单张拿牌、删牌或升级的边际变化。商店用价格，事件用宿主能提供的金币、治疗、失血等数值。 |

敌方意图来自本回合已经显示的类型和数值。游戏中存在固定衔接、条件分支和随机分支。编码目前没有敌人内部招式 ID、过去几回合的意图序列、下一招候选表或随机数状态。它能按敌人身份和回合数形成习惯，也能用显式 HP、Buff 和当前意图区分局面。一般战斗状态中的弃牌明细、最大能量、完整遗物清单和状态描述虽然出现在宿主数据中，但当前评分只消耗各输入行所列的特征。完整抽牌顺序留在宿主，选牌弹窗可以看到引擎允许选的牌。

## 读评测

查看输出目录的 `anchor.json` 和 `round-0001.json` 等报告。先对比固定种子的第一幕 Boss 到达数、过幕数和总尝试数，再看 Boss 入场 HP、存活战斗的净失血中位数以及对应战斗数量。遇敌和生还样本都可能因为策略变化而改变。每场还记录入场与出场 HP、类型、可用药决策数和实际用药数，可以按房间检查。计数 `greedy_changes` 比较新策略所遇每个多选局面上的新旧最高分动作。

联合训练日志的 `before_travel_hp` 是进入该局最后一层前、选择该地图节点时的 HP 和最大 HP。未观察到移动前血量时显示 `-`。JSON 的 `last_floor_entry` 保留移动前和首次可操作时的快照。此前的战斗报告可从同楼层的 `fights[].hp_in` 回查首次战斗观察时的 HP。

`selection_decisions` 统计战斗中有多个合法选项的弹窗决策。`selection_changes` 统计这些局面里，战斗评分层对最高分选项的改变次数。要评估选牌带来的净失血变化，可以在相同战况和候选牌下比较不同选法，或增加独立的完整对局评测。
