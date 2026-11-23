# 标准模式骨头人 · 路线与构筑训练

先保留当前已经训练过首战的 checkpoint，再比较两种新观察。每个战略决策同时给所有候选动作打分，最终评测和实机目标仍是直接选最高分。

## 模型需要改哪里

| 模块 | 这阶段的变动 | 对候选的影响 |
| --- | --- | --- |
| 局面编码 | 逐层统计公开地图接下来四层的房间比例、可达宽度和真实分叉比例。另读当前 HP、最大 HP、血量比例和可见 Boss。牌组汇总包括单体伤害、范围伤害、格挡、抽牌、能量生成、费用、Osty 召唤、灾厄和卡手牌指标 | 同一选择可随血量、路线和牌组缺口得到不同分数。走图时从该候选房间起算，其他战略界面从当前房间的后继起算 |
| 动作编码 | 给每个合法候选一行特征。选牌、跳牌、删牌、升级、事件选项、商品、休息点与地图节点分别编码 | 卡牌加入或删去时估算方向和幅度。升级优先使用升级前后差值。跳牌是零牌组变化的有效候选。事件保留游戏报告的选项和部分数值。商店分别记录金币价格和卡牌能量费用 |
| 动作打分 | 冻结原有 6 层 Transformer 及旧打分头。对战略动作新增 `旧分数 + 线性调整`，每组训练 4,096 个参数。战斗动作继续用旧网络 | 可以测出战略输入是否有价值，也可以保留原有出牌能力 |
| 训练目标 | 三组分别从同一权重开始。训练时在合法战略候选中采样，用整局结局、到达幕和楼层做有限步数的策略梯度更新。评测时选最高分 | 可以学到后续过幕和整局结果中的差异。训练局的宿主异常与超时不用于梯度 |

对照包括只评测的 **anchor** 和三组同步训练的模型。**current** 使用原来的模型输入和通用战略偏好，**route** 增加公开地图统计，**boss_deck** 再增加 Boss 标识、牌组代理指标和动作对牌组的预估变化。三组用同一批训练种子和开发种子，保留同样的训练预算。牌组指标是基于卡面可见数字的近似，特殊连锁、卡牌长期成长和隐藏敌人序列没有真实反事实模拟。当前也尚未纳入分幕普通怪与精英遭遇池和每个 Boss 的机制标签，适合作为后续单独实验。[战略特征](../scripts/strategic_features.py) · [训练入口](../scripts/train_strategy.py)

## 预计的训练轮数

一轮对应**一批新对局和一次批量参数更新**。三组各用 `--games` 个相同的种子采样。例如 `--games 8` 一轮实际发起 24 次训练尝试，每组最多更新 1 次。每局可能产生很多选牌、路径和事件决策。对每局有效决策先求梯度平均，再按整局结果聚合成这一次更新。失败的宿主局、超时和步数上限不参与梯度，报告用 `valid_training_episodes`、`optimizer_steps` 和 `weights_changed` 分别记录有效局数、更新次数和参数是否真的变化。在复测轮，三组另外各跑 `--eval-games` 局；第一轮还测一次冻结的 anchor。以下工时按单进程每局 **8–25 秒**估算，包含训练局和开发评测，排除最终测试集。

| 阶段 | 建议轮数与采样量 | 三组对局量 | 预计 Windows 用时 |
| --- | --- | ---: | ---: |
| 标准 A0 初筛 | 4 轮，每组每轮 8 局，每 2 轮各复测 8 局 | 152 局 | 约 20–65 分钟 |
| 标准 A0 延长观察 | 总计 12 轮，沿用初筛参数 | 总计 440 局 | 约 1–3 小时 |
| 标准 A5 迁移 | 16–24 轮，每组 16 局，每 4 轮各复测 16 局 | 976–1,456 局 | 约 2–10 小时 |
| 标准 A10 改善 | 24–48 轮，每组 32 局，每 4 轮各复测 32 局 | 2,912–5,792 局 | 约 6.5–40 小时 |

4–12 轮是低成本初筛。它只更新 4,096 参数的战略打分层，用来查差异与样本覆盖。A10 表中 24–48 轮也只有每组 768–1,536 局、最多 24–48 次更新，适合观察趋势。现阶段无法预测达到某个 A10 胜率所需轮数。正式拉长可先定 **每组 64 局 × 100 轮**，每 10 轮复测，观察开发集是否进步后再决定是否增加到 200 轮。100 轮是每组 6,400 次训练尝试，三组合计 19,200 局，仍需按实际有效局与 Boss 样本数判断。如果大多数对局停在第一幕，增加早期对局无法提供足够的后期决策反馈。最终报告建议用 **每组至少 128 个预留种子**，胜场仍稀少时保留区间。

此前短检查的一轮包含三组共 6 局，耗时 19.3 秒，累计采样 57 个有多个候选的战略决策。另有 1 局未训练模型对照。这些对局都在第一幕结束，后期对局的工时需要重新量测。

`Dense Vegetation → Rest` 的宿主空引用已修复。补丁只绕开无画面宿主的音效、演出等待和震屏分支，保留原始治疗、休息触发和战斗。`R04CTH7DYALA` 与 `8668D8L9UBCT` 两个 A10 种子分别从 30 恢复到 49 HP、从 44 恢复到 63 HP，然后进入同一战斗并正常结算。原 A10 留出种子用冻结对照和三组新版检查点重跑，均越过出错的第 5 层，在第 6 层正常死亡。可运行 `scripts/check_dense_vegetation.py` 复核事件。[宿主补丁](../patches/HeadlessEventPresentation.cs)

## 在你的 Windows 自主启动

在当前仓库根目录打开 PowerShell。以下使用已有的 Python 与 NumPy，CPU 即可。本机已重建匹配的游戏宿主。如在其他检出目录操作，先用同一个 Python 运行 `scripts/beta_build.py` 从本地 Steam v0.113.1 安装准备宿主。新版地图与牌组特征采用协议 2，旧检查点继续保留用于旧实验。请用下文的 `-v2` 新目录。

```powershell
$py = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $py scripts/train_strategy.py --help

# 先跑一个短检查，约 1–3 分钟，使用独立目录
& $py scripts/train_strategy.py --out runs/strategy-small --rounds 1 --games 1 --eval-games 1 --max-steps 180 --episode-timeout 25

# A0 初筛
$a0 = @('--out','runs/strategy-a0-v2','--ascension','0','--games','8','--eval-games','8','--eval-every','2')
& $py scripts/train_strategy.py @a0 --rounds 4

# 从上次完整保存的轮次继续，到总共 12 轮
& $py scripts/train_strategy.py @a0 --rounds 12 --resume
```

训练以前台方式运行，关闭当前任务不会自动定时启动。按 **Ctrl+C** 可停止，在已有输出目录上加 `--resume` 会从最后完成的轮次继续。保持原来升阶、局数、种子键、评测间隔、单局限制和代码版本。`--rounds` 表示目标总轮数，每轮生成检查点。每个输出目录只启动一个训练进程。默认单局上限是 750 个动作与 65 秒，进入后期后可在新阶段适当提高。

等 A0 开发集有足够完整的路线样本，可从该检查点建立新的难度阶段。准备好最后候选后，用独立测试种子评测一次。

```powershell
$a5 = @('--out','runs/strategy-a5','--ascension','5','--games','16','--eval-games','16','--eval-every','4','--seed-key','necro-a5')
& $py scripts/train_strategy.py @a5 --rounds 16 --start-from runs/strategy-a0-v2/checkpoint.json
$a10 = @('--out','runs/strategy-a10','--ascension','10','--games','32','--eval-games','32','--eval-every','4','--seed-key','necro-a10')
& $py scripts/train_strategy.py @a10 --rounds 24 --start-from runs/strategy-a5/checkpoint.json
& $py scripts/train_strategy.py @a10 --final --final-games 128
```

如果 A10 初筛已经有足够的后期样本，可建一个正式预算目录，接着训练 A10 检查点。按 100 轮、每 10 轮一轮开发评测计算，三组合计约 21,184 次对局尝试，未含最后的留出评测。按每局 8–25 秒的串行假设约需 47–147 小时，先用少量轮次测本机后期对局耗时。

```powershell
$long = @('--out','runs/strategy-a10-long-v2','--ascension','10','--games','64','--eval-games','64','--eval-every','10','--seed-key','necro-a10-long')
& $py scripts/train_strategy.py @long --rounds 10 --start-from runs/strategy-a10/checkpoint.json
& $py scripts/train_strategy.py @long --rounds 100 --resume
```

也可以测 A0 检查点直接迁移到 A10，使用 `@a0 --final --test-ascension 10 --final-games 128`。当前阶段输出适用于无画面宿主评测。要将新战略权重用于 F7 实机版，需要在游戏内提供同样的地图、Boss、牌组和候选物品数据，再校验同一局面的动作分数。当前已安装的 Mod 仍使用原首战 checkpoint。

## 怎么看结果

- `checkpoint.json` 保存三组参数和最后完成的轮次。每份 `round-NNNN.json` 保存三组训练和开发局、候选覆盖及结算状态。`anchor.json` 是同开发种子的原模型参照。
- `act_clear_rate` 以全部尝试为分母。进入下一幕即通过前一幕，第三幕以整局通关为准。`win_rate` 和 Wilson 95% 区间同样按全部尝试计算。宿主异常、步数上限和真实死亡有独立计数。
- `boss_hp` 分幕和第几个 Boss 汇总。能从地图观察到进入前血量时写入 `median_before_travel_hp`。另外保留每场首次可操作战斗状态的血量及比例，覆盖自动转场。未遇到 Boss 的局不会混入血量中位数，报告里保留对应样本数。
- `offered` 和 `selected` 用来检查训练是否真正覆盖加牌、跳牌、删牌、事件、商店与火堆。若某类操作几乎未出现，相应结论应等待更有覆盖的训练局和评测局。
