# Windows v0.113.1 骨头人基线 · 2026-11-23

在本机 Windows x64 上，以 Steam 游戏程序集 v0.113.1（游戏提交 `d09d48e5`）运行。训练设计覆盖 public-beta v0.113.0 引入的规则。下面的测量都来自 v0.113.1。原生 .NET 9 宿主负责游戏推进，Python 和 NumPy 负责策略。摄政王复现和 Ubuntu 均为可选路径。[官方 v0.113.0 公告](https://store.steampowered.com/news/app/2868840/view/708914333639245837) · [社区宿主 PR 107](https://github.com/wuhao21/sts2-cli/pull/107)

## 实测结果

| 骨头人策略 | 模式 | 完成 / 尝试 | 胜场 | 正常终局楼层中位数（第一幕） | 总动作 |
| --- | --- | ---: | ---: | ---: | ---: |
| [规则对照](../results/windows-v0.113.1/standard-a0/rule/summary.json) | 标准 A0 | 3 / 4 | 0 | 7 | 358 |
| [随机初始化 Transformer](../results/windows-v0.113.1/standard-a0/initial/summary.json) | 标准 A0 | 4 / 4 | 0 | 2 | 46 |
| [64 次示范更新后的 Transformer](../results/windows-v0.113.1/standard-a0/imitation/summary.json) | 标准 A0 | 3 / 4 | 0 | 2 | 1,144 |
| [规则对照](../results/windows-v0.113.1/rule/summary.json) | 标准 A10 | 7 / 8 | 0 | 7 | 669 |
| [随机初始化 Transformer](../results/windows-v0.113.1/initial/summary.json) | 标准 A10 | 8 / 8 | 0 | 2 | 78 |
| [64 次示范更新后的 Transformer](../results/windows-v0.113.1/imitation/summary.json) | 标准 A10 | 8 / 8 | 0 | 2 | 1,067 |

标准 A10 使用 8 个预定种子，A0 使用其中前 4 个，各难度内的三组对照共用种子。全部为单人骨头人、全解锁。规则策略优先打第一张可打牌，其余选择用独立的固定随机数生成器。每个难度各自在另一个种子 `RWN5T3VD6N2E` 上收集规则示范，对有多个候选的决策做 64 次交叉熵更新，学习率 0.001，冻结主干。每条候选打分后贪心选择。两组 A10 神经策略合计执行 1,145 个动作，逐局前向次数均等于动作数；推理路径中没有搜索或模拟分支。[策略源码](../scripts/necro_policy.py) · [A10 训练局记录](../results/windows-v0.113.1/imitation/training.json) · [A0 训练局记录](../results/windows-v0.113.1/standard-a0/imitation/training.json)

随机策略八局共出牌 2 次。短训后共出牌 715 次，八局中有七局仍停在第 2 层，另一局到第 4 层；其中一个种子在第 2 层就做了 652 个动作。模仿更新显著改变了出牌行为，整局推进仍弱于简单规则。8 局零胜对应 Wilson 95% 区间约 0%–32.4%，不宜推断长期胜率。下一轮适合在 A0 检查能否击杀首战敌人，记录伤害、格挡、击杀与战斗回合数，再决定是否扩大训练。

规则组中 `8668D8L9UBCT` 到达第一幕第 3 层、剩余 44 HP。Dense Vegetation 的 Rest 选项在游戏方法 `DenseVegetation.Rest()` 触发空引用，独立复跑重现。该局记录为 `engine_error`，包含在尝试总数内，楼层中位数仅统计正常终局。其余七局正常死亡。修复这个事件后用同一面板另报结果，原始测量保留。

标准 A0 出现两个未结算样本。规则组在 `MHJA6MWFPM5D` 第 7 层选择事件时遇到宿主异常。短训模型在 `Q484ZHBQM8VB` 第 2 层用满 1,024 步，记录为 `step_limit`。下轮训练需同时观察首战胜率和回合数，避免模型学会长期格挡却没有消灭敌人。

## 复现

从本仓库根目录在 PowerShell 执行。需要 Windows x64、Git、.NET 9 SDK、Python 3.11+；本次使用 NumPy 2.3.5。若全局 `python` 指向 Windows Store 占位程序，请把 `$py` 指向实际 Python。本机可用如下设置。

```powershell
$py = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $py -c "import numpy; print(numpy.__version__)"
& $py scripts/beta_build.py
& $py scripts/beta_eval.py --policy rule
& $py scripts/beta_eval.py --policy initial
& $py scripts/beta_eval.py --policy imitation
```

游戏不在默认 Steam 库时，向 `beta_build.py` 传入 `--game "实际安装目录"`。构建器检查版本、游戏 commit 和主程序集 SHA-256，将依赖复制到独立目录，从游戏 PCK 刷新本地化并构建宿主。使用原始游戏 DLL，宿主使用 .NET 测试模式和 Godot 替身。每轮评测前重新核对宿主、依赖和本地化指纹。[原生构建脚本](../scripts/beta_build.py) · [锁定的宿主源码](https://github.com/wuhao21/sts2-cli/tree/9842f7dbb52f2b35d2bad73728f9c7ac54679ed9)

重现已保存的标准 A10 短训策略，或在标准 A0 建立训练起点，可分别执行：

```powershell
& $py scripts/beta_eval.py --policy imitation --load-from results/windows-v0.113.1/imitation --games 1
& $py scripts/beta_eval.py --mode standard --ascension 0 --policy initial --games 4
```

保存物由初始化种子、NumPy 版本、读出头参数和模型 SHA-256 组成。加载时重新生成主干并核对完整模型指纹。已实际重载短训权重，在 `YESSX3NDTXXZ` 重现了同一终局、27 个动作和各类动作计数。运行明细写入新建 `runs/` 子目录；有异常或达到上限时脚本保留结果并返回非零。每次最多 8 个评测种子，每局最多 1,024 个动作、90 秒；示范训练最多 2 个独立种子和 64 次更新。[评测脚本](../scripts/beta_eval.py) · [已保存的实验协议](../results/windows-v0.113.1/imitation/protocol.json)

## 成绩的适用范围

- **模型。** 训练与推理使用 6 层、256 维、4 个注意力头；4,096 桶特征嵌入，前馈宽度 5,024。模型以手牌、敌人、意图、状态及候选动作等有限字段为输入，当前候选得分取最大值。不同模组、特征编码或 NumPy 随机初始化协议要重新核对。原论文的旧版词表和训练权重未参与这次试验。
- **宿主。** 这是调用当前游戏程序集的社区无画面宿主测量。宿主自动处理部分奖励；桥接器的药水、商店和多选组合有覆盖上限。胜负由宿主的终局状态报告，完整图形客户端上的逐步规则一致性仍待核对。[桥接协议与限制](../scripts/beta_env.py)
- **版本。** 当前只推进标准模式。历史的 [两局 Blitz 冒烟记录](../results/windows-v0.113.1/blitz/summary.json) 供备查。只有拿到 v0.113.0 的实际程序集并新建构建指纹、运行对应种子后，才能报告该精确版本的结果。
