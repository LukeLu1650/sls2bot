# 标准模式实机接管

从 Steam 启动 Windows v0.113.1，启用 `Necrobinder · Live Pilot`，自己开任意种子的单人标准骨头人对局。实机面板默认关闭。用 **F7** 或面板开关启用，模型从正在运行的对局读取决策，直接走游戏的出牌、用药和选项入口。关闭会丢弃待执行的模型选择，并停止本地推理进程。已经提交给游戏的动作会继续结算。

默认模型是标准 A0 首战继续训练的 18.10M 参数检查点。面板显示所选动作、最高三项分数和 Python 推理毫秒数。Mod 每 45 ms 检查一次游戏是否接受决策，在动作执行、动画或过场中等待。调快游戏内动画速度可以缩短实际等待。强制流程如领取金币或继续对话由实机适配器执行。遇到无法处理的界面、异常或游戏拒绝动作时，面板交回控制权。卡牌多选逐张选取再确认，F7 会给每张候选发独立索引、牌面信息和当前选牌数量。

用 `--checkpoint` 打包联合训练或课程训练检查点时，F7 同时使用战斗和路线构筑权重。战斗层用于出牌、用药和战中选牌。路线构筑层用于地图、奖励、商店、事件和其他局外选择。只部署战斗层时使用 `--combat-checkpoint`。

```powershell
$py = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
& $py scripts/package_live.py --checkpoint runs/curriculum-a0/checkpoint.json --out artifacts/NecroPilot-current.zip
```

装好支持两组权重的 Mod 后，退出游戏，从工作区更新到同一编码和基座的新检查点。

```powershell
& $py scripts/update_live_weights.py runs/curriculum-a0/checkpoint.json
```

脚本先验证检查点和已安装的推理代码，备份旧文件到 `runs/necropilot-weight-backups/`，然后写入并核对新权重。Steam 库在别处时传入 `--game "实际安装目录"`。手动更新时，从同一份 ZIP 的 `NecroPilot/` 一起复制 `combat.dat`、`strategy.dat` 和 `model.dat` 到游戏的 `mods/NecroPilot/`。基座或推理代码更新时用整包覆盖。

本机构建已通过编译。打包后的推理进程用训练宿主的真实标准 A0 首战验证了 17 个决策，均与同模型的直接推理一致。推理中位数为 13.6 ms，最慢 20.1 ms，数值仅涵盖 Python 模型计算。尚未在已登录 Steam 的游戏里验证中途切换或整局动画用时。

实机版通过私有标准输入输出调用本地 Python 和 NumPy。安装包的 `NecroPilot/pilot.cfg` 指向打包时的 Python。本机已经有对应解释器；若转移到别的电脑，填写可运行 NumPy 的 Python 路径。Mod 安装只需要把 ZIP 内的 `NecroPilot` 文件夹放进游戏 `mods/`。ZIP 的 `Source` 目录包含派生自上游 GPL-2.0-only 实码的构建源码。文件名与旧回放 Mod 不同。[上游实机控制接口](https://github.com/HalfVulpes/sts2ai/tree/d667489560cc46154638e1208ff032ed96707aa3/mod)

## 路线和选牌打分的现状

现有权重按可达节点坐标和当前局面打分。训练宿主返回可见 Boss 和牌组信息，但现行特征编码没有把 Boss 标识、可见后续路径、牌组组成或普通怪与精英的分布用于评分。训练教师在非战斗场景用简单对照策略，首战胜率也无法衡量路线或构筑水平。上游实机适配器有独立的路线统计逻辑，其输出未接入这个 checkpoint 的打分。

路线规划可先用公开信息做成一组新的训练特征。对于每个可选房间，沿公开地图统计接下来几层的休息、商店、精英、分叉及换路线机会。对于当前牌组，用费用和过牌估算前几回合的单体输出、范围输出、格挡、能量和抽牌，另记 Osty、灾厄及成长。Boss 用已公开的标识和机制，普通怪与精英用当前幕的公开遭遇池和已走房间形成风险估计。实际未揭晓的敌人身份和抽牌顺序留在环境中。

这些特征可以一起用于估计打精英的风险、商店补强的收益，以及一张待选卡能否填补缺口。比如同一个精英房对「单体伤害高但范围输出不足」的牌组，价值会随该幕的精英池改变。先用相同种子和采样预算比较现有策略、只增加路线特征、再增加 Boss 与牌组指标三组。看每幕通过率、进 Boss 时血量和标准 A10 完整胜率。可以在训练侧计算公开地图的统计，实战继续只做一次前向。
