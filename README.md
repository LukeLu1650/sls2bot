# 杀戮尖塔 2 · 骨头人零搜索策略

当前只训练和评测 **标准模式** 的死灵契约师（Necrobinder）。从 A0 学习战斗与构筑，目标是标准 A10 整局通关。每次决策读入可见局面及候选动作，一次前向推理直接选动作。本地基线使用 **Windows public-beta v0.113.1** 游戏程序集、原生 .NET 9 和 NumPy，模型有 **18,099,905 个参数、6 层、宽度 256、4 头**。

| 入口 | 内容 |
| --- | --- |
| [Windows 骨头人基线](docs/windows-baseline.md) | 本机构建、复测命令、实测结果、模型与宿主的适用范围 |
| [当前模型与可选动作](docs/current-policy.md) | 标准模式的动作、观测，以及继续训练后的首战结果 |
| [实机接管 Mod](docs/live-pilot.md) | 用 F7 随时接管当前标准骨头人对局，并查看模型打分 |
| [骨头人训练设计](docs/necrobinder.md) | 玩家可见特征、角色机制、PPO 课程和实机验收 |
| [游戏版本迁移](docs/versions-and-modes.md) | 标准模式的新构建验证和历史模式资料 |
| [GPT‑6.2 Sol 交接提示词](docs/sol-handoff.md) | 可独立执行的后续开发与验证任务，有明确的预算和停止条件 |
| [旧版摄政王实验（可选）](docs/regent.md) | 论文所用 Linux v0.107.1 的历史复现方法 |

本仓库包含实验代码和小批评测记录。构建脚本将公开宿主固定到特定提交，游戏文件从自己的 Steam 安装复制到私有工作目录。源码检出、完整程序集、运行缓存和训练中间文件分别位于 `external/`、`artifacts/` 或 `runs/`，可从同一构建重新生成。

视频作者公开的架构与自我对局规模用于设定目标。Chen 与 Xie 的 v0.107.1 工作提供了零搜索 PPO、留出种子与实机导出方面的参照。Windows 宿主采用社区 v0.113.1 适配补丁。[原视频](https://www.bilibili.com/video/BV19Bxq6wE9k/) · [论文](https://arxiv.org/abs/2611.12088) · [v0.113.1 宿主适配](https://github.com/wuhao21/sts2-cli/pull/107)
