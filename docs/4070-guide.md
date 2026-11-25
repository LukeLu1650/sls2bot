# 骨头人（Necrobinder）RL 协作训练指南 · RTX 4070 游戏本（32 GB 内存）

这份文档面向 **Windows + RTX 4070 游戏本（8 GB 显存 / 32 GB 内存）** 的协作节点，采用 **独立跑对照实验（方式 A）**——无需内网穿透或双机联调，按步骤配好环境即可直接开跑。

> **为什么不用 Mac M2？** 无头宿主依赖 `.NET 9 + 0Harmony / MonoMod` 运行时 IL 补丁，macOS ARM64 容易触发 JIT 内存权限兼容问题，且 MPS 对变长打包注意力支持较差；直接只用 **4070 Windows 游戏本** 最省心、吞吐高一个量级。

---

## 一、你这边收到的文件清单（总计 < 40 MB）

把我发给你的压缩包解压到**不含中文且不在 OneDrive 同步目录里的纯英文路径**（强烈推荐如 `D:\sts2-bot\`，避免 OneDrive 后台同步干扰或路径编码问题）：

```text
D:\sts2-bot\
├─ scripts\                              # 训练与评测脚本（train_ppo_v2.py 等）
├─ patches\                              # 无头宿主补丁（HeadlessSteamRun.cs 等）
├─ tests\                                # 29 项单元测试
├─ runs\
│  ├─ aligned-ppo-v1-input\progress.save # 统一 Steam 进度快照（必须一致）
│  ├─ benchmark-a0-128.json              # 128 锁定盲测种子（统一评测面板）
│  └─ ppo-v2-a0-s6-r180\
│     └─ best-checkpoint.pt              # 本轮对照的统一父权重（S6 最佳权重）
└─ docs\friend-4070-guide.md             # 本文档
```

---

## 二、一次性环境准备（约 5–10 分钟）

### 1. 安装基础运行环境
1. **Git**（Windows 版）
2. **.NET 9 SDK**（版本 `9.0.x`，安装后在 PowerShell 运行 `dotnet --list-sdks` 应看到 `9.0.xxx`）
3. **Python 3.11 或 3.12**（建议官网安装包或 venv，避开 Windows Store 版）
4. 创建虚拟环境并安装 **PyTorch CUDA + NumPy**（在 `D:\sts2-bot` 目录打开 PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

验证 GPU 可用（应输出 `True` 和 `NVIDIA GeForce RTX 4070 Laptop GPU`）：
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. 核对本地正版《杀戮尖塔 2》版本（最关键！）

两台机器必须使用**完全同一版 `v0.113.1` 的 `sts2.dll`**，否则随机种子、卡池与观测都会漂移。请把路径替换为你本机的 Steam 安装路径后执行：

```powershell
$game = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"
(Get-FileHash "$game\data_sts2_windows_x86_64\sts2.dll" -Algorithm SHA256).Hash.ToLower()
```

核对输出是否**逐字等于**：
```text
62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3
```
> 如果哈希不一致（说明 Steam 已自动更新到更高测试版，如 `v0.114.0`），先停一下告诉我，别直接往下编译。

### 3. 编译并校验无头宿主

```powershell
python scripts\beta_build.py --steam-parity --game "$game"
```
看到输出 `Verified v0.113.1 d09d48e5: ...\host-steam.json` 即表示宿主构建和校验全部通过。

### 4. 跑一遍单元测试（约 4 秒）

```powershell
python -m unittest discover -s tests -v
```
确认输出 `Ran 29 tests ... OK`，说明环境 100% 就绪。

---

## 三、启动对照实验：Act 2 / Act 3 专项战斗重放（S7 对照线）

### 实验目的
- 起点与我这边完全一致：`runs/ppo-v2-a0-s6-r180/best-checkpoint.pt`（S6 最优权重）。
- 每轮总预算保持 **48 局**不变，但把 **25% 预算（`4×3 = 12` 局）** 专门用于从真实对局中提取并重放 **Act 2 / Act 3 的普通战、精英战与 Boss 战**（`--focus-acts 2,3`），其余 **75%（`18×2 = 36` 局）** 继续打完整局。
- 脚本会自动记录 **R0 贪心基线**（`round-0000.json`）、保留开发集 **Top-3 权重**（`top-k/`），并且因 `--out` 目录名独立，**训练种子会自动与所有历史阶段零重合**。

### 专为 4070 移动版（8 GB 显存 / 32 GB 内存）调优的启动命令

```powershell
.\.venv\Scripts\Activate.ps1

python -u scripts\train_ppo_v2.py `
    --progress runs/aligned-ppo-v1-input/progress.save `
    --init runs/ppo-v2-a0-s6-r180/best-checkpoint.pt `
    --out runs/ppo-v2-a0-s7-a2focus-4070 `
    --preset base --value-bins 41 --ascension 0 `
    --rounds 120 --fresh-seeds 18 --rollouts-per-seed 2 `
    --focus-cases 4 --focus-rollouts 3 --focus-acts 2,3 `
    --dev-games 64 --eval-every 8 --eval-games 64 `
    --top-k 3 --minibatch 256 --epochs 3 `
    --lr 4e-5 --target-kl 0.015 `
    --workers 10 --device cuda
```

> **资源占用参考**：显存约 **2 GB**（远低于 8 GB 上限）、内存约 **6–10 GB**（32 GB 非常宽裕）。建议插电并开启游戏本散热模式；如果白天需要同时轻度办公，可以把 `--workers 10` 临时改成 `8` 再用下面命令续跑。

### 如果中途关机 / 中断：一键无缝续跑

去掉 `--init`，在原命令末尾加 `--resume` 即可从上一轮完整断点继续（自动恢复模型、优化器、难例池与 Top-3 记录）：

```powershell
python -u scripts\train_ppo_v2.py `
    --progress runs/aligned-ppo-v1-input/progress.save `
    --out runs/ppo-v2-a0-s7-a2focus-4070 `
    --preset base --value-bins 41 --ascension 0 `
    --rounds 120 --fresh-seeds 18 --rollouts-per-seed 2 `
    --focus-cases 4 --focus-rollouts 3 --focus-acts 2,3 `
    --dev-games 64 --eval-every 8 --eval-games 64 `
    --top-k 3 --minibatch 256 --epochs 3 `
    --lr 4e-5 --target-kl 0.015 `
    --workers 10 --device cuda --resume
```

---

## 四、跑完后如何验货 & 发回哪些文件

### 1. 用统一的 128 锁定未见种子做同场复核（约 4 分钟）

训练跑完 120 轮并生成 `evaluation.json` 后，顺手跑一条 128 种子盲测（把你的最佳权重和父权重放在完全相同的 128 个未见种子上直接对比）：

```powershell
python -u scripts\eval_compare.py `
    --progress runs/aligned-ppo-v1-input/progress.save `
    --seeds-from runs/benchmark-a0-128.json `
    --out runs/benchmark-a0-128-s7-4070.json `
    --games 128 --workers 10 --device cuda `
    S6-best=runs/ppo-v2-a0-s6-r180/best-checkpoint.pt `
    S7-best=runs/ppo-v2-a0-s7-a2focus-4070/best-checkpoint.pt `
    S7-final=runs/ppo-v2-a0-s7-a2focus-4070/ppo.pt
```

### 2. 只需要打包发回这几个文件（合计 < 120 MB）

**不用传** 120 份 `round-*.json` 和庞大的优化器状态 `ppo.pt`，只发以下核心产物即可：
1. `runs/ppo-v2-a0-s7-a2focus-4070/protocol.json`
2. `runs/ppo-v2-a0-s7-a2focus-4070/round-0000.json`（R0 基线）
3. `runs/ppo-v2-a0-s7-a2focus-4070/best-checkpoint.pt`（约 29 MB）
4. `runs/ppo-v2-a0-s7-a2focus-4070/top-k/` 整个文件夹（3 个候选权重 + `manifest.json`，约 87 MB）
5. `runs/ppo-v2-a0-s7-a2focus-4070/evaluation.json`
6. `runs/benchmark-a0-128-s7-4070.json`（如已跑完 128 盲测）
7. 如 `diagnostics/` 文件夹非空，把里面的 `*.json` 也一并打包给我排查。

---

## 五、后续第二阶段预告（AlphaDefect 高吞吐原生栈）

我这边正在原生 Windows 上压测 `AlphaDefect-Public` 的高吞吐常驻引擎栈（目标是把采集速度从 ~140 步/秒拉到 1,000+ 步/秒，冲 **A10 连胜**）。等我这边三档测速调通后会同步给你一套新配置，**提前记一下 4070 移动版（8 GB 显存）的专属安全参数**：
- 模型（19.4M）在 BF16 下用 **`minibatch = 1024`** + **`grad_accum = 4`**（稳压进 8 GB 显存，不触发 CUDA OOM）；
- 32 GB 内存可安全开 **`actors = 2, envs_per_actor = 12 ~ 16`**（共 24–32 路常驻引擎，约占 5–6 GB 内存）；
- Windows 下关闭 `learner_compile`（保留 `learner_fast = true` 打包变长 SDPA）。

---

## 六、常见问题速查

- **看到 `CIRCUIT BREAKER TRIGGERED` 自动停机**：这是保护熔断机制（单轮异常率超 25% 自动存档退出）。直接把 `runs/.../diagnostics/` 里的小 JSON 发我就行，已跑完的轮次完全不会丢。
- **报 `OutOfMemoryException`**：通常是后台开了占内存的大软件（如浏览器几十个标签页或 IDE）导致 Windows 虚拟内存吃紧；关掉不用的软件并把 `--workers` 从 `10` 降到 `8`，用 `--resume` 接着跑即可。
- **如何判断训练进展正常**：终端每轮都会打印 `Round N/120 saved (... kl=..., err=0/48, best=R...)`，正常情况下 `err=0/48`，每 8 轮跑一次 64 种子开发集选优。
