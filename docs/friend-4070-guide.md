# 骨头人（Necrobinder）高吞吐 RL 训练指南 · AlphaDefect `v0.114.0` 原生 Windows（RTX 4070 游戏本）

这份文档面向 **Windows + RTX 4070 游戏本（8 GB 显存 / 32 GB 内存）**，直接使用已打好原生 Windows 补丁的 **`AlphaDefect-Public/game-v0.114.0/`** 高吞吐栈（对应 2026-11-25 最新 Steam `public-beta` `v0.114.0`，零真人数据、严格 No-SL、目标 A10 骨头人）。

> **为什么跳过 Mac M2？** 无头宿主依赖 `.NET 9 + 0Harmony` 运行时 IL 补丁，macOS ARM64 容易触发 JIT 权限问题，且 MPS 对变长打包注意力支持差；只用 **4070 Windows 游戏本** 最省心、速度高一个量级。

---

## 一、你这边收到的文件（< 30 MB，不含游戏本体）

把我发给你的 `game-v0.114.0` 目录解压到**不含中文且不在 OneDrive 同步目录里的纯英文路径**（强烈推荐 `D:\sts2-v114\`）：

```text
D:\sts2-v114\
├─ engine\          # 已含 Windows 补丁（VirtualFs.cs 支持 STS2_USER_DIR）与 v0.114.0 词表/内容目录
├─ python\          # 已含原生 Windows 适配（dotnet 自动定位、异步管道读线程、非 mmap 权重加载、publish 重试）
├─ tools\
│  ├─ extract_pck.py       # 提取游戏本地化文本
│  ├─ win_bench.py         # 原生 Windows 三档吞吐测速脚本
│  ├─ necro_win_4070.json  # 专为 4070 移动版（8 GB 显存 / 32 GB 内存）调好的骨头人从零开训配置
│  └─ ability_test.py      # 固定种子贪心评测脚本
└─ seeds\           # 统一未见评测种子（如 RV1.json）
```

---

## 二、一次性环境准备（约 5–10 分钟）

### 1. 安装基础依赖
1. **.NET 9 SDK**（`9.0.x`，安装后在 PowerShell 运行 `dotnet --list-sdks` 应看到 `9.0.xxx`）
2. **Python 3.11 或 3.12**（官网安装包，避开 Windows Store 版）
3. 在 `D:\sts2-v114` 目录打开 PowerShell，建虚拟环境并安装 PyTorch CUDA + NumPy：

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

### 2. 复制本地正版 Steam《杀戮尖塔 2》（`v0.114.0`）文件并解包文本

请确认 Steam 里已切到 **`public-beta` 分支并更新到 `v0.114.0`**，把下面 `$game` 改成你本机 Steam 路径后执行：

```powershell
$game = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"
$dest = (Join-Path (Get-Location) "game_files\v0.114.0")

New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -Recurse -Force "$game\data_sts2_windows_x86_64\*" $dest
python -I tools\extract_pck.py "$game\SlayTheSpire2.pck" "$dest\res" localization/eng/ localization/zhs/ localization/completion.json
```

### 3. 单路编译 `Sts2Headless` 并跑通 `selfcheck`

```powershell
dotnet build engine\Sts2Headless\Sts2Headless.csproj -c Release -m:1 "-p:Sts2GameDir=$dest" -p:Sts2GameVersion=v0.114.0
dotnet engine\Sts2Headless\bin\Release\net9.0\Sts2Headless.dll selfcheck
```

预期结果：
- 编译 `0 个错误`（有 4 个可空引用警告属正常）；
- `selfcheck` 六个角色全部通过，骨头人（Necrobinder）0 异常、0 卡住。

---

## 三、先跑一遍原生 Windows 三档测速（约 2–3 分钟，摸清 4070 本机上限）

先在 PowerShell 激活虚拟环境并设置 `PYTHONPATH`：

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location) "python")
```

依次运行三档测速，把输出的 JSON 结果截图或贴给我：

```powershell
# 第 1 档：纯引擎异步采集（先 16 路，再 28 路）
python -u tools\win_bench.py --tier collect --envs 16 --seconds 20
python -u tools\win_bench.py --tier collect --envs 28 --seconds 20

# 第 2 档：采集 + 19.4M 骨头人网络 CUDA BF16 批量推理
python -u tools\win_bench.py --tier infer --envs 16 --seconds 20 --device cuda
python -u tools\win_bench.py --tier infer --envs 28 --seconds 20 --device cuda

# 第 3 档：完整多 Actor 端到端 PPO（含 50% 整局 + 50% 存档战斗练习 + Muon + EMA）
python -u tools\win_bench.py --tier ppo --actors 2 --envs-per-actor 14 --batch-steps 16384 --minibatch 1024 --grad-accum 2 --total-steps 65536 --device cuda
```

> **已在本机实测验证（原生 Windows · 0 异常 / 0 卡住）**：
> - **纯采集**（16 路）：`2,616 决策/秒`；
> - **完整端到端 PPO**（`2 Actors × 16 路 = 32 路`，`minibatch=1024`，BF16，含 50% 整局 + 50% 存档战斗练习 + Muon/EMA）：稳态 **`~940 有效训练决策/秒`**（实测显存峰值仅 **`3.95 GB`**，GPU 温度 `68°C`），相比旧版提速 **6.6×**！
> - **重要实测结论（Windows 必看）**：Windows WDDM 下 **`actors` 保持 `2` 即可，不要开到 `4` 个 CUDA Actor**（本机实测 4 个 CUDA Actor + 1 个 Learner 共 5 进程抢显卡调度会导致 `t_learn` 慢 5 倍，吞吐反而从 `940` 掉回 `509–732 dec/s`）。
> - **4070 移动版（8 GB 显存 / 32 GB 内存）**：按 `actors=2, envs_per_actor=14~16, minibatch=1024, grad_accum=4` 实测显存仅 **~4 GB**（占 8 GB 一半），非常稳健。

---

## 四、正式开跑：从零训练 `v0.114.0` 骨头人（A10，零真人数据）

三档测速没问题后，直接用预配好的 `tools\necro_win_4070.json` 启动（已严格对齐论文骨头人从零开训配方：Format 28、`Adam 3e-4 + Muon 6e-4`、`EMA 0.999`、`batch_steps=32768`、`50%` 整局 + `50%` 存档战斗练习、双时间尺度 SMDP GAE + 战后地图截断、完整局零 HP 塑形、未来读数头）：

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Join-Path (Get-Location) "python")

python -u python\train_defect.py --config tools\necro_win_4070.json
```

### 中断后续训 & 调参要点
- **断点续训**：直接重新运行同一条命令（`train.py` 会自动检测 `runs/alphadefect-necro-v114-4070/latest.pt` 续跑）。
- **白天轻度办公降载**：若觉得风扇太吵，把 `tools\necro_win_4070.json` 里的 `"envs_per_actor": 14` 改成 `10`（共 20 路）再重启即可。
- **平台期降噪阶梯（跑过几千万~几亿步后再用）**：按论文法则，当评测胜率停滞时，将 `lr` / `muon_lr` 减半（`3e-4/6e-4 → 1.5e-4/3e-4`）并把 `batch_steps` 翻倍（`32768 → 65536`）。

---

## 五、统一未见种子评测 & 回传文件

### 1. 用固定种子做贪心评测（约 2–3 分钟）
```powershell
python -u tools\ability_test.py `
    --character NECROBINDER `
    --ckpt runs\alphadefect-necro-v114-4070\latest.pt `
    --asc 10-10 --runs 256 --seeds-file seeds\RV1.json `
    --envs 12 --device cuda --out runs\eval-necro-v114-4070 `
    --req-extra '{"obs_format": 28, "loop_cut": true, "stall_turns": 20}'
```

### 2. 需要发回给我的最小产物（< 150 MB）
1. 三档测速的控制台 JSON 输出；
2. 训练目录里的 `log.jsonl`、`status.json`、`resource.jsonl`、`config.json` 与 `latest.pt`（或阶段备份 `backups/*.pt`）；
3. 评测输出目录 `runs\eval-necro-v114-4070\` 里的摘要 JSON。

---

## 六、常见问题速查

- **CUDA OOM（爆显存）**：把 `tools\necro_win_4070.json` 的 `"minibatch": 1024` 降为 `512`、`"grad_accum": 4` 改为 `8`（等效小批不变，显存立减近一半）。
- **卡在编译或报错缺 `GodotSharp.dll`**：确认第 2 步 `Copy-Item` 确实把 `data_sts2_windows_x86_64\` 下的全部 DLL 复制进了 `$dest`，且编译命令传入了正确的 `"-p:Sts2GameDir=$dest"`。
- **遇到单局引擎异常**：`VecEnv` 会自动记日志并重拉该槽位引擎进程，不影响整场训练继续。
