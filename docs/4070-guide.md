# 骨头人（Necrobinder）RL 协作训练指南 · RTX 4070 游戏本（8 GB 显存 / 32 GB 内存）

> **当前阶段（方案 C · 从 1 亿步 `v3050` 直接分叉续训 `+3,000 万步`）**
> 本指南面向 **Windows + RTX 4070 游戏本（8 GB 显存 / 32 GB 内存）**，运行 **`v0.114.0` 原生 Windows 高吞吐常驻引擎栈**（Ascension 10 · Necrobinder · 纯 RL 零真人数据、严格 No-SL）。
>
> **本次协作分工（无需从零重练，4070 仅需 ~12–16 小时！）**
> - **主机（5070 Ti）**：从 1 亿步锚点 `v3050` 续训 **Baseline 对照（`run_fraction=0.50`，`2×20=40` 路，每 Actor 精确 `10/20 = 50.0%`，`100M → 130M`）**；
> - **协作机（4070）**：从**同一份 280 MB `v3050 latest.pt`** 续训 **后期多整局分支（`run_fraction=0.75`，`2×12=24` 路，每 Actor 精确 `9/12 = 75.0%`，`100M → 130M`）**，数学超参（`batch=32768`、等效微批 `512×8 = 4096`、`lr/muon/ema`）、游戏版本（`v0.114.0`）、里程碑间隔（每 `150` 更新存一块：`v3150..v3900`）与主机 **100% 一致**。首轮按**相同新增决策步数（`+3,000 万步`）**比较策略强度，并分别记录耗时。

> **为什么不用 Mac M2？** 常驻无头引擎 `Sts2Headless` 依赖 `.NET 9 + 0Harmony / MonoMod` 运行时 IL 补丁，且模型使用 CUDA BF16 打包变长注意力；直接用 **4070 Windows 游戏本** 最稳、最快。

---

## 一、收到文件后的目录摆放（关键：放好 `latest.pt`！）

1. 在装有 Git 的 PowerShell 中克隆主仓库，恢复固定版本的 AlphaDefect 训练源码、Windows 补丁和 4070 配置。以下示例使用 `D:\sts2-bot`。

```powershell
git clone https://github.com/LukeLu1650/sls2bot.git D:\sts2-bot
cd D:\sts2-bot
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_4070.ps1
New-Item -ItemType Directory -Force runs\fork-v3050-rf75-necro-v1140-a10-4070 | Out-Null
```

   安装脚本将上游固定在 `d5b3c384b4d36c8239d36b2b99e1e3b44a1c3dcb`，训练器和实机 Mod 共用同一个源码目录。上游授权说明见克隆目录中的 `LICENSE` 和 `NOTICE.md`。

2. 首次训练前，从 [v3050 Release 下载 latest.pt](https://github.com/LukeLu1650/sls2bot/releases/download/v3050/latest.pt)，把 **`latest.pt`（279,635,251 字节）** 放进 `runs\fork-v3050-rf75-necro-v1140-a10-4070\latest.pt`。中断后继续训练时保留机器自己写出的新版断点。

```text
D:\sts2-bot\
├─ docs\4070-guide.md                                                     # 本文档
├─ scripts\
│  ├─ eval_v1140.py                                                       # 128 锁定 A10 种子评测脚本（EMA / raw）
│  ├─ verify_boss_solver.py                                               # 可选：Boss 错题搜寻与干净引擎复验器
│  └─ eval_handover_probe.py                                              # 冻结战斗策略接手探针
├─ runs\
│  └─ fork-v3050-rf75-necro-v1140-a10-4070\
│     └─ latest.pt                                                        # ← 把 280 MB 的 v3050 起点放这里！
└─ external\AlphaDefect-Public\game-v0.114.0\
   ├─ engine\Sts2Headless\                                                # C# .NET 9 常驻无头引擎（已含 Windows 适配）
   ├─ python\
   │  ├─ train_defect.py                                                  # 训练主入口
   │  ├─ sts2env\                                                         # Windows 异步多路引擎封装
   │  └─ sts2rl\                                                          # 训练器与网络（含 milestone_every 归档）
   └─ tools\
      ├─ extract_pck.py                                                   # 解包游戏本地化资源
      ├─ win_bench.py                                                     # 原生 Windows 测速脚本
      └─ necro_fork_v3050_rf75_win4070.json                               # 本次 4070 专用配置（rf=0.75, 100M→130M）
```

首次启动前核对 `latest.pt` 的完整 SHA-256，应为 `f629a703eea549a6b5f00f372a9514c45a42e0b0764b1e6eb96c28cce4828dcc`。
```powershell
(Get-FileHash "D:\sts2-bot\runs\fork-v3050-rf75-necro-v1140-a10-4070\latest.pt" -Algorithm SHA256).Hash.ToLower()
```

---

## 二、一次性环境准备（约 5–10 分钟）

### 1. 宿主机长跑防掉线检查（**主机实战血泪经验！**）
1. **电源模式**：切到游戏本控制中心的 **性能 / Performance 档**（**不要开狂暴超频 Turbo 档**，容易长时间满载触发 CPU 热保护重启），垫高笔记本底部保证进风，并在 Windows 电源选项中将“接通电源时睡眠”设为 **从不**。
2. **虚拟内存（页面文件）**：建议在剩余空间 > 100 GB 的固态盘上手动设为 **初始 `16384 MB` / 最大 `49152 MB`**（避免 24 路 `.NET` 引擎长跑十几小时撞 Windows 提交内存上限）。

### 2. 安装基础运行环境
1. **.NET 9 SDK**（版本 `9.0.x`，装完运行 `dotnet --list-sdks` 应看到 `9.0.xxx`）
2. **Python 3.11 或 3.12**（官网安装包，避开 Windows Store 版）
3. 在 `D:\sts2-bot` 打开 PowerShell，建虚拟环境并装 **PyTorch CUDA + NumPy**：

```powershell
cd D:\sts2-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
```

验证 GPU 正常（应输出 `True`、`NVIDIA GeForce RTX 4070 Laptop GPU`、`bf16= True`）：
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), 'bf16=', torch.cuda.is_bf16_supported())"
```

### 3. 核对本地正版《杀戮尖塔 2》版本（**严格同版本口径**）

两台机器必须使用**逐字节一致的 `v0.114.0` `sts2.dll`**。把 `$game` 改成你的 Steam 路径后执行：

```powershell
$game = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"
(Get-FileHash "$game\data_sts2_windows_x86_64\sts2.dll" -Algorithm SHA256).Hash.ToLower()
```

核对输出是否**逐字等于**：
```text
ddceb87905b6096deb7be975922e24a53992453675df26df545b1c46b1826c5d
```
> 如果哈希不一致，先停一下告诉我，别直接往下编译。

### 4. 隔离复制游戏程序集 & 解包本地化资源

```powershell
$dst = "D:\sts2-bot\artifacts\alphadefect\game\v0.114.0"
New-Item -ItemType Directory -Force -Path "$dst\res" | Out-Null
Copy-Item "$game\data_sts2_windows_x86_64\*" $dst -Force

python external\AlphaDefect-Public\game-v0.114.0\tools\extract_pck.py `
  "$game\SlayTheSpire2.pck" "$dst\res" localization/eng/ localization/zhs/ localization/completion.json
```
看到解包出 ~224 个本地化文件即可。

### 5. 编译 `v0.114.0` 常驻引擎并运行全角色自检（约 15 秒）

```powershell
$dst = (Resolve-Path "artifacts\alphadefect\game\v0.114.0").Path
$env:STS2_USER_DIR = (Join-Path (Get-Location) "artifacts\alphadefect\user")

dotnet build external\AlphaDefect-Public\game-v0.114.0\engine\Sts2Headless\Sts2Headless.csproj `
  -c Release -p:Sts2GameDir="$dst"

dotnet external\AlphaDefect-Public\game-v0.114.0\engine\Sts2Headless\bin\Release\net9.0\Sts2Headless.dll selfcheck
```

确认输出末尾包含 `兼容层登记 184 处`、`词表 2559 项 ... 不在词表的 0 个`、全 6 角色 `异常 0，卡住 0` 与 **`自检通过`**。

---

## 三、启动 `v3050 → rf=0.75` 续训（仅需 ~12–16 小时，随时可 `Ctrl+C` 续训）

本配置已针对 **4070 8 GB 显存** 与 **精确 `75.0%` 整局占比** 预配 **`actors=2, envs_per_actor=12, minibatch=512, grad_accum=8`**（每 Actor 精确 `9` 路整局 + `3` 路专项 = `75.0%`，避免 `14` 路下 `round(10.5)=10` 落到 `71.4%`；等效微批 `4096` 与主机 `1024×4` 数学等价，显存安全压在 ~5 GB 内防 OOM）：

```powershell
cd D:\sts2-bot
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path "external\AlphaDefect-Public\game-v0.114.0\python").Path
$env:STS2_USER_DIR = (Join-Path (Get-Location) "artifacts\alphadefect\user")

python external\AlphaDefect-Public\game-v0.114.0\python\train_defect.py `
  --config external\AlphaDefect-Public\game-v0.114.0\tools\necro_fork_v3050_rf75_win4070.json
```

**启动首屏必须看到以下三行（确认确实是从 `v3050` 续训，而不是从零重练！）**：
- `从检查点继续：版本 3050 ... 已用 100023231 步`
- `滑动平均权重：衰减 0.999，接着检查点里的`
- `Muon：... 动量接着检查点里的`

**挂机与进度说明**：
- 总目标 `130,000,000` 步（即从 `v3050` 往前再跑 **`+3,000 万步`**，约 `~915` 次更新到 `v3965` 自动停止）；
- 每 `25` 次更新（约 15–20 分钟）自动保存 `latest.pt`，随时 `Ctrl+C` 中断，重跑**同一条命令**自动无缝续训；
- 每 `150` 次更新自动保存轻量里程碑（`milestones/v003150.pt` ～ `v003900.pt`，共 6 块）。

**另开一个 PowerShell 实时监控（务必带 `-Encoding UTF8`）**：
```powershell
Get-Content "D:\sts2-bot\runs\fork-v3050-rf75-necro-v1140-a10-4070\log.jsonl" -Encoding UTF8 -Tail 5 -Wait | ForEach-Object {
  $r = $_ | ConvertFrom-Json
  "v$($r.version) | steps=$($r.total_steps) | sps=$($r.sps) | drop=$($r.dropped) stuck=$($r.stuck) | floor=$([math]::Round($r.run_floor_mean,2))(max $($r.run_floor_max)) | boss=$($r.boss_wins[0])/$($r.boss_n[0]),$($r.boss_wins[1])/$($r.boss_n[1]),$($r.boss_wins[2])/$($r.boss_n[2]) | act2=$($r.run_reach_act2) act3=$($r.run_reach_act3) win=$($r.run_win_rate) | kl=$($r.approx_kl)"
}
```

---

## 四、跑完后如何自测 & 发回哪些文件

### 1. 跑一遍统一 128 锁定 A10 种子 `EMA` 评测（仅约 3–5 分钟！）
```powershell
python -u scripts\eval_v1140.py `
  --ckpts runs/fork-v3050-rf75-necro-v1140-a10-4070/latest.pt `
  --out runs/eval-fork-v3050-rf75-4070 `
  --weights ema --runs 128 --seed-prefix A10EVAL --envs 12 --device cuda `
  --export-all-bosses runs/eval-fork-v3050-rf75-4070/final_ema_all_bosses.jsonl
```

### 2. 发回的核心文件
1. `runs/fork-v3050-rf75-necro-v1140-a10-4070/config.json`
2. `runs/fork-v3050-rf75-necro-v1140-a10-4070/log.jsonl`
3. `runs/fork-v3050-rf75-necro-v1140-a10-4070/latest.pt`（最终断点，含 `ema`）
4. `runs/fork-v3050-rf75-necro-v1140-a10-4070/milestones/`（`v3150..v3900` 里程碑，至少发 `v3450 / v3900`）
5. `runs/eval-fork-v3050-rf75-4070/summary_ema.json` 与 `final_ema_all_bosses.jsonl`（如已跑评测）

---

## 五、4070 移动版（8 GB 显存）排错速查

| 现象 | 原因 | 解决方法 |
| --- | --- | --- |
| **启动日志显示 `版本 0` 而不是 `版本 3050`** | `latest.pt` 路径没放对 | 立刻 `Ctrl+C`，确认 `runs\fork-v3050-rf75-necro-v1140-a10-4070\latest.pt` 存在且大小约 `280 MB` 后重跑 |
| **CUDA Out of Memory** | 浏览器/游戏占显存 | 关掉占显存软件；本配置已是 `minibatch=512, grad_accum=8`，若仍紧可将 `minibatch` 改 `256`、`grad_accum` 改 `16`（等效微批仍为 `4096`） |
| **速度腰斩** | 拔电降频、开省电模式，或把 `actors` 改成了 `4` | 保持插电 + `Performance` 档；确认 `actors = 2` |
| **跑十小时报 `系统资源不足`** | Windows 自动分页文件上限太小 | 按第二节第 1 条把虚拟内存最大设为 `49152 MB` 并重启，重跑原命令无缝续训 |
| **报 `ConvertFrom-Json` 解析错** | 监控命令漏了 `-Encoding UTF8` | 加上 `-Encoding UTF8` 即可（训练本身完全正常） |
