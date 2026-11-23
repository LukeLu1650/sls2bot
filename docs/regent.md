# 第一次实验 · 摄政王

**历史复现，可选。** 本机的实际入口已切换到 [Windows v0.113.1 骨头人基线](windows-baseline.md)。下文供另有 Linux v0.107.1 安装的研究者复现论文环境。

**实验问题**：这台笔记本能否在官方游戏逻辑上采样、完成梯度更新、导出策略，然后在相同种子上自动打完？使用单人标准模式 A10 和上游原始摄政王网络。4 个种子、2 次更新足以检查流程；小样本里的胜负变化没有统计解释力。上游在 2,560 个留出种子上的摄政王 A10 胜率为 12.5%，一次四局全败也可能是正常结果。[发布页与评测口径](https://github.com/HalfVulpes/sts2ai/releases/tag/v1.0)

## 1. 核对运行环境与游戏版本

在本仓库目录的 PowerShell 运行 `wsl -d Ubuntu-24.04`。下文命令使用同一个 Bash 会话，从本仓库根目录开始。首次准备可以优先核验游戏文件，再安装训练依赖。

```bash
ROOT="$PWD"
UP="$ROOT/external/sts2ai"
ASSETS="$ROOT/artifacts/v1.0"
if [ ! -d "$UP/.git" ]; then
  git clone --branch v1.0 --depth 1 https://github.com/HalfVulpes/sts2ai.git "$UP"
fi
git -C "$UP" rev-parse HEAD
python3 --version
cc --version
nvidia-smi
bash "$ROOT/scripts/regent-smoke.sh" --dry-run
```

预期提交为 `d667489560cc46154638e1208ff032ed96707aa3`。预演只打印最终训练参数，无需游戏或 GPU 依赖。实际导入需准备自己拥有的 **Linux x86-64 Steam v0.107.1** 安装副本，内含 `release_info.json`、`SlayTheSpire2.pck`、`data_sts2_linuxbsd_x86_64/`。当前查到的 Windows v0.113.1 可留作未来模式验证。旧版本可用性取决于个人备份或 Steam 提供的构建；拿不到匹配副本时，转到版本迁移流程，另建基线。[上游导入器与校验清单](https://github.com/HalfVulpes/sts2ai/tree/d667489560cc46154638e1208ff032ed96707aa3/tools)

```bash
read -r -p 'Linux v0.107.1 安装目录：' GAME_DIR
python3 "$UP/tools/import_game.py" --game-dir "$GAME_DIR" --verify-only
```

## 2. 建立宿主并准备权重

以下步骤在版本和 11 个 DLL 哈希全部匹配后执行。Ubuntu 内需要 .NET **9 SDK**，用 `dotnet --list-sdks` 确认有 9.x。训练使用虚拟环境和 CUDA 版 PyTorch。下例选上游锁定的 torch 2.11.0，cu128 需要兼容的宿主驱动；最后的小矩阵乘法检查 GPU 计算是否真正可用。[.NET 的 Ubuntu 安装说明](https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu) · [PyTorch 官方 wheel](https://pytorch.org/get-started/previous-versions/#v2110) · [上游 Python 依赖](https://github.com/HalfVulpes/sts2ai/blob/d667489560cc46154638e1208ff032ed96707aa3/train/requirements.txt)

```bash
cd "$UP"
dotnet --list-sdks
python3 -m venv "$HOME/.venvs/sts2ai-v1"
source "$HOME/.venvs/sts2ai-v1/bin/activate"
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r train/requirements.txt
python -c 'import torch; x=torch.ones((8,8),device="cuda"); print(torch.__version__, torch.cuda.get_device_name(), (x@x)[0,0].item())'
python3 tools/import_game.py --game-dir "$GAME_DIR"
dotnet build sim/Sts2Sim.csproj -c Release
dotnet sim/bin/Release/net9.0/Sts2Sim.dll vocab --out gamebin/v0.107.1/vocab.json --quiet
python3 -c 'import json; assert json.load(open("gamebin/v0.107.1/vocab.json"))["size"] == 2650'
```

导入命令用于首次安装，已有同版本导入时可直接核对 `IMPORT_MANIFEST.json` 并继续构建。下面只取第一轮需要的发布物。先核对下载文件的 SHA-256，再解包。用于评测的 `.bin` 和用于继续训练的 `.pt` 格式不同，两者都要准备。[v1.0 文件列表与校验值](https://github.com/HalfVulpes/sts2ai/releases/tag/v1.0)

```bash
mkdir -p "$ASSETS"
cd "$ASSETS"
release=https://github.com/HalfVulpes/sts2ai/releases/download/v1.0
for file in SHA256SUMS regent.bin sts2ai-weights-released-five.tar.gz; do
  curl -fL --retry 2 "$release/$file" -o "$file" || exit 1
done
for file in regent.bin sts2ai-weights-released-five.tar.gz; do
  awk -v name="$file" '$2 == name || $2 == "*" name {print; found=1} END {exit !found}' SHA256SUMS > "$file.sha256" &&
    sha256sum -c "$file.sha256" || exit 1
done
tar -xzf sts2ai-weights-released-five.tar.gz
test -s sts2ai-weights-released-five/training_inputs/regent.pt
```

## 3. 运行一次并读结果

```bash
cd "$ROOT"
bash scripts/regent-smoke.sh
```

脚本创建独立的 `runs/regent-standard-A10-…/`，依次在 4 个固定种子上评测公开策略、执行 2 次各 1,024 行的 PPO 更新，再将新 checkpoint 的 EMA 权重导出并复测。4 个环境并发采样，minibatch 为 64。训练器按上游十位种子协议将这 4 个评测种子排除。上游评测器会按输出目录复用旧结果，每次试验使用新目录可以避免串用模型成绩。[训练入口](https://github.com/HalfVulpes/sts2ai/blob/d667489560cc46154638e1208ff032ed96707aa3/scripts/train.py) · [评测入口](https://github.com/HalfVulpes/sts2ai/blob/d667489560cc46154638e1208ff032ed96707aa3/scripts/evaluate.py)

| 文件 | 验收方法 |
| --- | --- |
| `inputs.sha256`、`upstream-commit.txt`、`reserved.json` | 保存输入指纹和排除种子，可以在同一构建重跑 |
| `before/summary.json`、`after/summary.json` | 检查两边各 4 个相同种子、零引擎错误、每局均有步数和到达楼层。两次导出模型哈希应不同 |
| `train.log`、`train/latest.pt` | 检查两次更新完成、损失有限、引擎错误为零，checkpoint 能重新加载并在同种子完成整局 |
| 训练吞吐与显存 | 记录每轮新采样决策数、采样耗时、更新耗时和显存峰值。另算端到端速度＝新采样决策数 ÷ 训练墙钟秒数，排除评测时间；回放前缀步数单列 |

遇到显存不足，先减小 minibatch 和 token budget；遇到系统内存或共享内存不足，先减并行环境数并检查 `df -h /dev/shm`。每次只改一项并新建输出目录。需要估算一天能练多少时，加长到进入稳态，分别测 2、4、8 个环境，用端到端速度乘以 86,400。首次短训通过后，按骨头人流程开始正式对照。
