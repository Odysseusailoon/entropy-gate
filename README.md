# Entropy gate：答案分布估计的端到端净收益实验

本项目研究：**在达到相同的原模型答案分布估计误差时，entropy 引导的采样分配能否减少总 FLOPs 和 GPU 占用秒数？** 四组方法是 IID、共享前缀＋uniform、共享前缀＋entropy、Arithmetic Sampling。主要归因比较是 **entropy vs uniform**。

**正式实验使用 Qwen3-8B / 五张 H200。Qwen3-1.7B / A100 只用于工程调试。** 调试耗时、答案、reference 和标定计划都不能作为正式实验结果。

- [给接手 agent 的 RFC](rfcs/entropy-gate-net-benefit.md)：研究问题、设计依据、估计器、统计判据和完整 delivery 合同。
- [预注册协议](PREREGISTRATION.md)：正式实验的冻结规则。
- 本 README：代码入口、可执行步骤、恢复方法和产物位置。

本页是操作手册，不是实时运行状态。某次实验是否完成，应核对其 manifest、队列、seals、结果网格和交付报告。

## 1. 代码、配置与实验脚本

```text
entropy-gate-samole/
├── egs/                    配置记录、模型加载、数据、解析、质量检查、文件读写
│   └── net/                v3：采样、缓存、计量、调度、reference、统计分析
├── configs/                正式、合成演练与调试配置
├── scripts/                离线演练与五卡启动器
├── tests/                  共享工具、采样数值、真实小模型和完整流程测试
├── runs/                   每次运行的冻结输入、原始样本、成本、日志和分析
├── rfcs/entropy-gate-net-benefit.md
├── README.md
└── PREREGISTRATION.md
```

| 入口 / 文件 | 用途 |
|---|---|
| `python -m egs.net` | v3 主入口；`python -m egs` 和安装后的 `egs` 等价 |
| [experiment.py](egs/net/experiment.py) | 四种估计器、分支权重、完整运行计时和 FLOP replay |
| [engine.py](egs/net/engine.py) | 单次 prompt prefill、前缀 KV 复用、逐 token 生成 |
| [arithmetic.py](egs/net/arithmetic.py) / [flops.py](egs/net/flops.py) | token 采样与执行算子计数 |
| [pipeline.py](egs/net/pipeline.py) / [queue.py](egs/net/queue.py) | 开发标定、配对块、worker、checkpoint 和恢复 |
| [reference.py](egs/net/reference.py) / [analysis.py](egs/net/analysis.py) | 独立 reference、误差、净成本、bootstrap、图表和决策 |
| [net-benefit.yaml](configs/net-benefit.yaml) | **正式 8B/H200** 的基础配置 |
| [net-demo.yaml](configs/net-demo.yaml) | CPU 合成演练，不产生 GSM8K/GPU 结论 |
| [net-qwen17b-a100.yaml](configs/net-qwen17b-a100.yaml) | **仅调试**的 1.7B/A100 覆盖文件，用于准备开发题 prompt |
| [run_net_demo.sh](scripts/run_net_demo.sh) | 一键执行合成 freeze、preflight、标定、测试/reference 和分析 |
| [launch_net_five_gpu.sh](scripts/launch_net_five_gpu.sh) | 五张同型号空闲 GPU 的共享队列；接受 `calibration`、`test`、`reference-N` |
| [launch_net_h200.sh](scripts/launch_net_h200.sh) | 转发到五卡启动器，实际硬件依据冻结配置检查 |
| [debug.py](egs/net/debug.py) | 单卡、开发题兼容性探针；输出 `debug_only=true` |

## 2. 安装与本地验证

以下命令均在项目根目录执行。需要 Python 3.11+；真实模型测试和 GPU 实验还需要 Transformers、Datasets 等依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[gpu,analysis,test]'
python -m pip check
python -m pytest -q
```

本地合成演练也使用 PyTorch，但不需要下载模型或访问 CUDA：

```bash
bash scripts/run_net_demo.sh
```

输出目录为 `runs/net-demo/`。脚本包含四种采样、独立 reference、开发标定、bootstrap 和两张图；GPU 时间为空，GPU 图会注明未测量。

freeze 不覆盖已有运行。若 `runs/net-demo/` 已冻结，不要删除结果来强行重跑；创建新的 run-name YAML 覆盖文件，按第 4 节的逐阶段命令执行，使用相应的新 run 路径。正式运行同理。

## 3. 正式配置：Qwen3-8B / H200

正式命令只加载 `configs/net-benefit.yaml`，不叠加 1.7B/A100 文件。

| 项目 | 当前正式值 |
|---|---|
| 模型 / 模式 | Qwen/Qwen3-8B，non-thinking |
| 设备 / 数值 | 五张 H200；bfloat16；SDPA；最大 batch=8 |
| 采样 | temperature=1、top_p=1、top_k=-1、min_p=0，全词表 |
| Demonstrations / 开发题 / 测试题 | GSM8K train 0–7 / train 8–17 / test 1000–1049 |
| IID 预算 / 重复 | 16、32、64；每题每配置 10 次测试运行 |
| 开发标定 | 7 个候选数量，每题每候选 3 次；仅使用成本选数量 |
| 前缀分叉 | R=8；生成到第 64 token；beta=1；n_min=1 |
| 长度 guard | 1024；正式允许截断率为 0 |
| Reference | 独立 IID，1024/题，必要时扩到 2048/4096 |
| 指标 / 阈值 | JS bits、平方 L2；epsilon=0.01、0.02、0.05 |
| Bootstrap | 2000 次，按题目与完整配对运行重采样 |

正式预检若发生截断，需在测试前修订 guard、创建新 run 并使用匹配 reference。调试配置的 4096-token guard 不代表正式上限已经更改。任何额外曾用于调参的题目都要登记在 `study.tuning_question_ids`，不能与测试集相交。

## 4. 正式实验执行顺序

### 4.1 确定环境与空闲卡

先激活实验环境，再选择实际空闲的五张 H200。下面的卡号是示例，启动时会重新检查占用情况。

```bash
export EGS_PYTHON="$(command -v python)"
export EGS_GPU_IDS=0,1,2,3,4
export EGS_RUN=runs/net-benefit
python -m egs.net.devices --devices "$EGS_GPU_IDS" --required-name H200
```

启动器拒绝已占用设备，只管理自己创建的 worker。不得通过终止其他人的 GPU 进程腾卡。

### 4.2 Freeze 与开发题预检

```bash
python -m egs.net freeze --config configs/net-benefit.yaml
EGS_PREFLIGHT_GPU="${EGS_GPU_IDS%%,*}"
CUDA_VISIBLE_DEVICES="$EGS_PREFLIGHT_GPU" python -m egs.net preflight --run "$EGS_RUN"
```

freeze 保存精确题目、prompt IDs、模型/数据 SHA、代码与依赖指纹。preflight 只生成开发题，产出至少 50 条 rollout 和审核表，并检查：

- invalid <5%，truncation=0，有采样多样性。
- 同 seed、同 batch 配置的计时运行与 FLOP replay 的 token、权重、分配一致。
- 每种方法每次独立运行只有一次 prompt prefill，遇到未覆盖浮点算子停止。
- 相同固定前缀、batch=1 与配置 batch 的 next-token 概率最大行 TV≤0.01。

BF16 跨 batch 的完整路径相同与否只作诊断；局部 TV 检查不是完整答案分布误差保证。

### 4.3 人工审核

由人检查 `$EGS_RUN/manual_review.csv`，填写 `human_answer` 与 `approved=yes`，保留原题目、文本和 parser 输出。Agent 不得自动填批准字段来替代人工审核。

```bash
python -m egs.net audit --run "$EGS_RUN"
```

只有真实审核通过且记录与 preflight 匹配，才进入后续实验。

### 4.4 可选：导入兼容 reference

此步骤必须在首次初始化队列之前执行。不导入则测试阶段自动生成独立 reference。

```bash
python -m egs.net import-reference --run "$EGS_RUN" --source /path/to/independent-8b-reference
```

仅接受已封存的 v3 reference；模型/tokenizer、采样/EOS/长度、parser、dtype/attention、数值环境、每题精确 prompt、原始 hash 和 seed 独立性都必须匹配。1.7B/A100 的 reference 不能导入正式 8B 实验。

### 4.5 开发成本标定

```bash
bash scripts/launch_net_five_gpu.sh "$EGS_RUN" calibration
python -m egs.net status --run "$EGS_RUN" --phase calibration
```

脚本先初始化队列，五个 worker 共同执行，再自动运行 `calibrate`。完成后检查 `calibration_plan.json` 和 `calibration_seal.json`。

开发标定选择各方法最接近 IID FLOP 锚点的采样数量，包含预注册准备成本摊销；不使用测试/reference 答案调参。最大相对成本偏差为 35%。候选范围不足时停止，必须在新协议中处理。

### 4.6 配对测试、独立 reference 与分析

```bash
bash scripts/launch_net_five_gpu.sh "$EGS_RUN" test
python -m egs.net status --run "$EGS_RUN" --phase test
```

每个 `(题目, 预算, 重复编号)` 是完整配对块，四种方法在同一 GPU 上随机排序执行。Reference 的 128-rollout 小块穿插在同一队列中。

脚本结束时自动运行 `analyze`。完成网格应有 **840 个开发结果、6000 个测试结果**；未导入时初始 reference 为 **51,200 条 rollout**。这些数量不含 warmup 和计数重放。

若测试完成而 reference 尚不稳定，分析停止；按冻结的倍增规则补充 reference，不重跑测试采样：

```bash
python -m egs.net extend-reference --run "$EGS_RUN" --samples 2048
bash scripts/launch_net_five_gpu.sh "$EGS_RUN" reference-2048
python -m egs.net analyze --run "$EGS_RUN"
```

仍不稳定时对应改为 `4096` 和 `reference-4096`。达到冻结最大值仍不稳定则交付 inconclusive。已接受的 reference 不再修改。

### 4.7 状态、恢复与重新分析

```bash
python -m egs.net status --run "$EGS_RUN" --phase test
python -m egs.net recover --run "$EGS_RUN" --phase test
bash scripts/launch_net_five_gpu.sh "$EGS_RUN" test
python -m egs.net analyze --run "$EGS_RUN"
```

只在确认本 run 的 worker 崩溃后使用恢复步骤；正常完成时不需要再次启动 worker。`recover` 只回收同机已退出进程留下的 `running` 任务，并保留原 worker/GPU 归属。它不会重置 OOM、漏计等真实 `failed` 任务。不能删失败行、跳过 seed 或更换卡后继续一个未完成配对块。

如需手工调度，`initialize` 只初始化任务，`worker` 才执行；calibration 完成后要显式 `calibrate`，test 完成后要显式 `analyze`。五卡脚本已经串好这些命令。

## 5. 调试专用：Qwen3-1.7B / A100

调试用于排查模型加载、KV cache、算子覆盖、解析和计时。仅使用开发 train 题，不运行本 README 的正式 calibration/test 阶段，不从调试结果推断 8B/H200 speedup。

准备独立的调试 prompt：

```bash
python -m egs.net freeze \
  --config configs/net-benefit.yaml \
  --config configs/net-qwen17b-a100.yaml
```

该覆盖文件当前输出到 `runs/net-qwen17b-a100-v2/`。如该目录已经冻结且当前代码有变化，用新的 run-name 覆盖文件准备 prompt，不修改旧 manifest。然后选一张空闲 A100 执行探针，`--out` 必须是新目录：

```bash
python -m egs.net.debug \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --questions runs/net-qwen17b-a100-v2/development_questions.jsonl \
  --fingerprint runs/net-qwen17b-a100-v2/development_fingerprint.json \
  --gpu 0 --samples 16 --batch-size 8 --max-tokens 4096 \
  --out runs/debug-qwen17b-a100-001
```

`--gpu 0` 是示例；探针会检查该卡空闲。`debug.py` 当前使用所给开发题列表的第一题，输出 `debug_session.json`、每种方法的 JSON、`summary.json` 和 `debug_complete.json`，均应作为工程证据交付。

覆盖配置本身不具有“禁止正式运行”的程序权限隔离；不能因为通用 CLI 能接受它，就把 A100 的完整队列运行称为正式实验。正式交付的验收身份仍然是 8B/H200。

## 6. 离线环境与失败定位

`freeze --offline` 只接受已经缓存的固定 model/tokenizer/dataset SHA。正式基础配置的 revision 默认是 `main`，因此正式离线运行需要先准备一个含 **8B** 精确缓存 SHA 的 YAML 覆盖文件；不能用 1.7B revision 代替。缺少缓存或 Datasets 回退到其他 revision 时应停止。

可按本机情况为进程设置 `HF_HUB_CACHE` 和 `HF_DATASETS_CACHE`。`freeze --offline` 在该命令内设置离线模式；后续命令需要离线时，也要设置 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`。服务器若需要特定 `LD_LIBRARY_PATH` 才能加载 PyArrow 等依赖，应记录实际路径到交付环境说明，避免修改公共环境。

| 失败点 | 先检查的证据 | 后续处理 |
|---|---|---|
| 无空闲卡 / 型号不匹配 | GPU admission 输出 | 选择空闲 H200，等待资源 |
| 冻结身份变化 | manifest 的代码、依赖与协议指纹 | 使用匹配源码/环境；修改规则则建新 run |
| 预检或人工审核未通过 | `preflight.json`、rollouts、审核表 | 修复原因，按新协议重新预检；不自批审核 |
| 截断、解析失败过多 | 原始 token、finish_reason、parser 输出 | 保留失败证据；测试前修订并重新 freeze |
| 候选成本无法覆盖预算 | 开发结果与标定错误 | 新协议扩展候选；不看测试误差补点 |
| 算子未覆盖 / replay 不一致 | cost 报告、worker 日志 | 修复和验证后使用新 run；不记零或放宽检查掩盖问题 |
| Reference 未收敛 | `reference_checks.json`、收敛曲线 | 只按已冻结的扩充规则继续 |

## 7. 必须交付什么

所有正式交付必须来自同一 8B/H200 协议与封存计划。详细字段和验收规则见 [RFC 的 Delivery 合同](rfcs/entropy-gate-net-benefit.md#8-delivery-合同)。

| 交付 | 路径，相对于 `$EGS_RUN` | 谁生成 |
|---|---|---|
| 冻结输入与环境身份 | `config.yaml`、`manifest.json`、两组 questions/fingerprint 文件 | freeze |
| 正确性与人工审核证据 | `preflight.json`、`preflight_rollouts.jsonl`、`manual_review.csv`、`manual_audit.json` | preflight＋人工＋audit |
| 开发计划与全部原始结果 | `calibration_plan.json`、`calibration_seal.json`、`results/calibration/`、`results/test/` | worker / calibrate |
| 成本与调度 | 结果中的 `cost`、`ledger`，`worker_overhead/`、`logs/`、一致的 `queue.sqlite` 备份 | worker / queue；执行者归档 |
| 独立 reference | `reference_chunks/` 或 `reference_import/`，`reference_distributions.jsonl`、`reference_convergence.csv`、`reference_checks.json`、`reference_complete.json` | reference / analyze |
| 统计表 | `net_analysis/per_run_results.csv`、`observed_cost_error_curves.csv`、`paired_comparisons.csv`、`speedups.csv` | analyze |
| 两张核心图 | `net_analysis/error_vs_total_flops.png/.pdf`、`error_vs_gpu_seconds.png/.pdf` | analyze |
| 机器可读结论 | `net_analysis/decision.json` | analyze |
| 解释与接手说明 | `delivery/report.md`、`delivery/validation.md`、`delivery/handoff.md`、`delivery/environment.md` | **执行 agent 补写，CLI 不自动生成** |
| 可复现归档 | `delivery/source/`、`delivery/SHA256SUMS` | **执行 agent 补齐** |

报告必须同时给出：entropy 对 uniform、IID、Arithmetic、最佳 baseline 的各 epsilon 结果；JS/L2；总 FLOPs/GPU 秒数；inference-only 与准备成本摊销；CI 和未达到阈值的状态；reference/数值/实现限制。

主图的成本是跨题目求和、跨独立运行取平均的 cohort 成本；误差是题目和运行的宏平均。不能把 `per_run_results.csv` 的单题成本直接当作整组图的横轴。

## 8. 如何判断交付完成

- [ ] 正式模型为 Qwen3-8B，设备为 H200；debug/synthetic 数据未混入。
- [ ] 冻结文件与源码、依赖一致；预检和实际人工审核通过。
- [ ] 开发计划封存；开发/测试结果网格完整；四方法在同块同 GPU 运行。
- [ ] FLOP 覆盖和同配置 replay 通过，准备成本与计数重放记录齐全。
- [ ] 独立 reference 通过质量和收敛检查，原始样本与 seal 可追溯。
- [ ] 两张图、四张统计表、`decision.json` 和 agent 补写的交付说明齐全。
- [ ] 正结果、负结果、阈值未达到和 inconclusive 均如实报告。

当前实现只在观察配置上比较成本，不插值或外推；bootstrap 单位是题目和完整配对运行。CI 以固定 reference 和标定计划为条件。只赢 IID 不足以证明 entropy 的贡献；只有 FLOPs 更低也不足以证明实际 GPU 时间收益。

交付完成不要求研究假设成立。若仍有失败、缺卡、缺人工审核、未完成网格或不稳定 reference，交付阶段报告与下一步命令，并明确剩余项。
