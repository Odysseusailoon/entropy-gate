# RFC-EGS-001：Entropy gate 能否降低答案分布估计的端到端成本？

- 日期：2026-09-07。
- 对应实现：`egs/net/`，协议标识 `net-benefit-v3`。
- 读者：接手实现、实验执行、统计分析或独立复核的 agent。
- 状态：已有实现的研究问题、设计依据与交付合同；不是实验结果报告。
- 正式实验：**Qwen3-8B / 五张 H200**。**Qwen3-1.7B / A100 仅用于工程调试**。

执行命令见 [README](../README.md)，冻结的统计规则见 [PREREGISTRATION](../PREREGISTRATION.md)。某次运行的实际配置和身份由其 `config.yaml`、`manifest.json`、`calibration_plan.json` 及对应 seal 确定。发现文档、代码和冻结记录不一致时，先记录差异；不能通过修改已有 manifest 或删掉失败结果来使它们一致。

## 1. 我们要解决的问题

给定一道题 x，固定语言模型按原始采样分布生成完整推理并输出数值答案 Y。我们需要估计整个答案分布 P(Y|x)，包括不同错误答案和解析失败的概率质量。

直接方法是反复独立生成完整 rollout。它容易解释，但每个样本都要生成完整推理。候选改进是：先独立生成若干前缀，复用其 KV cache，再把更多后续采样分配给 token entropy 较高的前缀。

这个想法存在三个独立问题：

1. **统计问题**：前缀中的 token 不确定性，是否能帮助降低最终答案分布的估计误差？
2. **系统问题**：前缀复用节省的计算，是否足以覆盖 entropy、分支调度、缓存复制和采样的开销？
3. **归因问题**：观察到的收益究竟来自共享前缀，还是来自 entropy 引导的分配？

因此主问题是：**在相同的答案分布估计误差阈值下，entropy 方法是否降低包含方法特有准备开销的总 FLOPs 和 GPU 占用秒数？** 最重要的比较是 entropy 对 uniform。

“找到更常见的答案”“提高一次回答的准确率”和“更少生成 token”都不能单独回答这个问题。Gold answer 用于数据与解析检查；本轮主要终点是与独立 reference 的分布差异。

## 2. 研究边界与可证伪假设

| 假设 | 对应证据 | 可能的否定结果 |
|---|---|---|
| 共享前缀有计算收益 | uniform 与 IID 的成本、误差比较 | 相关性带来的误差超过缓存收益 |
| entropy 分配有额外价值 | entropy 与 uniform 的配对比较 | entropy 未改善误差，或收益被 gate 开销抵消 |
| 净收益能落到实际时间 | 同时报告 FLOPs 和 GPU 秒数 | FLOPs 更低，但 GPU 占用时间没有改善 |
| 优势不只针对最弱对照 | 同时与 Arithmetic 和最佳观测 baseline 比较 | 赢过 IID，但其他 baseline 更好 |

本轮固定模型权重，不训练模型，不修改 token 分布，不进行 beam search、答案去重、奖励重加权或基于完成时间的筛选。

调试结果只能支持“实现能否运行、数值检查是否通过”等工程结论。1.7B/A100 的参考分布、标定计划和耗时都不能替代 8B/H200 的正式证据。

## 3. 四种方法与公平性

| 方法 | 生成方式 | 最终权重 | 回答的问题 |
|---|---|---|---|
| `iid` | N 条独立完整 rollout | 每条 1/N | 常规 Monte Carlo 基线 |
| `uniform` | R 个独立前缀，近似均匀分配 N 个 children | 父前缀质量固定，每条 child 为 1/(R k_r) | 前缀共享本身的收益 |
| `entropy` | 与 uniform 同一生成引擎，按前缀 entropy 分配 children | 同上 | entropy 分配的增量价值 |
| `arithmetic` | 共享随机平移的 code points，沿 token CDF 递归解码 | 每条 1/N；同一运行内样本相关 | 另一种分布保持采样基线 |

每种方法的每次独立运行都付费计算一次 prompt prefill，并在该次运行内复用。不是四种方法共同免费使用一次 prefill。各方法使用同一 dtype、attention、最大 batch 和缓存管理规则；不跨独立运行共享 prompt cache。

uniform 和 entropy 在同一配对块内共享前缀及 child 的随机流命名规则；四种方法在同一 GPU 上按随机顺序执行。每张 GPU 都运行所有方法，不能固定“一张卡一种算法”。

Arithmetic Sampling 的研究来源是 [Vilnis et al., ICML 2023](https://proceedings.mlr.press/v202/vilnis23a.html)。本仓库采用共享平移、CDF 区间映射及逐位置词表置换；原始参考实现见[作者代码](https://github.com/google-research/google-research/blob/master/arithmetic_sampling/t5x/decoding.py)。本实现使用 PyTorch 与精确有理数 residual，不将其速度描述为作者 T5X 实现的复现。

## 4. Entropy gate 的估计器

设 H_r 是第 r 个独立前缀，R=8。默认在生成第 64 个 token 后分叉；若此前到 EOS，则直接保留该前缀的质量。

对前缀中已生成位置计算完整词表 entropy，以 bits 为单位，再取均值 h_r。词表概率包含特殊 token 的质量；抽到特殊 token 的位置从 gate 均值中排除。

令总 child 数为 N，每个前缀至少分配 n_min 条。剩余预算按 softmax(beta × h_r) 分配，使用 Hamilton rounding 得到整数 k_r，使总数严格等于 N。当前 beta=1、n_min=1。uniform 在同一分配函数中使用相等分数。

最终估计器为：

\[
\widehat P(y\mid x)=\frac{1}{R}\sum_{r=1}^{R}\frac{1}{k_r}\sum_{j=1}^{k_r}\mathbf{1}\{Y_{rj}=y\}.
\]

每个父前缀贡献 1/R 的质量；分支多不意味着它在最终分布里权重更高。所有叶子等权会把被过度采样的前缀放大，改变估计目标。

在前缀来自原模型、后续路径按原条件分布独立生成、k_r 只依赖已观察前缀的理想采样条件下，先对 continuation 取条件期望，再对前缀取期望，可恢复原模型边缘答案概率。实现中的浮点误差仍需单独检查，不能由这个推导排除。

需要保留的限制：

- token entropy 不等于最终答案的条件方差；它只是待检验的分配信号。
- R 固定时，增加 children 不能消除前缀抽样产生的全部方差；可能出现误差下限。
- 叶子共享前缀，不能当作独立样本计算置信区间。
- 生成到 EOS 的前缀保留质量；达到长度 guard 的样本进入 `__TRUNCATED__`，不伪装成正常答案。
- 分叉直接复用已计算的 K/V，但可写 continuation cache 需要复制/拼接。这些时间都必须计入成本。

## 5. 正式实验设计

以下设置来自 [net-benefit.yaml](../configs/net-benefit.yaml) 及其配置默认值。正式运行不叠加 A100 调试配置。

| 项目 | 正式设置 |
|---|---|
| 模型 / 设备 | Qwen/Qwen3-8B / 同一节点五张 H200 |
| 模式 / 采样 | non-thinking；temperature=1、top_p=1、top_k=-1、min_p=0 |
| 数值 / batch | bfloat16、SDPA、最大 batch=8 |
| Demonstrations | GSM8K train 0–7，固定 8-shot |
| 开发集 | GSM8K train 8–17，共 10 题 |
| 测试集 | GSM8K test 1000–1049，共 50 题 |
| IID 预算锚点 | 16、32、64 条完整 rollout |
| 开发候选数量 | 8、16、32、48、64、96、128；每题每候选重复 3 次 |
| 测试重复 | 每题每预算每方法 10 次 |
| Gate | R=8、prefix_tokens=64、beta=1、n_min=1 |
| 长度 guard | 当前正式配置 1024；调试配置的 4096 不自动沿用 |
| Reference | 独立 IID，1024/题；必要时扩到 2048、4096 |
| 主要 / 次要误差 | JS divergence，bits / 平方 L2 |
| JS 阈值 | 0.01、0.02、0.05 |
| Bootstrap | 2000 次，先抽题目，再抽完整配对运行 |

freeze 必须固定模型、tokenizer、数据集 revision，保存精确 prompt token IDs，并登记其他曾经用于调参的题目。测试集不能与开发题或已登记调参题相交。

开发标定只根据成本，为每种方法选择最接近 IID FLOP 锚点的候选数量。当前最大相对偏差为 35%；不满足时停止，不能自行外推数量。所有数量在正式测试前封存，测试时必须完成既定路径，不能因耗时、答案或长度提前停止整个估计器。

预期规模用于检查缺失结果，不代表已有产出：

- 开发：10 × 7 × 3 = 210 个配对块，四方法共 840 个结果文件。
- 测试：50 × 3 × 10 = 1500 个配对块，四方法共 6000 个结果文件。
- 初始 reference：50 × 1024 = 51,200 条独立 rollout；按 128 条分块时为 400 个任务。导入 reference 时任务数相应减少。
- Warmup 和 FLOP replay 另有实际执行量，不包含在上述结果文件数量中；必须保留各自成本记录。

## 6. 成本与统计判据

### 6.1 两套成本

FLOPs 是实际执行的 tensor 算子与形状计数，涵盖模型、entropy、CDF、分配和加权 histogram。FMA=2，其他算子按冻结规则计数；遇到未覆盖浮点算子停止。它不是硬件指令计数。

GPU 秒数是 GPU 同步后的完整算法 wall time：从 prompt prefill 前，到解析和分布汇总后。包括缓存复制、CPU 大整数、Python 调度、内存操作与同步造成的占用，不等于 kernel busy time。

先执行无计数器的计时运行，再用完全相同 seed 重放统计 FLOPs；两次 token、权重、分配和执行账本必须一致。Profiler replay 的时间单列，不并入算法速度。

对方法 m 的每次测试运行，主成本为：

\[
C_{\mathrm{net},m}=C_{\mathrm{inference},m}+\frac{C_{\mathrm{development},m}+C_{\mathrm{warmup},m}}{Q\,B\,J},
\]

其中 Q=50、B=3、J=10；公式分别用于 FLOPs 和 GPU 秒数。包含开发候选的全部成本以及 calibration/test worker 的 warmup；代码在分析时补入实际 test worker warmup。额外调试、reference 扩充阶段的 warmup 等开销应在工程运行记录中单列，不隐称已计入当前主成本公式。

Reference、模型加载、正确性预检和 profiler replay 是共同实验/测量基础设施，按预注册范围单独报告。交付中同时给出 inference-only、摊销净成本和可获得的实验开销，不把净成本解释成购买这整轮 GPU 资源的全部账单。

### 6.2 Reference 与不确定性

Reference 必须来自同一目标分布的独立采样。复用只接受兼容、已封存的 v3 reference，核对模型/分词器、EOS、长度 guard、parser、dtype/attention、数值环境、精确 prompt IDs、原始样本 hash 和独立 seed。

平均 half-vs-full JS 必须不超过 0.01 bits，并通过 invalid/truncation 检查；不稳定时按冻结规则倍增到 2048 或 4096。通过后封存。Reference 是高预算近似，不是数学真值。Bootstrap CI 以 reference 和开发标定计划固定为条件，不含它们的估计不确定性。

### 6.3 在观察点上计算 speedup

对每个预先固定的 epsilon，取该方法已观察配置中平均 JS≤epsilon 的最小成本，再计算 baseline_cost / entropy_cost。没有达到阈值则返回未达到；不插值、不外推、不强制单调化。

Bootstrap 先抽题目，再在每个题目/预算内抽完整配对运行。四种方法和两种成本用相同索引；每次重新选择最佳 baseline。若不足 95% 的 draw 同时达到阈值，不给出数值 CI。

| 观测结论 | 可交付的解释 |
|---|---|
| 对 IID 和 uniform，两种成本的 speedup CI 下界都大于 1 | 该阈值下支持此实现的净收益，另报是否赢过最佳 baseline |
| 仅 FLOPs 有收益 | 尚未证明 GPU 时间收益，需要分析实现开销 |
| 只赢 IID，未赢 uniform | 尚未证明 entropy 分配的贡献 |
| 未达到阈值、CI 覆盖不足或缺 GPU 时间 | inconclusive，不补造比较点 |
| 完整有效实验未显示优势 | 报告未证明净收益；保留全部负面结果 |

判定按每个 epsilon 分别报告。成功交付不要求得到正面研究结果。

## 7. 接手 agent 的工作顺序

```mermaid
flowchart LR
    A[工程测试与开发题调试] --> B[8B/H200 freeze]
    B --> C[开发题 preflight 与人工审核]
    C --> D[开发成本标定]
    D --> E[封存 sample counts]
    E --> F[配对测试与独立 reference]
    F --> G[reference 收敛检查]
    G --> H[分析、图表与交付报告]
```

1. 阅读本 RFC、预注册、README；检查当前 run 的模型/设备身份，不从目录名推断实验已完成。
2. 跑共享工具与 v3 测试。需要 CUDA 排错时用 1.7B/A100 和开发题，记录 `debug_only` 证据。
3. 回到正式 8B/H200，使用独立 run 重新 freeze、预检、人工审核、标定。不能复用 debug 的计划或结果。
4. 启动完整配对测试和 reference。发生失败时定位原因并保留证据，不跳过失败 seed。
5. 输出以下交付包；若阶段尚未通过，明确标为 partial 或 blocked，并交代完成条件。

## 8. Delivery 合同

### 8.1 程序自动生成的证据

路径均相对于正式运行目录。必须交付整个运行证据，不仅是 PNG。

| 交付组 | 必需文件或目录 | 验收要点 |
|---|---|---|
| 冻结身份 | `config.yaml`、`manifest.json`、`questions.jsonl`、`development_questions.jsonl`、`prompt_fingerprint.json`、`development_fingerprint.json` | 8B/H200，精确 revisions、题目、prompt 与代码/依赖指纹 |
| 正确性预检 | `preflight.json`、`preflight_rollouts.jsonl`、`manual_review.csv`、`manual_audit.json` | 预检通过，至少 50 条由人审核；不能由 agent 自行填批准 |
| 开发标定 | `results/calibration/`、`calibration_plan.json`、`calibration_seal.json` | 840 个结果，sample counts 与成本来源可追溯 |
| 正式采样 | `results/test/` | 6000 个结果；配对完整，同块同 worker/GPU，包含 raw token IDs、答案、权重和 allocation |
| 方法成本 | 结果中的 `cost`、`ledger`；`worker_overhead/` | FLOP 覆盖通过、replay 一致；保留推理、warmup、开发与计数重放记录 |
| 调度证据 | `queue.sqlite`、`logs/` | 无未完成/失败的必需任务；保留 worker 和错误信息 |
| 独立 reference | `reference_chunks/` 或 `reference_import/`，以及 `reference_distributions.jsonl`、`reference_convergence.csv`、`reference_checks.json`、`reference_complete.json` | 样本完整、收敛/质量通过、seal 与源 hash 可核对 |
| 统计结果 | `net_analysis/` 中的 `per_run_results.csv`、`observed_cost_error_curves.csv`、`paired_comparisons.csv`、`speedups.csv`、`decision.json` | 同时包含 JS/L2、两套净成本、配对比较和所有 epsilon 的状态 |
| 核心图 | `net_analysis/` 中的 `error_vs_total_flops.png/.pdf`、`error_vs_gpu_seconds.png/.pdf` | 两张图，实际观察点与 CI，图例清楚，成本口径明确 |

`results/*/*.json` 的文件外层是 `sha256` 和 `result`。验收时用 `load_result()` 验证，而不是只数文件。每条结果中保留 `method`、`samples_requested`、`seed`、`distribution`、`samples`、`allocation`、`ledger`、`cost`，以及题目、预算、重复编号、协议、计划和设备身份。

### 8.2 由执行 agent 补齐的说明

以下文件是本 RFC 要求的交付物，**当前 CLI 不会自动生成**。放在正式运行的 `delivery/` 下：

| 文件 | 必须回答的问题 |
|---|---|
| `report.md` | 正式模型/设备和样本范围是什么？每个 epsilon 的 entropy-vs-uniform、IID、Arithmetic、best baseline 结果是什么？CI 是否足够？误差、净 FLOPs、净 GPU 秒数、inference-only 成本分别是多少？结论的限制是什么？ |
| `validation.md` | 测试命令、日期、环境和实际结果；预检、人工审核、reference、完整网格、checksum、同 GPU 配对检查是否通过；调试与正式证据如何区分？ |
| `handoff.md` | run 路径、协议 ID、计划 ID、当前阶段、队列状态、失败原因、尚缺产物、接下来一条可执行命令，以及谁需要提供人工审核？ |
| `environment.md` | 实际 Python/依赖、模型缓存路径、GPU/驱动和进程级环境设置；给出恢复环境的方法，注明未能重建的依赖，不包含凭据。 |
| `source/` | 保存与 manifest 指纹匹配的源码、配置、脚本、预注册、依赖描述，以及本 RFC/README 的副本。仅给仓库当前路径不足以重现已经变更的代码。 |
| `SHA256SUMS` | 对最终交付文件生成校验清单，自身除外；生成后不继续改写已归档文件。 |

SQLite 使用 WAL。应在 worker 停止后做一致归档，或使用 SQLite backup；不能在运行中只复制一个 `queue.sqlite` 并声称已保存完整状态。

### 8.3 完成定义与失败交付

完整交付必须同时满足：正式身份正确、预检和真实人工审核通过、开发计划封存、完整测试网格与独立 reference 可验、两张图和统计表齐全、说明报告能追溯到同一协议。

允许交付有效的负结果或 inconclusive。若仅完成 CPU/demo、1.7B/A100 debug、部分正式采样，或 reference 仍不稳定，则只能交付阶段报告；写明缺少什么以及为何不能下正式结论。不得将未运行、跳过、失败或人工未批准写成通过。

## 9. 需要保持的实现约束

- 模型分布保持全词表原始采样；parser 的 invalid 和长度截断质量不可丢弃。
- gate 只能看已生成前缀。保持父权重与 child 权重守恒，不额外乘模型似然。
- 完成预先确定的全部路径；不按耗时选择幸存样本。
- 同配置计时与 FLOP replay 严格一致。跨 batch 路径一致性仅为诊断；固定前缀的最大行概率 TV≤0.01，不能解释成完整答案分布保证。
- 不修改运行中的源码、冻结配置、预注册或 seals。规则变化需要新 run 和相匹配的 reference。
- 真实失败会停止队列。`recover` 只恢复同机已退出进程留下的 running 任务，不重置真实 failed 任务。
- 五卡启动器只管理自己的 worker，不驱逐共享机器上的其他任务。

优先阅读 [experiment.py](../egs/net/experiment.py)、[engine.py](../egs/net/engine.py)、[pipeline.py](../egs/net/pipeline.py)、[analysis.py](../egs/net/analysis.py)。涉及数值、reference 或调度的修改，分别复核 [arithmetic.py](../egs/net/arithmetic.py)、[reference.py](../egs/net/reference.py)、[queue.py](../egs/net/queue.py) 与对应测试。
