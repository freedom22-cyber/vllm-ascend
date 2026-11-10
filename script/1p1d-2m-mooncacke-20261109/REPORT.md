# IQuest-Q1 1P1D 服务化性能分析与吞吐优化报告（-prof 沙盒）

日期：2026-11-07 ｜ 沙盒：new5(10.222.34.63, P0) / new6(10.222.44.142, D0)
代码：vllm-ascend worktree `vllm-ascend-prof`，分支 `iquest-q1-pd-quant-prof`
任务：分析 vLLM 服务化性能瓶颈 → 制定并执行 prefill/decode 优化 → 验收吞吐 ×2
> 措施清单与可迁移经验（根因诊断/profiling 坑/瓶颈结构）已沉淀至 `docs/PD_性能优化经验沉淀.md`。

## 1. 结论（TL;DR）

| 口径：vllm bench serve 经 proxy 1999，dataset random ISL=2048 OSL=512 ignore-eos，聚合输出 tok/s |
|---|---|---|---|
| 指标 | 基线（1b88fd5c8 + D eager） | 优化后（b4d43fd1b + D 图模式 depth3） | 提升 |
| c32 输出吞吐 | 293.1 tok/s | **1292.9 tok/s** | **4.41×** |
| c64 输出吞吐 | 550.5 tok/s | **2144.9 tok/s** | **3.90×** |
| c32 总吞吐 | 1465.7 tok/s | 6464.5 tok/s | 4.41× |
| c64 总吞吐 | 2752.5 tok/s | 10724.6 tok/s | 3.90× |
| c32 TPOT | 92.8 ms | 20.2 ms | 4.6×↓ |
| c64 TPOT | 96.4 ms | 25.1 ms | 3.8×↓ |
| c32 TTFT 均值 | 2363 ms | 1385 ms | 1.7×↓ |
| c64 TTFT 均值 | 2027 ms | 1339 ms | 1.5×↓ |

**吞吐 ×2 目标达成（c32 4.41×，c64 3.90×），且 0 失败请求。**

## 2. 基线为何取“未修复 + D eager”

- 未修复代码（1b88fd5c8，含 2d69f5917 引入的 packed 页级交错 KV 分支）**在本 sandbox（新型号
  hyperinstance）图模式完全不可服务**：首个真实请求必炸
  `EE1006 Not_Supported: Updating the task group failed ... Only tasks running on Cube/Vector
  Core support task group update`（外层包装成 torch.OutOfMemoryError，与显存无关：
  kv pin 42.85/47.88GiB × depth5/depth3 复现 4/4）。
- 根因 = [[iquest-kv-interleave-copy-bug]]：`vllm_ascend/models/iquest_q1/attention.py` 的
  eager/capturing 路径无条件 `.contiguous()`，把页级交错的 K/V cache 物化成大拷贝任务并被
  捕获进 ACL 图；updatable-graph（enable_npugraph_ex）回放时对图内任务做地址更新，
  memcpy 类任务不支持更新 → EE1006。-opt 沙盒（老机器）能跑通是 CANN/驱动版本差异。
- 主部署（hyperinstance 43.66/42.240）2026-11-07 00:58 重启后尚未承载真实流量，同样代码
  同样配置，**预计首个请求就会踩同一问题**，需要 cherry-pick 本沙盒同款修复。
- 因此基线取未修复代码在本机唯一可运行形态：D 侧 eager（C4 消融形态），其余配置不变。

## 3. 优化内容（prof 分支 b4d43fd1b，自基线 1b88fd5c8 起 4 个提交）

均来自 -opt 沙盒已验证的链路，本沙盒 cherry-pick（-x -s 保留溯源）：

1. `0f2dfc0c2`（71c4c6b33）FIA v2 draft replay key names by backend capability
   —— eagle draft 回放的 FIA 参数键名按后端能力选择（actual_seq_qlen vs actual_seq_lengths）。
2. `e1748e675`（76f70f7d9）IQuestQ1RMSNorm 融合为 npu_rms_norm（P/D 双侧受益）。
3. `7f45a2c5d`（3c60b40fd）**核心修复**：仅在 `block_table is None` 时 `.contiguous()`，
   不再每步物化整层交错 KV（原 ~48GB/步 HBM 流量 + 图内大拷贝）。修复后图模式可服务。
4. `b4d43fd1b`（eb3c19d07）清理捕获期调试打印。

配置侧（env.override.sh）：D 图模式 FULL_DECODE_ONLY + MTP eagle depth 3（-opt 消融保留值）
+ `--kv-cache-memory 47881106554`（47.88GiB，-opt 同值）；P 侧保持 eager + prefix-caching。
depth3 实测 MTP acceptance length ≈ 2.28 tokens/step（接受率 ~43-46%）。

## 4. Profiling 采集与分析（双侧 torch_npu profiler，conc8/32 prompts 负载下）

产物：`logs/profile/`（SFS）16 rank 目录 + kernel_details.csv 等。
工具：collect_profile.sh（同步采集）→ analyse_profile.sh（容器内 torch_npu analyse）→
tools/analyze_trace.py（kernel_details.csv 流式分析：busy/idle、分桶、top、空隙切分）。

### 4.1 P 侧（prefill，rank0，28.9s 窗口）

| 构成 | 数值 |
|---|---|
| 设备 Free（空闲） | **81.4%**（63 个 >3ms 空隙，最大 4.2s） |
| Computing | 8.1%（2.36s） |
| Communication（未重叠） | 10.4%（3.02s，**大于计算**） |

- P 设备大部分时间空闲是 1P1D 形态使然（prefill 突发 → 等 D 解码），不直接限制吞吐。
- 忙时内部：comm 19.9% > matmul 3.8% > moe 1.4%（见 kernel 分桶）；
  top 内核 AivKernel 2.6s/4631 次、GroupedMatmulSwigluQuant 428ms、MatMulV3 349ms、
  FusedInferAttentionScore 仅 121ms。
- **P 侧 TTFT 优化方向**（设备侧）：TP8 allreduce 占 prefill 忙时大头 →
  通信计算重叠（MC2/融合 AR+norm）、或增大 `max-num-batched-tokens`（当前 4096）
  减少切片次数；调度侧：TTFT P90 4.3s（c32）主要来自排队与 KV 传输串行。

### 4.2 D 侧（decode，rank1，36.4s 窗口，设备忙时 93.1%）

> rank0 的解析产物被早期误杀进程截断（只含 CPU 事件）；rank1（1.6GB，干净单实例解析 30min）数据完整，
> kernel_details.csv 470 万行。图模式下设备忙时 93.1%、>3ms 空隙仅 0.2%——D 侧已压满。

| 分桶 | 占忙时 | 说明 |
|---|---|---|
| matmul | **36.0%**（13.1s） | GroupedMatmulSwigluQuant 6.3s + GroupedMatmul 4.0s + MatMulV3 1.5s + QuantBatchMatmul 1.0s |
| comm | **27.9%**（10.1s） | TP8 allreduce，每层×每 MTP 微步，未与计算重叠 |
| other | 11.9% | AivKernel 5.0s（20.8 万次，小向量核任务） |
| moe | 8.2% | routing/init/finalize（MoeInitRouting 2.1s） |
| norm_rope | 7.6% | RmsNorm 2.4s（51.4 万次，avg 4.7us——已是 npu_rms_norm 融合核） |
| copy_transpose | 6.6% | 均为 2-2.1us 量级小 cast/slice；**修复前的 53.7% 大拷贝已消失** |
| attention | 5.0% | FusedInferAttentionScore 1.8s（10 万次 avg 18.3us） |
| quant + sample | 3.9% | DynamicMxQuant 0.9s 等 |

按核分布：AI_VECTOR_CORE 28.2% ≈ COMMUNICATION 27.7% > MIX_AIC 25.3% > AI_CORE 15.8%——
向量核开销（norm/小拷贝/AivKernel）与通信相当，"小核税"明显。

- **对比修复前**（-opt 沙盒 pre-fix profile）：单 aclnnInplaceCopy_Slice 占设备 53.7%（~48GB/步 HBM
  流量）→ 现在 copy_transpose 合计 6.6% 且无单点大核；设备忙时从大量空转变为 93.1% 满负荷。
- **D 侧剩余瓶颈排序**：① matmul（MoE 分组 GEMM 为主，结构固有）；② TP8 allreduce 串行链
  （每层阻塞式通信，36%+28% 之和 = 64% 都在等 GEMM 与通信）；③ 小核税（<5us 核 200 万+ 次调用）。
- **下一步候选（D）**：AR+RmsNorm 融合（MC2，可同时吃 comm 与 norm_rope 两桶）；eagle depth 负载下
  复调（depth↑ 增加 GEMM/comm 次数但摊薄每 token 开销，3→5 需重测）；`DECODE_MAX_NUM_SEQS` 上调
  （attention 仅 5%，KV 余量足，batch 加大摊薄固定开销）；图捕获 batch 与实际分布对齐。

### 4.3 引擎宏观指标

- c32/c64 压测期间 D 引擎 Running=32/64，KV usage <8%（47.88GiB KV 余量充足）。
- MTP acceptance length ≈ 2.28（depth3），即 TPOT ≈ step_time / 2.28。

## 5. 复现与运维要点

- 重启引擎：宿主机侧 `bash start.sh stop prefill|decode` + `start.sh start prefill|decode`
  （勿用 stop p0/d0，会连 proxy/master 一起杀；容器内无 docker）。
- bench：`bash bench.sh <conc> [num] [label]`（ISL/OSL 环境变量可调），结果 `logs/bench/<label>.json`。
- profiling：`bash collect_profile.sh <conc> [num] [label]`（端点用节点业务 IP，
  P 引擎只绑业务 IP，127.0.0.1 不通）；analyse 需在容器内跑
  `docker exec iquest-pd-<x>-prof bash /pd/analyse_profile.sh rank0_<pid>`，
  **不可并发多实例**（同目录互踩，torch_npu analyse 内部是 父+子 进程组，清理要用
  `pkill -f "python3 - rank0"` 才杀得干净）。
- 本机图模式必需 `--kv-cache-memory` 钉值（引擎自检值上浮 ~2GiB 瞬态余量；
  util 0.94 裸跑会 OOM）。

## 6. 后续候选优化（按预期收益排序）

1. **D 侧**（依据 §4.2 kernel 分桶 + cannbot 技能逐层分解，详见 `docs/PD_吞吐x2第二阶段计划.md`）：
   - L1 depth3→5（draft 前向仅 0.79ms/次，984698ade 实测 depth5 accept 3.73 vs depth3 2.28）→ +50-60%
   - L2 并发 c128/c256 上探（attention 仅 7.8%，KV usage <8%）→ +30-60%
   - L3 层间暴露 allreduce 消除（bubble 52.2us/层=15.1%）：AR+RmsNorm 融合（9df31e4e6 参照）或 SP MoE
   - L4 小核税清理（elementwise_copy 13.9% + norm 7.5%）：norm+residual+quant 融合
   - L1+L2 复合即达 ×2（测算 ~4300-4900 tok/s）
2. **P 侧 TTFT**：max-num-batched-tokens 4096→8192/16384；TP8 通信重叠（comm 未重叠 10.4%）。
3. **主部署同步**：cherry-pick 7f45a2c5d（=3c60b40fd）等 4 提交并重启两侧引擎
   （当前主部署代码 = 未修复 1b88fd5c8，图模式在新型号机上首个请求即 EE1006 崩溃）。

## 7. Profiling 数据完整性说明

- P 侧 8 rank 与 D 侧 rank1 的 kernel_details.csv 已生成并分析；D rank0 的早期解析产物
  `trace_view.json`（437MB）为被误杀进程留下的**截断文件**（仅含 CPU 事件，尾部 JSON 不完整），
  已弃用；new6 上对 D rank0（3.0GB）的干净重解析仍在后台进行，完成后可对 rank0 复核，
  但 rank1 数据已足以定性（TP 组内各 rank 负载对称）。
- 所有原始产物在 `logs/profile/`（SFS），分析工具在 `tools/`。

## 9. 追加轮（2026-11-07 晚）：MTP depth5 A/B —— 负结果，保持 depth3

env.override.sh 唯一变量 `num_speculative_tokens` 3→5（seqs=128 不变），c64：
输出吞吐 **2207.4 → 2016.1（-8.7%）**，TPOT 24.9 → 26.3ms。仓库提交 `984698ade` 的
depth5 accept 3.73 在本 workload（ISL 2048/OSL 512，c64）不复现——draft 前向开销
（×5/步）超过 accept 增益。**结论：depth 结论按负载分治**：本沙盒 depth3 最优；
SLA 沙盒（64k/1k 长上下文）n5 最优已实测（n5final > n7 > n3safe）。第二阶段计划 L1 作废，
转入 `docs/SLA_吞吐6000调优计划.md`（SLA 口径路线）。

## 8. 追加轮（2026-11-07 下午）：单卡吞吐 300 tok/s 目标达成

目标口径：聚合输出 tok/s ÷ 8 卡 ≥ 300（即总量 ≥2400）。基线复测 c64 = 2207.4 tok/s（275.9/卡，TPOT 24.9ms）。

| 配置 | 并发 | 输出吞吐 | **单卡** | TPOT | 失败 |
|---|---|---|---|---|---|
| seqs=64（原） | 64 | 2207.4 | 275.9 | 24.9ms | 0 |
| **seqs=128** | 96 | **2854.5** | **356.8** | 28.9ms | 0 |
| **seqs=128** | 128 | **3510.4** | **438.8** | 31.8ms | 0 |

- 唯一变量：`DECODE_MAX_NUM_SEQS` / `GRAPH_DECODE_MAX_NUM_SEQS` 64→128（env.sh）。
  依据 §4.2：attention 仅占忙时 5%、KV usage <8%（47.88GiB KV 大量余量），加大 batch
  摊薄每步固定开销（allreduce 次数不变但每 token 分摊、MTP/采样/小算子 per-step 开销）。
- 引擎日志实证：Running 实际到 128（此前被 64 封顶），无 preempt/无 OOM。
- TPOT 代价：c96 28.9ms / c128 31.8ms（>25ms SLA）。吞吐优先口径下目标已达成；
  若需同时守 TPOT≤25ms，取 c≈80（吞吐约 320+/卡，待实测）。
- 仓库历史 PR 排查结论（本轮参考）：`e205cc961`（MoE 输出 combine 的 MulsAdd 融合扩展，
  覆盖 shared-first foreach_add）**不在本分支**，本分支 `fuse_muls_add` 默认开但匹配面窄
  → cherry-pick 候选，可打 §4.2 的 other/AivKernel 11.9% 桶；`426a26bc1`（shared-expert
  multistream overlap）已启用（additional_config）；`401f15948`（EPLB）单机 TP8 不适用；
  `a76ae6d4b`（MegaMoE）不适用。

## 10. 追加轮（2026-11-07 下午，P0 标定）：seqs128 冷突发不稳定 + depth 归因修正

**c128@seqs128 冷启动突发不稳定（3 次独立死亡，1 次确认 OOM 栈）**：
- 15:26 depth5+seqs128 冷启动首波 512 请求 → Worker NPU OOM（"Tried to allocate 424.00 MiB,
  87.63/91.75 GiB already allocated"）→ EngineCore 死。OOM 栈完整（engine.log 07:26:02）。
- 15:5x 重启 boot 仅 C=8 小探针预热后 c128 → 同signature死亡（TPOT 91ms、415 失败、EngineDead）。
- 12:19 boot 同配置存活——**差异是 c96（384 prompts）先行全量预热**；小探针预热不足。
- **规则（进运维要点）**：引擎重启后第一次压测必须 ramped（c32→c64→c128）或把
  `--kv-cache-memory` 47.88→47.0G 换 ~0.9G 瞬态余量（KV 容量损失 ~4%，吞吐无感）。
  c128 数值结论（3510 tok/s）依赖预热路径，作为 SLA 上探参考值仍有效。

**depth 归因修正（配置漂移链）**：`start.sh` 只 source env.sh（base=depth5）；env.override.sh
存在**两处 DECODE_SPEC_CONFIG 赋值且在两会话间翻转**（15:1x=3 → 15:3x=5）。15:20 boot 实跑
depth5（SpeculativeConfig 行实锤）。§8 的 c96/c128（2854/3510）= depth5+seqs128 组合，
§9 已证 depth5 劣于 depth3（c64 2016 vs 2207）→ **depth3+seqs128 矩阵格待补**（被 c128 冷突发
不稳定阻塞；需先落 ramped 预热或 pin 调整）。**任何 depth/配置结论前先核对 engine.log 的
SpeculativeConfig 行。**

**P0 标定产出**（详见 docs/IQuest-Q1_下一阶段SLA吞吐调优计划_20261107.md §6）：
AR 微基准（decode 载荷延迟主导 → custom AR/融合优先于带宽优化）；MTP 流式 chunk≈每步
（TPOT 用 usage 重算，C=8 TPOT≈11ms、acceptance 3.07）；mooncake span 与 pqing b493cbd
对照=非漏修。引擎已 clean stop（8/8 NPU 空闲）交还共享沙盒。
