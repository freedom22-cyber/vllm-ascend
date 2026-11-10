# profiling 沙盒 override（继承主部署图+MTP 基线 + P/D 双侧 profiler 端点）
# - 图模式: D 侧 FULL_DECODE_ONLY；P 侧 eager（与主部署一致）
# - MTP: eagle depth 5（env.sh 默认）
# - profiler: torch(torch_npu) profile，产物落 /pd/logs/profile（=SFS 本目录 logs/profile）
ENABLE_GRAPH=1
PREFILL_ENABLE_GRAPH=0
DECODE_ENABLE_GRAPH=1
ENABLE_MTP=1

# 【最终态】优化栈: 修复分支(b4d43fd1b) + D 图模式 + depth3 + pin 47.88GB。
# 基线轮历史: 未修复代码在本机图模式不可服务(EE1006 复现 4/4),基线取未修复+D eager。
DECODE_ENABLE_GRAPH=1
# 【基线轮 A】未修复代码(1b88fd5c8)基线:与 -opt 沙盒 pre-fix 状态完全等效
#   (spec depth 3 + kv pin 47.88GB),图模式若可跑即以此测基线;若仍 EE1006 再退 eager。
# 【M1-L1】depth 3→5(吞吐6000优化计划): draft 前向仅 0.79ms/次,实验室 accept 3.73 vs depth3 2.28。
#   负载下 accept 衰减是主要风险,depth4/5 双测取拐点;KV pin 47.88G 需复核(draft KV 多 2 层)。
# 【2026-11-07 深夜定稿】depth 负载下 A/B 结论: c64 depth5=2016 tok/s / TPOT 26.3ms,
#   劣于 depth3=2207/24.9(-8.7%),与 -opt batch=1 消融同向 → 短上下文(2048/512)depth3 最优。
#   depth 结论按负载分治: SLA 沙盒(64k/1k)n5 最优。depth4 拐点未测,P1 备选。
#   只保留一处 DECODE_SPEC_CONFIG 赋值(历史上两处赋值+两会话翻转导致 depth 漂移,勿再加行)。
DECODE_SPEC_CONFIG='{"method":"eagle","model":"/models/IQuest-Q1/mtp","num_speculative_tokens":3}'

# 【2026-11-09】profiler 产物分侧落盘: P→logs/profile/prefill、D→logs/profile/decode。
# 注意: 对正在运行的引擎无影响,引擎下次重启后生效(首次 c32 采集的旧数据已手工分拣)。
PROFILER_ARGS_P='--profiler-config {"profiler":"torch","torch_profiler_dir":"/pd/logs/profile/prefill","torch_profiler_with_stack":false}'
PROFILER_ARGS_D='--profiler-config {"profiler":"torch","torch_profiler_dir":"/pd/logs/profile/decode","torch_profiler_with_stack":false}'
# 【M3】P 侧 EP8+MC2（P 供给 20.5k→24k 主攻）: prefill MoE dispatch/combine 走 FUSED_MC2
#   （SLA 沙盒 colocated 已验证；选择逻辑无容量门，8192-tok chunk 实测验证）。
#   失败回退: PREFILL_EXTRA_ARGS 去掉 --enable-expert-parallel 并置 PREFILL_ADDITIONAL_CONFIG="-"。
PREFILL_EXTRA_ARGS="${PROFILER_ARGS_P} --enable-expert-parallel"
# 【L5】triton QK-norm(消 qkv split 拷贝 ~3.1ms/步)+ o_proj GEMM+AR 融合(L3a,
#   npu_mm_all_reduce_base);两者各自独立可回退(键删掉即回退)。
PREFILL_ADDITIONAL_CONFIG='{"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}'
# D 侧: profiler 参数 + KV 显存钉 32GiB(M1: 原 44.6GiB,EP8+MC2 下图池/MC2 buffer 吃满 91.75G 在
#   Running~64 时 OOM;2048/512 负载 c256 仅需 ~24G KV,降 pin 让位给图池)+ EP8(M2-config)
DECODE_EXTRA_ARGS="${PROFILER_ARGS_D} --kv-cache-memory 34359738368 --enable-expert-parallel"
# 【M2-config】FUSED_MC2(.enable_fused_mc2): SLA 沙盒验证 +1%(colocated eager);D 图模式(aclgraph)下未验证,
#   与 SP MoE 互斥(2044419f8),不与 flashcomm1 同开。失败则回退本行(去掉 enable_fused_mc2)。
DECODE_ADDITIONAL_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}'
# 【M1-L2b】捕获档位显式覆盖到 1024: seqs=256 后 MTP 验证批 = seqs×4 = 1024 token,
#   默认推导只到 512,768/1024 批次回退 eager(实测 c192 步进 74→333ms,吞吐崩到 1271)。
#   档位含 c64/96/128/192/256 的验证批精确值(256/384/512/768/1024);图池显存靠 pin 32G 让位。
GRAPH_CAPTURE_SIZES_JSON='[1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024]'

# ==== 4.4/4.5 纯P/纯D 配比测试(2026-11-09)：P 侧测试配置（方法论 4.4）====
# 测完删除本块并重启 P 即恢复生产配置(32/8192/131072)。
# max_model_len = 110000+1024+1000；110k/1k 口径,纯P prefix90%/纯D prefix99%
PREFILL_MAX_NUM_SEQS=8
PREFILL_MAX_NUM_BATCHED_TOKENS=32768
PREFILL_MAX_MODEL_LEN=112024

# ---- D 侧 110k/1k 测试配置: num_blocks 5890×128=753,920tok ÷111,024/seq=6.79→6(88%占用) ----
GRAPH_DECODE_MAX_NUM_SEQS=6
DECODE_MAX_NUM_BATCHED_TOKENS=192
