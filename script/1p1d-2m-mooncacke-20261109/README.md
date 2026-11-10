# IQuest-Q1 1P1D 双机 PD 分离（Mooncake 实时 KV 传输, vLLM 0.30.0）

两台 950DT 机器（P0=node7 10.222.39.253，D0=node6 10.222.44.20；2026-11-06 两次换节点：
43.87 → 44.62（卡1楔死，npu-smi 已无该卡）→ 39.253，见 env.sh 头注）：P 侧 prefill + proxy，
D 侧 decode + MTP 投机解码；128k 上下文。基于 deepseek-v41-1p1d-2m-v0.30.0-mooncake
参考部署改造，主要差异：

| 项 | deepseek 参考 | 本部署 |
|---|---|---|
| 引擎形态 | external DP（8 进程×1rank×TP1，launch_online_dp.py） | **每侧 1 进程 TP8**（606G bf16 权重单卡放不下） |
| KV 连接器 | MultiConnector = MooncakeHybrid + AscendStore(mooncake 池) | 默认**仅 MooncakeHybridConnector**（P→D 实时传输）；池化模式见下文 `ENABLE_KV_POOL` 一节 |
| mooncake_master | 需要（池后端） | 默认不需要；池化模式下需要（自动管理） |
| MTP | dspark（P/D 都开） | **eagle（仅 D 侧）**，num_speculative_tokens=5 |
| 上下文 | 1M | 128k（模型原生支持 512k，取 128k 为 KV 预算平衡点） |
| eager | P eager / D 允许图 | 默认两侧 eager；图模式见 start_pd_graph_mtp.sh |

## 架构

```
              client ──► proxy:1999 (P0 = 10.222.39.253 node7)
                           │ 请求路由到 prefiller
                           ▼
              P: vllm serve TP8, 端口 7100, kv_producer (kv_port 28010)
                           │ Mooncake 实时 KV 传输（28010 → 29010）
                           ▼
              D: vllm serve TP8 + MTP, 端口 7100, kv_consumer (kv_port 29010)
                 (D0 = 10.222.44.20 node6)
```

仓库树（fork 分支 `iquest-q1-pd-quant` @2d69f5917 = 7e970807f + 7 个 PD 适配提交：
ModelSlim 量化加载 ×3、Mooncake 交错 KV span 注册、draft spec 全注意力声明等）
与部署目录均放 SFS（/mnt/sfs_turbo/chensq/pd/），
两台机器各自 bind-mount 为容器 /workspace 与 /pd。

## 已验证状态（2026-11-06）

- **A 套 W8A8C8-mxfp8 全绿**: 短问答/完整生成/17.5k+125k 口令命中（125,004 tokens
  全链 P prefill → Mooncake KV → D decode），D 侧 MTP acceptance 3.0-4.0/6、
  draft 接受率 41-60%（与单机水平一致）。
- **B 套 bf16 全绿**: 同上全部用例通过（125k 口令 ✓），MTP acceptance 2.8-3.4/6、
  接受率 35-47%；temp=0 身份问答三连一致（smoke 用例带温度下的错字属采样噪声）。
- 交错 KV 布局坑: Ascend 后端每层给连接器的是 `[blocks,2,tokens,h,d]` 单缓冲的
  K/V strided 视图（V 偏移一个 K 页），transfer engine 拒绝重叠注册 — 连接器按
  span 注册（page=[K_b|V_b]=64KB），见 `mooncake_hybrid_connector.py` 非 hybrid 分支。
- MTP×PD 冲突已解: draft 的 KV spec 声明为全注意力（内核窗口 512 不变），
  否则 disable-hybrid 单组合并时 {4096,512} 两个窗口值报错。

## 文件

- `env.sh` — 全部可调项（IP/端口/并行/显存/MTP 开关/KV 池化）
- `container.sh` — 当前机器创建角色容器（p|d）
- `start.sh` — 编排：start/stop [decode|prefill|proxy|master|p0|d0|all]
- `prefill/launch_engine.sh` `decode/launch_engine.sh` — 容器内引擎启动
- `mooncake/mooncake.json` — 池化模式 mooncake 存储配置（master 地址/段大小等）
- `mooncake/run_master.sh` — 池化模式 mooncake_master 容器内启动器（仅 P0）
- `run_proxy.sh` — proxy（镜像自带 disaggregated_prefill_v1 load_balance_proxy）
- `check_pd_health.sh` — 端点/容器/NPU 健康检查
- `smoke_test.sh` — 端到端问答（LONG=1 追加 ~100k token 长上下文用例）
- `sync.sh` — node11 上把部署目录同步到 SFS
- `runtime_env.sh` — 容器内 CANN/ATB 环境加载
- `serve_cmd.sh` — P/D 侧 vllm serve 参数统一生成（launch 与 plan 共用，避免漂移）
- `start_pd_graph_mtp.sh` — **PD 混部 + 图模式 + MTP 一体化启动**（见下节）

## PD + 图模式 + MTP 一体化启动（start_pd_graph_mtp.sh）

把「双机 1P1D PD 分离」「D 侧 ACL graph」「D 侧 MTP」合成一条命令。
图/MTP 开关写进部署目录的 `env.override.sh`，P/D 两侧容器内的 `/pd/env.sh`
都会 source 到它 → 一处开关，两台机器同时生效。

```bash
# 只看将要生效的配置与 serve 命令（不启动，不占卡）
bash start_pd_graph_mtp.sh plan

# 启动（自动写覆盖文件 → 同步 SFS → D0 decode → P0 prefill → proxy → 健康检查）
bash start_pd_graph_mtp.sh start

# 别人在用卡时会被拦下，确认可抢占再加 FORCE=1
FORCE=1 bash start_pd_graph_mtp.sh start

# 回到 eager PD 行为
bash start_pd_graph_mtp.sh disable
```

其它子命令：`enable`（只写配置）、`disable`、`preflight`（静态检查，不占卡）、
`sync`（只同步到 SFS/D0）、`status`（健康检查）、`stop`。

### 图模式下的关键取舍

| 项 | eager PD（已验证） | 图模式 PD |
|---|---|---|
| D 侧模式 | `--enforce-eager` | `--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}` |
| 上下文 | 131072 | 131072（2026-11-07 从 32768 上调：A 套 KV 1.02-1.07M tokens/侧，128k 序列余 7.8-8.2x 并发；128k P→D 传输 eager 已实测） |
| D 侧并发 | 64 | 16（`GRAPH_MAX_NUM_SEQS`） |
| D 侧显存利用率 | 0.92 | 0.94（`GRAPH_DECODE_GPU_MEM_UTIL`） |

原因：图池吃显存（单机实测 util 0.94 时 KV 仅剩 0.11GiB，见
`MTP_适配与验证报告.md` §4.4）。想更长上下文就压图池：
`GRAPH_CAPTURE_SIZES_JSON='[1,2,4,8]'`（须 ≤ 该侧 max_num_seqs）。
调参入口全部在 `env.sh` 的「图模式」段。

P 侧默认仍 eager（只做 prefill，开图白吃图池显存）；需要 P 侧也开图时把
`PREFILL_ENABLE_GRAPH` 置 1。

## Mooncake KV Cache Pool 池化（ENABLE_KV_POOL，参考文档 5.4 流程适配）

在实时 P→D 传输之上叠加**可复用 KV 池**：`MultiConnector` 的两个子连接器 ——
`MooncakeHybridConnector`（实时 P→D，与非池化配置逐字段相同）+
`AscendStoreConnector`（backend=mooncake，池存取），并运行 mooncake_master
集中管理池元数据。默认关闭（`ENABLE_KV_POOL=0`），对已验证的非池化行为零影响。

### 与文档 5.4（DSV4.1-Flash）的适配差异

| 项 | 5.4 文档 | 本部署 |
|---|---|---|
| 拓扑 | P DP4/TP4 ×4 引擎，D DP8/TP2 ×8 引擎 | P 1×TP8 / D 1×TP8（PD_EXTRA 不变） |
| use_layerwise | false（CED 混合注意力） | **false（IQuest-Q1 同为 hybrid attention：full+sliding 混合层，config.json use_hybrid_layers=true）** |
| MTP | dspark | eagle（仅 D 侧，沿用） |
| P 侧前缀缓存 | 开 | 池化模式自动开（非池化仍关） |
| master | 独立进程/节点 | P0 容器内（`start.sh start master` 管理） |

### 开启步骤

```bash
# 1) 写开关（P/D 两侧容器内的 /pd/env.sh 都会 source 到 env.override.sh）
echo 'ENABLE_KV_POOL=1' >> env.override.sh

# 2) 启动（顺序 master → decode → prefill → proxy；start.sh start all 已按此编排）
#    手动逐台（P0→D0 未配免密时）:
ssh root@10.222.39.253 ".../start.sh start p0"      # master(池化时) + prefill + proxy
ssh root@10.222.44.20 ".../start.sh start d0"
#    或在 P0 上: bash start.sh start all（需 P0→D0 免密）
```

### 组成与行为

- `mooncake/mooncake.json`：`metadata_server=P2PHANDSHAKE`、`protocol=ascend`、
  `master_server_address=10.222.39.253:50088`（master 固定 P0，换节点时随 `P0_IP` 改），
  `global_segment_size=24GB`（每引擎进程贡献的池内存段）。
  env 变量 `MOONCAKE_MASTER`/`MOONCAKE_GLOBAL_SEGMENT_SIZE` 优先于 json
  （launch_engine.sh 自动导出，与 json 保持一致）。
- `MOONCAKE_MASTER_ARGS`（env.sh）：eviction 水位 0.9、驱逐比 0.1、
  `default_kv_lease_ttl=11000ms` —— **须大于 `ASCEND_CONNECT_TIMEOUT` 与
  `ASCEND_TRANSFER_TIMEOUT`**（连接器 env 默认值；若调大超时需同步调 TTL）。
- use_layerwise 固定 false：IQuest-Q1 为 full+sliding 混合注意力层，不支持逐层池访问
  （与文档对 DSV4.1-Flash 的限制同理），写死在 env.sh 生效配置段的 store 子连接器里。
- 日志：`logs/mooncake/master.log`；`check_pd_health.sh` 在池化模式下自动追加
  master 存活检查。

### 验证（对应 5.4 §5.4.4）

1. `bash check_pd_health.sh` → prefill/decode/proxy/master 全 OK。
2. 池写入（PUT）: 发一个 >128 token 的请求，`logs/mooncake/master.log` 的
   `Mem Storage` 应增长（>0 即写池成功）。
3. 池读取（GET）: 让 P 侧 HBM 前缀缓存冷（重启 P 引擎，或
   `POOL_PREFILL_PREFIX_CACHING=0`），重发同一前缀请求:
   - `logs/prefill/engine.log` 出现 `Reqid: ..., kvpool hit tokens: N`（N>0）；
   - `curl http://10.222.39.253:7100/metrics | grep ascend_store_load_get_keys_total`
     应 >0；
   - `logs/decode/engine.log` 仍有正常 P→D KV 传输记录（实时传输未被池破坏）。

### 注意

- **HBM 优先、池兜底**（代码 `pool_scheduler.py` 非 layerwise 分支）：请求前缀若已被
  P 侧 HBM 前缀缓存完全命中，池查找会被设计性跳过（`num_computed_tokens >= token_len
  → return 0`）。因此验证池 GET 时要保证 HBM 冷（刚重启的引擎，或
  `POOL_PREFILL_PREFIX_CACHING=0` 关掉 P 侧前缀缓存）。
- 池化模式下 P 侧前缀缓存由 `POOL_PREFILL_PREFIX_CACHING`（env.sh 默认 1）控制；
  0 时 P 侧退回 `--no-enable-prefix-caching`（此时每次重复请求都查池）。
  该组合与 `--disable-hybrid-kv-cache-manager` 在 IQuest-Q1 上已实测可启动并正常服务。
- 池内存为 DRAM 段（24GB/卡 × 8 rank/侧，`real_client Mounting segment` 日志可确认），
  与图池/NPU KV 预算相互独立；加大时同步改 json 与 `MOONCAKE_GLOBAL_SEGMENT_SIZE`。
  池数据落在各引擎 worker 注册的段里：**重启某侧引擎会丢掉它贡献的段**，
  跨节点对象仍在（master 元数据保留，GET 按 key 定位存活副本）。
- master 生命周期由 `start.sh stop master|p0|all` 管理；引擎初始化时连接 master，
  因此**必须先起 master 再起引擎**（`start.sh start all` 已保证顺序）。
- master 存活检查用 TCP 探端口（`/dev/tcp`），不能用 `pgrep -f`（会匹配到自身
  命令行造成假阳性）。

## 快速启动

```bash
# 0) 工作区机器（node11）: 同步部署目录到 SFS（仓库树已同步）
bash sync.sh

# 1) 两台机器各建容器（P0↔D0 之间未配免密，从 node11 分发执行）
ssh root@10.222.39.253 "bash /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/iquest-1p1d-2m-v0.30.0-mooncake/container.sh p"
ssh root@10.222.44.20 "bash /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/iquest-1p1d-2m-v0.30.0-mooncake/container.sh d"

# 2) 启动（逐台分发；`start.sh start all` 需 P0→D0 免密 ssh，未配置勿用）
ssh root@10.222.44.20 "bash /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/iquest-1p1d-2m-v0.30.0-mooncake/start.sh start decode"
ssh root@10.222.39.253 "bash /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/iquest-1p1d-2m-v0.30.0-mooncake/start.sh start prefill"
ssh root@10.222.39.253 "bash /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/iquest-1p1d-2m-v0.30.0-mooncake/start.sh start proxy"
#    顺序: (池化时 master 最先) → D 侧 decode → P 侧 prefill → proxy；权重加载 A 套约 6 分钟 / B 套约 10 分钟+

# 3) 验证
bash check_pd_health.sh
bash smoke_test.sh          # 短问答
LONG=1 bash smoke_test.sh   # ~17.5k token 长上下文口令（128k 用 ~125k token 手工请求已验证）

# 停止（分别在两台机器上执行；不要在 node11 上跑 stop——本机有同名旧容器 iquest-pd-p）
ssh root@10.222.39.253 "bash /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/iquest-1p1d-2m-v0.30.0-mooncake/start.sh stop p0"
ssh root@10.222.44.20 "bash /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/iquest-1p1d-2m-v0.30.0-mooncake/start.sh stop d0"
```

服务入口：`http://10.222.39.253:1999/v1/chat/completions`（model=IQuest-Q1，
思维链在 reasoning_content 字段）。

## 双权重 A/B 切换（env.sh 两行）

| 套 | MODEL_PATH | QUANT_ARGS | 说明 |
|---|---|---|---|
| A 量化 | `/models/IQuest-Q1-W8A8C8-mxfp8` | `--quantization ascend` | 309G；config.json 无 quantization_config，必须显式指定 |
| B bf16 | `/models/IQuest-Q1` | （置空 `QUANT_ARGS=""`） | 606G 原版 |

MTP 草案头两套共用 bf16 `/models/IQuest-Q1/mtp`（量化目录无 mtp/）。切套：
改 `env.sh` 的 `MODEL_PATH` + `QUANT_ARGS` → `bash sync.sh` → 两侧重启引擎。

## KV 预算与调参

- 88 层 GQA(8 KV 头)×128 head_dim，bf16，TP8 → **128k 单序列 ≈ 5.9G/卡**。
- B 套 bf16: 权重 75.8G/卡 + MTP 草案 ~1.2G/卡；util 0.90/0.92 下 KV 约 7-9G/卡 →
  可驻留 1 个 128k 序列 + 余量，或若干短请求。
- A 套 W8A8C8-mxfp8: 权重 38.6G/卡 → KV 余量 ~40G/卡 ≈ **7 个 128k 序列**。
- 启动若报 KV 不足（model max len larger than KV cache）：抬
  `*_GPU_MEM_UTIL`（0.92/0.94）或降 `PREFILL_MAX_NUM_BATCHED_TOKENS`（4096→2048，
  降低 prefill 激活 profile 峰值）。**禁用 --kv-cache-memory**（绕过 Ascend 多 group
  KV 预算缩放，OOM，见主仓库坑 11）。
- MTP×PD 已实测可行（A 套全绿，见"已验证状态"）；若异常，`env.sh` 置
  `ENABLE_MTP=0` 先让 PD 绿再排查。
- HCCL_BUFFSIZE 未按参考部署设置（1024/1600 每卡多占 ~1G，128k 预算紧张）；
  若遇 HCCL 超时/报错再加。
- 前缀缓存两侧均关闭（`--no-enable-prefix-caching`，与 PD 交接语义对齐，
  同参考部署）；池化模式例外：P 侧自动改为 `--enable-prefix-caching`（池查找需要）。

## 端口表

| 用途 | 端口 |
|---|---|
| prefill / decode API | 7100（各自机器） |
| KV 传输 | P 28010+N / D 29010+N（N=rank 偏移，TP8 → 8 个） |
| proxy 对外入口 | 1999（仅 P0） |
