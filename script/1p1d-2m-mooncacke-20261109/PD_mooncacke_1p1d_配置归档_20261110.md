# PD mooncacke 1P1D 配置归档（2026-11-10）

> 本文档归档 2026-11-10 15:00–15:15（CST）在 new2/new8 上拉起并验证通过的
> pd-mooncacke 1P1D 双机部署的**完整生效配置**。以当时进程实际命令行为准
> （非脚本反推）。引擎代码、脚本均在 SFS，容器是 --rm 临时层。

## 一、拓扑与载体

| 项 | 值 |
|---|---|
| P0（prefill） | new2 / 10.222.42.240，容器 `pd-mooncacke-p`（host 网络，--rm，Up 自 11-09） |
| D0（decode） | new8 / 10.222.41.88，容器 `pd-mooncacke-d`（同上） |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler`（vllm 0.30.0，CANN 9.2.1，cann-9.2.1 目录即 toolkit latest） |
| 容器创建 | `docker_start.sh`（NAME 改 p/d；**shm 仅 1g、无 /pd /workspace /models 挂载**，靠容器内手工符号链接补齐） |
| 容器内符号链接 | `/workspace → /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/vllm-ascend-opt`<br>`/models/IQuest-Q1-W8A8C8-mxfp8 → /mnt/sfs_turbo/models/…`（P、D 都有）<br>`/models/IQuest-Q1 → /mnt/sfs_turbo/models/IQuest-Q1`（仅 D 需要，eagle 草稿）<br>`/pd/* → 部署目录各文件/子目录`（P、D 都有；**/pd/logs 是容器层真实目录，不是链接，--rm 即丢**） |
| 代码栈 | vllm 核心=镜像自带 0.30.0；vllm_ascend=**vllm-ascend-opt worktree**（经 PYTHONPATH 追加覆盖） |
| 时区坑 | 容器日志时间戳为 **UTC**，宿主机 CST+8。日志里 07:xx = 下午 15:xx |

## 二、mooncake_master（P0，new2）

```text
/usr/local/python3.12.16/.../mooncake/mooncake_master \
  --port 50088 --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.1 --default_kv_lease_ttl 11000
```
- 由 `bash /pd/mooncake/run_master.sh` 拉起（docker exec -d pd-mooncacke-p）
- 配置 `/pd/mooncake/mooncake.json`：master_server_address=10.222.42.240:50088，
  global_segment_size=24GB，metadata_server=P2PHANDSHAKE，protocol=ascend
- ENABLE_KV_POOL=1（env.pool.override.sh）；池化要求启动顺序 master → decode → prefill → proxy

## 三、Prefill（new2:7100）

启动：`docker exec -d pd-mooncacke-p bash /pd/prefill/launch_static.sh`
脚本：`prefill/launch_static.sh`（全字面值）。生效配置：

| 项 | 值 |
|---|---|
| 模型 | `/models/IQuest-Q1-W8A8C8-mxfp8` + `--quantization ascend`（W8A8，**KV=float，C8 未启用**） |
| 并行 | TP8 × DP1 + `--enable-expert-parallel` |
| 模式 | `--enforce-eager`（P 侧无图模式） |
| 上下文 | `--max-model-len 112024`（110k/1k 口径） |
| 批次 | `--max-num-batched-tokens 32768`、`--max-num-seqs 8` |
| block_size | 128 |
| 显存 | `--gpu-memory-utilization 0.90` |
| 缓存 | `--enable-prefix-caching`（池查找依赖）+ `--disable-hybrid-kv-cache-manager` |
| KV 传输 | MultiConnector(producer, kv_port 28010, engine_id 0) = MooncakeHybridConnector + AscendStoreConnector(backend mooncake, use_layerwise=false) |
| additional_config | `{"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}` |
| profiler | torch → `/pd/logs/profile/prefill`（端点已配，需 --profiler-config 才有 /start_profile） |
| KV dtype | 未设 → **float**（checkpoint 的 mxfp8 KV 配方未启用，日志有提示警告） |

## 四、Decode（new8:7100）

启动：`docker exec -d pd-mooncacke-d bash /pd/decode/launch_static.sh`
脚本：`decode/launch_static.sh`（全字面值）。生效配置：

| 项 | 值 |
|---|---|
| 模型 | 同 P（W8A8C8-mxfp8，KV=float） |
| 并行 | TP8 + EP8 |
| 模式 | 图模式 **FULL_DECODE_ONLY**，cudagraph_capture_sizes 25 档 `[1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024]`；`ascend_compilation_config.enable_npugraph_ex=true, enable_static_kernel=false` |
| 上下文 | `--max-model-len 131072` |
| 批次 | `--max-num-batched-tokens 192`、`--max-num-seqs 6`（5890 blocks×128÷111,024≈6.79→6，88% 占用） |
| KV 显存 | 显式 pin `--kv-cache-memory 34359738368`（32GiB） |
| 显存 | `--gpu-memory-utilization 0.94` |
| MTP | `--speculative-config {"method":"eagle","model":"/models/IQuest-Q1/mtp","num_speculative_tokens":3}` |
| 调度 | `--async-scheduling`（on） |
| prefix caching | 关 |
| KV 传输 | MultiConnector(consumer, kv_port 29010, engine_id 1) = 同款两连接器 |
| additional_config | `{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true,"enable_fused_mc2":1,"enable_iquest_triton_qk_norm":1,"enable_fused_o_proj_ar":1}` |
| profiler | torch → `/pd/logs/profile/decode` |

## 五、Proxy（P0，new2:1999）

启动：`docker exec -d pd-mooncacke-p bash /pd/run_proxy.sh`
- 镜像自带 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`
- 路由：prefill=10.222.42.240:7100，decode=10.222.41.88:7100
- 探活端点 `/healthcheck`（不是 /health）；日志 `/pd/logs/proxy/proxy.log`

验证过的客户端命令：
```bash
curl -s http://10.222.42.240:1999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"..."}],"max_tokens":512,"stream":false}'
```
PD 全链路（P 预填→KV 传输→D 生成）15:14 实测 200 OK；reasoning 内容在
`message.reasoning`（iquest_q1 parser），正文在 `message.content`。

## 六、当日修复记录（2026-11-10）

1. **PYTHONPATH 覆盖坑（两个 launch_static.sh 同病）**：原 `export PYTHONPATH=/workspace`
   会冲掉 CANN set_env.sh 注入的 `ascend-toolkit/latest/python/site-packages`
   （acl/acl.so）。spawn worker 里 camem.py:27 `from acl.rt import memcpy` →
   `No module named 'acl'` → worker spawn 失败、EngineCore 卡死管道写。
   已改为 `export PYTHONPATH="${PYTHONPATH:-}:/workspace"`（追加式）。
   事故现场：P 侧 14:44 CST 首启失败，进程被 Ctrl+C 后残留僵尸 3282/3598，已清理。
2. **D 侧 MTP 草稿模型缺失**：speculative-config 引用 `/models/IQuest-Q1/mtp`，
   decode 容器 /models 下没有 → pydantic 校验报 "no config.json"（14:43 CST 失败）。
   已补符号链接（见第一节）。
3. **日志重定向**：曾补过 `> /pd/logs/…/engine.log 2>&1`（当时 exec 无重定向、
   日志靠外部手动重定向）；同日按用户要求**改回直打屏不落盘**（引擎日志不存文件，
   `docker exec -d` 启动时输出会被丢弃，要看日志就用不带 -d 的前台 exec）。
   注意 /pd/logs 在容器层（--rm 即丢）这一事实不变。

## 七、另存：当天上午的手动配置变体（11:48 CST 成功跑通）

容器本地 `engine-current.log`（UTC 03:48）记录的一次成功启动，与上文冻结配置
**不是一套**，注意区分：
- 端口 7200；max_model_len 65535；max_num_batched_tokens 16384；max_num_seqs 64；block_size **512**
- **KV C8 启用**：`--kv-cache-dtype mxfp8 --kv-cache-dtype-skip-layers sliding_window`
- 连接器 **MooncakeConnectorV1**（非 MooncakeHybridConnector）+ AscendStoreConnector
- additional_config 含 `enable_flashcomm1`、`enable_cpu_binding`、`recompute_scheduler_enable`
- async_scheduling on；enforce_eager；env 出现过 VLLM_ASCEND_ENABLE_FLASHCOMM1 /
  VLLM_ASCEND_ENABLE_PREFETCH_MLP / VLLM_RPC_TIMEOUT（镜像报 unknown 警告）
- 走镜像自带 vllm_ascend（当时 /workspace 链接尚未建好）

## 八、恢复生产值（脚本头部注释原话）

- P：`--max-num-seqs 32 --max-num-batched-tokens 8192 --max-model-len 131072`
- D：`--max-num-seqs 256 --max-num-batched-tokens 1024`
- 同步清理 `env.override.sh` 尾部"4.4/4.5 配比测试"冻结块（PREFILL_MAX_NUM_SEQS=8 等），
  否则 launch_engine.sh 路径启动会与 launch_static.sh 不一致。

## 九、重启 runbook（本套部署）

```bash
# 1. master（P0 new2，池化前置）
ssh root@10.222.42.240 'docker exec -d pd-mooncacke-p bash /pd/mooncake/run_master.sh'
#    探活: exec 3<>/dev/tcp/127.0.0.1/50088
# 2. decode（new8）
ssh root@10.222.41.88 'docker exec -d pd-mooncacke-d bash /pd/decode/launch_static.sh'
# 3. prefill（new2）
ssh root@10.222.42.240 'docker exec -d pd-mooncacke-p bash /pd/prefill/launch_static.sh'
# 4. proxy（new2）
ssh root@10.222.42.240 'docker exec -d pd-mooncacke-p bash /pd/run_proxy.sh'
# 探活: curl http://10.222.42.240:7100/health、http://10.222.41.88:7100/health、
#       http://10.222.42.240:1999/healthcheck
```
- 容器若被删（--rm）：按 `docker_start.sh` 重建 + 手工补 /workspace /models /pd 符号链接
  （或用 `container_mooncacke.sh` 建正规容器，注意其容器名是 iquest-pd-mooncacke-p/d）
- 杀引擎用 `pkill -f 'VLLM::|vllm serve'`（大写 proctitle 逃过小写 pkill），等 NPU 释放
- P 侧首启就绪 ~135s（SFS 页缓存热）；D 侧含 25 档图捕获 ~250s
