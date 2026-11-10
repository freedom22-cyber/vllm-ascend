#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 1P1D 双机 PD 分离（Mooncake 实时 KV 传输）—— 全部可调项集中于此
#
# 拓扑: 每台机器 1 个引擎进程 × TP8（IQuest-Q1 606G bf16，单卡放不下，必须 TP8）
#   P0 = 10.222.39.253 (node7/gw39_253)  prefill + proxy
#   D0 = 10.222.44.20 (node6/gw44_20)    decode（+ MTP 投机解码 + ACL graph）
#   【2026-11-06 第三次换节点】node2 的 1 号卡驱动楔死（杀引擎致驱动显存泄漏+
#   reset 被拒），P0 迁 node7（原 TP16 作业已按用户指示停止）。
#   历史换节点: 原 P0=10.222.43.87 / D0=10.222.44.62 → D0 曾改 node4(41.28，
#   被 TP16 占用) → P0=node2(44.62)/D0=node6(44.20) → P0=node7(39.253)。
#
# 【2026-11-06 重建说明】本文件的原版（含 GRAPH_*/EFFECTIVE_* 图模式段）在
# SFS 迁移 rsync 事故中被旧版覆盖且无备份；现按 serve_cmd.sh / start_pd_graph_mtp.sh
# 的变量契约重建。EFFECTIVE_* 的图模式默认值（GRAPH_MAX_MODEL_LEN 等）为重建值，
# 与原版可能有出入 —— 如需调整请写 env.override.sh（优先级最高），或直接改本段。
# =====================================================================

# ---------------- 模型 ----------------
# 两套权重分别测试: A=W8A8C8-mxfp8（量化，需 QUANT_ARGS）/ B=bf16（QUANT_ARGS 置空）
MODEL_PATH="/models/IQuest-Q1-W8A8C8-mxfp8"   # A 套（SFS: /mnt/sfs_turbo/models/...）
QUANT_ARGS="--quantization ascend"            # A 套用 --quantization ascend；B 套置空
DRAFT_MODEL_PATH="/models/IQuest-Q1/mtp"      # MTP 草案头（仅 D 侧；量化目录无 mtp/，复用 bf16 的）
SERVED_MODEL_NAME="IQuest-Q1"
ENABLE_MTP=1                                  # D 侧 MTP 开关；PD 跑通后若与 KV 传输冲突可置 0 排查
DECODE_SPEC_CONFIG='{"method":"eagle","model":"/models/IQuest-Q1/mtp","num_speculative_tokens":5}'

# ---------------- 节点 / 网卡 ----------------
# 【profiling 沙盒】独立于主部署（43.66/42.240）与 -opt 沙盒（node1/node6）。
# 本沙盒用 new5/new6（hyperinstance-00005/00006），引擎代码跑 worktree vllm-ascend-prof。
NIC_NAME="enp39s0f3"                          # 两台机器业务网卡同名（已核实）
P0_IP="10.222.42.240"                         # prefill 节点（new2/sjsbg-00002，2026-11-09 迁移：new7 被外部 4m 站占用）
D0_IP="10.222.41.88"                          # decode 节点（new8/sjsbg-00008）

# ---------------- 容器 ----------------
VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.30.0-a5-openeuler"
P0_CONTAINER="iquest-pd-p78"
D0_CONTAINER="iquest-pd-d78"
# 仓库树 = infer 工作区自身（2026-11-06 SFS 迁移后 /mnt/sfs_turbo/chensq/pd 已不存在）
# profiling 沙盒挂 worktree：优化改动在 vllm-ascend-prof 提交后重启引擎即生效
SFS_ROOT="/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16"
SFS_REPO="${SFS_ROOT}/vllm-ascend-prof"
# 部署目录已迁入 infer 工作区 deploy/（2026-11-06）；SFS_DEPLOY 即本目录自身，do_sync 自动跳过
SFS_DEPLOY="/mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/pd-ration/deploy"

# ---------------- 部署目录 / 日志 ----------------
# 宿主机上本目录；容器内统一挂载为 /pd
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${DEPLOY_DIR}/logs"

# ---------------- 端口 ----------------
PREFILL_API_PORT=7100
DECODE_API_PORT=7100                          # 两台不同机器，可同号
PREFILL_KV_PORT=28010                         # 基址 ≥28000，避开 AscendDirectTransport 随机段 [20000,28000)
DECODE_KV_PORT=29010
PROXY_PORT=1999                               # 对外统一入口（P0 上）

# ---------------- 上下文 / 调度 ----------------
MAX_MODEL_LEN=131072                          # 128k（config.json max_position_embeddings=524288，无需 rope 缩放）
BLOCK_SIZE=128
PREFILL_MAX_NUM_SEQS=32                            # M1-L-P0: 16→32(吞吐6000计划,P 供给解封)
DECODE_MAX_NUM_SEQS=256                           # M1-L2: 128→256(吞吐6000计划并发上探;capture 默认档自动到 1024)
PREFILL_MAX_NUM_BATCHED_TOKENS=8192                # M1-L-P0: 4096→8192(减 AR 切片次数,TTFT/供给)
DECODE_MAX_NUM_BATCHED_TOKENS=1024
# D 侧时延优化（2026-11-07，对应参考文档 5.4 D 侧配置）：
# async-scheduling 让 CPU 调度与 GPU 执行重叠（spec-decode 逐步采样的 Python 开销
# 不再串行暴露在关键路径）；additional-config 为 Ascend 配套（CPU 绑核、共享专家
# 多流重叠、scheduler 重计算、npugraph_ex）。效果不达标时逐项回退定位。
DECODE_ASYNC_SCHEDULING=1
DECODE_ADDITIONAL_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}'
PREFILL_GPU_MEM_UTIL=0.90
DECODE_GPU_MEM_UTIL=0.92

# ---------------- KV 实时传输连接器 ----------------
# MooncakeHybridConnector（本仓库注册名，kv_p2p.mooncake_connector.MooncakeConnector）
# dp_size=1 / tp_size=8 对应每侧单进程 TP8；字段名必须放 extra_config 内部
PD_EXTRA='{"prefill":{"dp_size":1,"tp_size":8},"decode":{"dp_size":1,"tp_size":8}}'
PREFILL_KV_CONFIG='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":'"${PREFILL_KV_PORT}"',"kv_connector_extra_config":'"${PD_EXTRA}"'}'
DECODE_KV_CONFIG='{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":'"${DECODE_KV_PORT}"',"kv_connector_extra_config":'"${PD_EXTRA}"'}'

# ---------------- Mooncake KV Cache Pool 池化（默认关；env.override.sh 可开）----------------
# 参考《5.4 使用 Mooncake KV Cache Pool 的 1P1D 部署》适配：
#   MultiConnector = MooncakeHybridConnector(实时 P→D) + AscendStoreConnector(池存储)
#   使用前置: 1) start.sh start master 启动 mooncake_master（仅 P0）
#             2) P/D 引擎进程读取 MOONCAKE_CONFIG_PATH（launch_engine.sh 自动导出）
#   注: IQuest-Q1 为 hybrid attention（full+sliding 混合层），池化固定 use_layerwise=false
#       （与文档中 DSV4.1-Flash 的 CED 结构限制相同），见下方生效配置段。
ENABLE_KV_POOL="${ENABLE_KV_POOL:-0}"          # 环境变量可带入；env.pool.override.sh 仍优先
MOONCAKE_MASTER_HOST="$P0_IP"                  # master 固定跑 P0（与 proxy 同机）
MOONCAKE_MASTER_PORT=50088
# default_kv_lease_ttl 需大于 ASCEND_CONNECT_TIMEOUT 与 ASCEND_TRANSFER_TIMEOUT
MOONCAKE_MASTER_ARGS="--port ${MOONCAKE_MASTER_PORT} --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000"
MOONCAKE_CONFIG_PATH="${DEPLOY_DIR}/mooncake/mooncake.json"   # 容器内即 /pd/mooncake/mooncake.json
MOONCAKE_GLOBAL_SEGMENT_SIZE="24GB"            # env 覆盖优先于 json（mooncake_backend.py 支持）
POOL_PREFILL_PREFIX_CACHING=1                  # 池化时 P 侧开前缀缓存（池查找依赖前缀匹配）；异常时可置 0

# ---------------- 图模式（PD 侧；由 start_pd_graph_mtp.sh enable 写 env.override.sh 开启）----------------
ENABLE_GRAPH=0                                # 图模式总开关
PREFILL_ENABLE_GRAPH=0                        # P 侧图模式（默认关: P 只做 prefill，eager 省图池显存）
DECODE_ENABLE_GRAPH=0                         # D 侧图模式
GRAPH_CUDAGRAPH_MODE=FULL_DECODE_ONLY         # 只对 decode 批捕获 ACL graph（与单机已验证配置一致）
GRAPH_CAPTURE_SIZES_JSON=""                   # 空=默认档位（MTP 下 41 档）；如 '[1,2,4,8,16]' 压缩图池
GRAPH_DECODE_GPU_MEM_UTIL=0.94                # 图模式下 D 侧 util（图池吃 ~14G/卡，需抬高回收显存）
GRAPH_MAX_MODEL_LEN=131072                     # 图模式下上下文（2026-11-07 从 32k 上调：A 套 KV 1.02-1.07M tokens/侧，128k 序列余 7.8-8.2x 并发；128k 全链 P→D 传输 eager 模式已实测）
GRAPH_DECODE_MAX_NUM_SEQS=256                 # 图模式下 D 侧并发(M1-L2: 128→256 配合 pin 32G;capture 档自动到 1024)

# ---- env.override.sh（由 start_pd_graph_mtp.sh enable 生成）最后 source，优先级最高 ----
if [ -f "$DEPLOY_DIR/env.override.sh" ]; then
    # shellcheck disable=SC1091
    source "$DEPLOY_DIR/env.override.sh"
fi
# ---- env.pool.override.sh（池化开关，独立于图模式开关）----
# 分开成两个文件的原因：start_pd_graph_mtp.sh enable/disable 会整份重写 env.override.sh，
# 若池化开关也放在里面，"开图" 会把已开的池化悄悄关掉。
if [ -f "$DEPLOY_DIR/env.pool.override.sh" ]; then
    # shellcheck disable=SC1091
    source "$DEPLOY_DIR/env.pool.override.sh"
fi

# ---------------- 生效值（EFFECTIVE_*：serve_cmd.sh 消费）----------------
if [ "${ENABLE_GRAPH:-0}" = "1" ] && [ "${DECODE_ENABLE_GRAPH:-0}" = "1" ]; then
    EFFECTIVE_DECODE_GRAPH=1
else
    EFFECTIVE_DECODE_GRAPH=0
fi
if [ "${ENABLE_GRAPH:-0}" = "1" ] && [ "${PREFILL_ENABLE_GRAPH:-0}" = "1" ]; then
    EFFECTIVE_PREFILL_GRAPH=1
else
    EFFECTIVE_PREFILL_GRAPH=0
fi
if [ "$EFFECTIVE_DECODE_GRAPH" = "1" ]; then
    EFFECTIVE_MAX_MODEL_LEN="$GRAPH_MAX_MODEL_LEN"
    EFFECTIVE_DECODE_MAX_NUM_SEQS="$GRAPH_DECODE_MAX_NUM_SEQS"
    EFFECTIVE_DECODE_GPU_MEM_UTIL="$GRAPH_DECODE_GPU_MEM_UTIL"
else
    EFFECTIVE_MAX_MODEL_LEN="$MAX_MODEL_LEN"
    EFFECTIVE_DECODE_MAX_NUM_SEQS="$DECODE_MAX_NUM_SEQS"
    EFFECTIVE_DECODE_GPU_MEM_UTIL="$DECODE_GPU_MEM_UTIL"
fi
if [ "$EFFECTIVE_PREFILL_GRAPH" = "1" ]; then
    EFFECTIVE_PREFILL_GPU_MEM_UTIL="$GRAPH_DECODE_GPU_MEM_UTIL"
else
    EFFECTIVE_PREFILL_GPU_MEM_UTIL="$PREFILL_GPU_MEM_UTIL"
fi

# ---------------- KV 池化生效配置（须在 env.override.sh 之后计算）----------------
# 把现有单连接器实时传输配置整体包成 MultiConnector 的第一个子连接器（逐字段不变），
# 追加 AscendStoreConnector 池存储子连接器。use_layerwise 固定 false（hybrid attention）。
if [ "${ENABLE_KV_POOL:-0}" = "1" ]; then
    _pool_store_json() {
        printf '{"kv_connector":"AscendStoreConnector","kv_role":"%s","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"mooncake","use_layerwise":false}}' "$1"
    }
    PREFILL_KV_CONFIG='{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_port":'"${PREFILL_KV_PORT}"',"engine_id":"0","kv_connector_extra_config":{"connectors":['"${PREFILL_KV_CONFIG}"','"$(_pool_store_json kv_producer)"']}}'
    DECODE_KV_CONFIG='{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_port":'"${DECODE_KV_PORT}"',"engine_id":"1","kv_connector_extra_config":{"connectors":['"${DECODE_KV_CONFIG}"','"$(_pool_store_json kv_consumer)"']}}'
    unset -f _pool_store_json
fi

# ---------------- 公共函数 ----------------
get_local_ip() {
    if [ -n "${local_ip:-}" ]; then printf '%s\n' "${local_ip}"; return 0; fi
    if command -v ip >/dev/null 2>&1; then
        ip -4 -o addr show dev "$NIC_NAME" scope global 2>/dev/null \
            | awk '{print $4}' | cut -d/ -f1 | head -n 1 && return 0
    fi
    # 容器内通常没有 ip 命令，python ioctl 兜底（与参考部署一致）
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$NIC_NAME" <<'PYEOF' 2>/dev/null
import fcntl, socket, struct, sys
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packed = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack('256s', sys.argv[1][:15].encode()))
    print(socket.inet_ntoa(packed[20:24]))
except OSError:
    sys.exit(1)
PYEOF
        return $?
    fi
    return 1
}

get_node() {
    case "$(get_local_ip 2>/dev/null)" in
        "$P0_IP") printf 'p0' ;;
        "$D0_IP") printf 'd0' ;;
        *)        printf 'unknown' ;;
    esac
}

get_container() {
    case "$(get_node)" in
        p0) printf '%s\n' "$P0_CONTAINER" ;;
        d0) printf '%s\n' "$D0_CONTAINER" ;;
        *)  return 1 ;;
    esac
}

wait_http() {  # wait_http <url> <timeout_sec> <名称>
    local url="$1" timeout_s="$2" name="$3" i
    echo "[INFO] 等待 ${name} 就绪: ${url} （最长 ${timeout_s}s，权重 606G 从 SFS 加载约 10 分钟+）"
    for ((i = 0; i < timeout_s / 5; i++)); do
        if curl -sf -o /dev/null "$url" 2>/dev/null; then
            echo "[INFO] ${name} 就绪 (耗时 ~$((i * 5))s)"; return 0
        fi
        sleep 5
    done
    echo "[ERROR] ${name} 在 ${timeout_s}s 内未就绪，请查日志: ${LOG_ROOT}" >&2
    return 1
}

local_ip="${local_ip:-$(get_local_ip 2>/dev/null || true)}"
