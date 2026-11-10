#!/usr/bin/env bash
# =====================================================================
# P 侧引擎: IQuest-Q1 TP8 × DP1，MooncakeHybridConnector kv_producer
# serve 参数由 serve_cmd.sh 统一生成（build_prefill_args）；eager/图模式由
# env.sh 的 ENABLE_GRAPH/PREFILL_ENABLE_GRAPH（env.override.sh 可覆盖）决定。
# 由 start.sh 通过 docker exec -d 调用（部署目录挂载为 /pd）。
# 【2026-11-06 重建】原版在 SFS 迁移 rsync 事故中丢失，按 serve_cmd.sh 契约重建。
# =====================================================================
set -Eeuo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DEPLOY_DIR/env.sh"
source "$DEPLOY_DIR/serve_cmd.sh"
source "$DEPLOY_DIR/runtime_env.sh"

[ "$(get_node)" = "p0" ] || { echo "[ERROR] 本机不是 prefill 节点 ($P0_IP)，拒绝启动" >&2; exit 1; }
load_cann_env

export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_BUFFSIZE=400
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# mooncake 传输引擎 .so（libtransfer_engine 等）补进 LD_LIBRARY_PATH
PY_BIN=/usr/local/python3.12.16/bin/python3
[ -x "$PY_BIN" ] || PY_BIN=python3
MOONCAKE_SITE="$("$PY_BIN" -c 'import os, mooncake; print(os.path.dirname(mooncake.__file__))')" \
    || { echo "[ERROR] 容器内未找到 mooncake 包" >&2; exit 1; }
export LD_LIBRARY_PATH="${MOONCAKE_SITE}:${LD_LIBRARY_PATH:-}:/usr/local/lib/"

# KV 池化（5.4）: AscendStoreConnector/mooncake 后端读取 MOONCAKE_CONFIG_PATH；
# MOONCAKE_MASTER 覆盖 json 中的 master_server_address（两处保持一致）。
export MOONCAKE_CONFIG_PATH="${DEPLOY_DIR}/mooncake/mooncake.json"
export MOONCAKE_MASTER="${MOONCAKE_MASTER_HOST}:${MOONCAKE_MASTER_PORT}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-24GB}"
if [ "${ENABLE_KV_POOL:-0}" = "1" ] && [ ! -f "$MOONCAKE_CONFIG_PATH" ]; then
    echo "[ERROR] ENABLE_KV_POOL=1 但缺少 $MOONCAKE_CONFIG_PATH（且须先 start.sh start master 就绪）" >&2
    exit 1
fi

VLLM_BIN=/usr/local/python3.12.16/bin/vllm
[ -x "$VLLM_BIN" ] || VLLM_BIN=vllm

LOG_DIR="${LOG_ROOT}/prefill"; mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/engine.log"
echo "[INFO] prefill 引擎启动: $(date '+%F %T') 端口=${PREFILL_API_PORT} 日志=${LOG_FILE}"

# KV 预算说明: 128k 单序列 KV ≈ 5.9G/卡；eager 省下图池内存给 KV
build_prefill_args
exec "$VLLM_BIN" serve "$MODEL_PATH" \
    "${SERVE_ARGS[@]}" \
    > "$LOG_FILE" 2>&1
