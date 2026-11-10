#!/usr/bin/env bash
# =====================================================================
# profiling 沙盒: P/D 双侧同步采集 torch_npu profiler。
# 用法: bash collect_profile.sh <conc> [num_prompts] [label]
#   流程: start_profile(P+D) → 压测 → stop_profile(P+D) → 等 trace 落盘(SFS)
#   前置: 引擎须带 --profiler-config 启动（env.override.sh 已配）。
#   产物: $DEPLOY_DIR/logs/profile/<双侧 rank 目录> + bench json
# =====================================================================
set -Eeuo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

CONC="${1:-8}"
NUM="${2:-$(( CONC * 4 > 32 ? CONC * 4 : 32 ))}"
LABEL="${3:-prof_c${CONC}}"
STAMP="$(date '+%Y%m%d_%H%M%S')"

start_profile() {  # start_profile <ip> <container> <port> <name>
    local code
    code=$(ssh -o BatchMode=yes "root@$1" "curl -s -o /dev/null -w '%{http_code}' -m 15 -X POST http://$1:$3/start_profile" 2>/dev/null || echo ERR)
    echo "[INFO] $4 start_profile -> $code"
    [ "$code" = "200" ] || { echo "[ERROR] $4 start_profile 失败（$code）——端点未注册?" >&2; return 1; }
}

stop_profile() {  # stop_profile <ip> <port> <name>
    local code
    code=$(ssh -o BatchMode=yes "root@$1" "curl -s -o /dev/null -w '%{http_code}' -m 120 -X POST http://$1:$2/stop_profile" 2>/dev/null || echo ERR)
    echo "[INFO] $3 stop_profile -> $code"
}

echo "=== profile 采集 ${STAMP} label=${LABEL} conc=${CONC} num=${NUM} ==="
echo "[INFO] 1) start_profile P(${P0_IP}) + D(${D0_IP})"
start_profile "$P0_IP" "$P0_CONTAINER" "$PREFILL_API_PORT" "P"
start_profile "$D0_IP" "$D0_CONTAINER" "$DECODE_API_PORT" "D"

echo "[INFO] 2) 负载（经 proxy）"
ISL="${PROF_ISL:-2048}" OSL="${PROF_OSL:-512}" \
  bash "$DEPLOY_DIR/bench.sh" "$CONC" "$NUM" "${LABEL}"

echo "[INFO] 3) stop_profile 双侧"
stop_profile "$P0_IP" "$PREFILL_API_PORT" "P"
stop_profile "$D0_IP" "$DECODE_API_PORT" "D"

echo "[INFO] 4) 等 trace 落盘（120s）"
sleep 120
echo "=== 产物 ==="
timeout 30 ssh -o BatchMode=yes "root@${P0_IP}" "ls -d ${DEPLOY_DIR}/logs/profile/* 2>/dev/null | head -20"
timeout 30 ssh -o BatchMode=yes "root@${D0_IP}" "ls -d ${DEPLOY_DIR}/logs/profile/* 2>/dev/null | head -20"
echo "[INFO] 完成。原始数据在 SFS: ${DEPLOY_DIR}/logs/profile/"
