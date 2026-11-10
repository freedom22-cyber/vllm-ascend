#!/usr/bin/env bash
# =====================================================================
# profiling 沙盒吞吐压测（vllm bench serve，经 proxy 1999 闭环并发）。
# 用法: bash bench.sh <conc> [num_prompts] [label]
#   可从任意机器执行（自动 ssh 到 P0 后在 P 容器内跑）。
#   ISL/OSL 用环境变量覆盖: ISL=2048 OSL=512 bash bench.sh 32
#   验收口径（吞吐 x2 任务）: ISL=2048 OSL=512 ignore-eos，聚合输出 tok/s。
# =====================================================================
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

CONC="${1:?usage: bench.sh <conc> [num_prompts] [label]}"
NUM="${2:-$(( CONC * 4 > 32 ? CONC * 4 : 32 ))}"
LABEL="${3:-c${CONC}}"

mkdir -p "${LOG_ROOT}/bench"

ssh -o BatchMode=yes -o ConnectTimeout=5 "root@${P0_IP}" \
  "docker exec ${P0_CONTAINER} bash /pd/bench_inner.sh ${CONC} ${NUM} ${LABEL}" \
  2>&1 | tee "${LOG_ROOT}/bench/${LABEL}.log"
