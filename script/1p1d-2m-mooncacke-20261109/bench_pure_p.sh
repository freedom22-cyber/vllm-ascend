#!/usr/bin/env bash
# 纯 P 容量(4.4 前置): 直连 P:7100, max_tokens=1, 无共享前缀, 默认 c8=P max_num_seqs
# 用法: bash bench_pure_p.sh [conc] [num] [label]
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"
CONC="${1:-8}"; NUM="${2:-200}"; LABEL="${3:-pureP_c${CONC}}"
mkdir -p "$DEPLOY_DIR/logs/bench"
ssh -o BatchMode=yes "root@${P0_IP}" "docker exec ${P0_CONTAINER} bash -c '\
/usr/local/python3.12.16/bin/vllm bench serve \
  --backend openai --base-url http://${P0_IP}:${PREFILL_API_PORT} --endpoint /v1/completions \
  --model ${SERVED_MODEL_NAME} --tokenizer ${MODEL_PATH} --trust-remote-code \
  --dataset-name random --random-input-len 110000 --random-prefix-len 99000 --random-output-len 1 \
  --num-prompts ${NUM} --max-concurrency ${CONC} --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir /pd/logs/bench --result-filename ${LABEL}.json'" \
  2>&1 | tee "${DEPLOY_DIR}/logs/bench/${LABEL}.log"
