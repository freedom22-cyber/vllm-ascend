#!/usr/bin/env bash
# 纯 D 容量(4.4): 走 proxy 开环 RR, 99% 共享前缀(2028/2048), OSL 512, 并发上限=decode max_num_seqs
# 用法: bash bench_pure_d.sh <RR> [cap] [num] [label]   TMO=秒 可覆盖(默认480)
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"
RR="${1:?usage: bench_pure_d.sh <RR> [cap] [num] [label]}"; CAP="${2:-6}"; NUM="${3:-$(( ${RR%.*} * 120 + 60 ))}"; LABEL="${4:-pureD_rr${RR}}"
TMO="${TMO:-480}"
mkdir -p "$DEPLOY_DIR/logs/bench"
ssh -o BatchMode=yes "root@${P0_IP}" "docker exec ${P0_CONTAINER} bash -c '\
timeout ${TMO} /usr/local/python3.12.16/bin/vllm bench serve \
  --backend openai --base-url http://${P0_IP}:${PROXY_PORT} --endpoint /v1/completions \
  --model ${SERVED_MODEL_NAME} --tokenizer ${MODEL_PATH} --trust-remote-code \
  --dataset-name random --random-input-len 110000 --random-prefix-len 108900 --random-output-len 1024 \
  --num-prompts ${NUM} --max-concurrency ${CAP} --request-rate ${RR} --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir /pd/logs/bench --result-filename ${LABEL}.json'" \
  2>&1 | tee "${DEPLOY_DIR}/logs/bench/${LABEL}.log"
