#!/usr/bin/env bash
# =====================================================================
# 容器内执行：经 proxy 的吞吐压测（由 bench.sh 通过 ssh+docker exec 调用）。
# 用法: bash /pd/bench_inner.sh <conc> <num_prompts> <label>
# =====================================================================
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

CONC="${1:?conc}"; NUM="${2:?num_prompts}"; LABEL="${3:-c${CONC}}"
IN="${ISL:-2048}"
OUT="${OSL:-512}"
PFX="${RANDOM_PREFIX_LEN:-0}"
OUTDIR="/pd/logs/bench"
mkdir -p "$OUTDIR"

exec /usr/local/python3.12.16/bin/vllm bench serve \
  --backend openai \
  --base-url "http://${P0_IP}:${PROXY_PORT}" \
  --endpoint /v1/completions \
  --model "$SERVED_MODEL_NAME" \
  --tokenizer "$MODEL_PATH" \
  --trust-remote-code \
  --dataset-name random \
  --random-prefix-len "$PFX" \
  --random-input-len "$IN" \
  --random-output-len "$OUT" \
  --num-prompts "$NUM" \
  --max-concurrency "$CONC" \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --save-result --result-dir "$OUTDIR" --result-filename "${LABEL}.json"
