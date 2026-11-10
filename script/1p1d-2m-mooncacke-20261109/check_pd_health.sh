#!/usr/bin/env bash
# =====================================================================
# PD 部署健康检查（宿主机执行，P0 上最全）
# =====================================================================
set -uo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

ok() { printf '  [%s] %s\n' "$2" "$1"; }

echo "===== 端点 ====="
code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://${P0_IP}:${PREFILL_API_PORT}/health" 2>/dev/null)
ok "prefill  http://${P0_IP}:${PREFILL_API_PORT}  -> ${code}" "$([ "$code" = 200 ] && echo OK || echo DOWN)"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://${D0_IP}:${DECODE_API_PORT}/health" 2>/dev/null)
ok "decode   http://${D0_IP}:${DECODE_API_PORT}  -> ${code}" "$([ "$code" = 200 ] && echo OK || echo DOWN)"
# example proxy 的探活端点是 /healthcheck（/health 为 404）
ph=$(curl -s -m 5 "http://${P0_IP}:${PROXY_PORT}/healthcheck" 2>/dev/null)
ok "proxy    http://${P0_IP}:${PROXY_PORT}  -> ${ph:-(无响应)}" "$([ -n "$ph" ] && echo OK || echo DOWN)"
# 池化模式: mooncake_master（仅 P0 上有进程）
# 注意: 不能用 `pgrep -f 'mooncake_master --port'` —— 它会匹配到 pgrep 自身所在的
# shell 命令行（假阳性）。改为直接探 TCP 端口（容器是 --network host，端口在宿主机可见）。
if [ "${ENABLE_KV_POOL:-0}" = "1" ]; then
    mh=$(timeout 3 bash -c "exec 3<>/dev/tcp/${MOONCAKE_MASTER_HOST}/${MOONCAKE_MASTER_PORT}" 2>/dev/null \
        && echo running || echo down)
    ok "master   ${MOONCAKE_MASTER_HOST}:${MOONCAKE_MASTER_PORT}  -> ${mh}" "$([ "$mh" = running ] && echo OK || echo DOWN)"
fi

echo "===== 容器 / NPU ====="
for pair in "${P0_IP}:${P0_CONTAINER}:prefill" "${D0_IP}:${D0_CONTAINER}:decode"; do
    ip="${pair%%:*}"; rest="${pair#*:}"; ct="${rest%%:*}"; role="${rest#*:}"
    echo "-- ${role} @ ${ip} --"
    ssh -o ConnectTimeout=5 -o BatchMode=yes "root@${ip}" \
        "docker ps --format '{{.Names}} {{.Status}}' | grep -F '${ct}' || echo '容器未运行';
         npu-smi info | grep -c 'No running processes' | xargs -I{} echo '空闲 NPU 数: {}'" 2>&1
done

echo "===== 日志尾部（最近错误） ====="
tail -5 "${LOG_ROOT}/prefill/engine.log" 2>/dev/null | sed 's/^/  [P] /'
tail -5 "${LOG_ROOT}/decode/engine.log" 2>/dev/null | sed 's/^/  [D] /'
tail -5 "${LOG_ROOT}/proxy/proxy.log" 2>/dev/null | sed 's/^/  [X] /'
