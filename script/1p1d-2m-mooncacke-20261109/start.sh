#!/usr/bin/env bash
# =====================================================================
# IQuest-Q1 1P1D PD 分离编排（宿主机执行）
#
# 用法:
#   bash start.sh start              # 按本机 IP 自动识别角色并启动
#                                    #   p0: master(池化) + prefill + proxy   d0: decode
#   bash start.sh start decode       # 显式指定（在对应机器上执行才有效）
#   bash start.sh start master       # 仅 P0 且 ENABLE_KV_POOL=1 时启动 mooncake_master
#   bash start.sh start proxy        # 仅 P0
#   bash start.sh stop [target]      # 停本机服务（target: prefill|decode|proxy|master|auto）
#
# 启动顺序建议（池化=5.4）: master → decode → prefill → proxy
#                          非池化:          decode → prefill → proxy
# 前置: 两台机器各自 bash container.sh [p|d]（容器就位）
# =====================================================================
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

ACTION="${1:-}"; TARGET="${2:-auto}"
usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ -n "$ACTION" ] || usage

node="$(get_node)"
if [ "$TARGET" = "auto" ]; then
    [ "$node" = "unknown" ] && { echo "[ERROR] 本机 IP(${local_ip:-未获取}) 不在 P0=$P0_IP / D0=$D0_IP 中，请显式指定 target" >&2; exit 1; }
    TARGET="$node"
fi

stop_proxy() {
    docker exec "$P0_CONTAINER" pkill -f load_balance_proxy_server_example 2>/dev/null \
        && echo "[INFO] proxy 已停止" || echo "[INFO] proxy 未在运行"
}

stop_engine() {  # stop_engine <容器名>
    if ! docker ps --format '{{.Names}}' | grep -qx "$1"; then
        echo "[WARN] 容器 $1 不在运行"
        return 0
    fi
    # 优雅停引擎: TERM → 等 30s → 补 KILL 孤儿 worker → 确认 NPU 释放。
    # 不要用 `docker restart`：容器的 PID 1 是 sleep infinity，docker restart 会
    # 直接 SIGKILL 掉 vllm（worker 处于 R 态拒收 TERM），可能造成驱动显存泄漏
    # 甚至 TsdOpen 失败（node2 卡 1 楔死的教训）。
    echo "[INFO] 停止容器 $1 内的 vllm 引擎（TERM → 30s → KILL）..."
    docker exec "$1" bash -c "pkill -TERM -f 'vllm serve' 2>/dev/null" >/dev/null 2>&1 || true
    sleep 30
    # 注意: 模式不能裸写 'vllm' —— proxy（load_balance_proxy_server_example.py）的
    # 路径里也含 vllm，会被误杀。引擎进程特征: vllm serve / VLLM::Worker_TP* / VLLM::EngineCore
    docker exec "$1" bash -c "pkill -KILL -f 'VLLM::|vllm serve' 2>/dev/null" >/dev/null 2>&1 || true
    sleep 5
    local left
    # grep -c 无匹配时退出码为 1，会触发 set -e 杀掉本脚本（stop 后 start 不执行的元凶），
    # 因此 || true 放在容器内命令末尾
    left=$(docker exec "$1" bash -c "ps -eo args | grep -c '[v]llm serve' || true" 2>/dev/null | tr -d ' \r')
    if [ "${left:-0}" != "0" ]; then
        echo "[WARN] 容器 $1 内仍有 ${left} 个 vllm 进程，强制重启容器兜底..."
        docker restart -t 60 "$1" >/dev/null
        sleep 10
    fi
    # 确认 NPU 上的占卡进程已清零
    local idle
    idle=$(npu-smi info 2>/dev/null | grep -c 'No running processes' || echo 0)
    echo "[INFO] $1 引擎已停止（宿主空闲 NPU 数: ${idle}/8）"
    if [ "${idle:-0}" != "8" ]; then
        echo "[WARN] 仍有 NPU 被占用，可能是孤儿 worker 或前次任务的显存残留；" \
             "确认 npu-smi 后再拉起引擎（否则新引擎可能因显存不足起不来）" >&2
    fi
}

start_master() {  # mooncake_master 仅 P0；ENABLE_KV_POOL=0 时跳过
    if [ "${ENABLE_KV_POOL:-0}" != "1" ]; then
        echo "[INFO] ENABLE_KV_POOL=0，跳过 mooncake_master"; return 0
    fi
    [ "$(get_node)" = "p0" ] || { echo "[ERROR] mooncake_master 仅在 P0 (${P0_IP}) 上启动" >&2; exit 1; }
    # 已在运行则不重复拉起（重复实例会抢 master 端口）。
    # 用 TCP 探端口而不是 pgrep -f：pgrep -f 会匹配到它自己所在的 shell 命令行（假阳性）。
    if timeout 3 bash -c "exec 3<>/dev/tcp/${MOONCAKE_MASTER_HOST}/${MOONCAKE_MASTER_PORT}" 2>/dev/null; then
        echo "[INFO] mooncake_master 已在运行，跳过启动"
    else
        # 渲染 mooncake.json：保证 master 地址/段大小与 env.sh 一致
        # （容器内 /pd 与宿主机部署目录是同一份 SFS 文件，两侧引擎读到的是同一份）
        cat > "${DEPLOY_DIR}/mooncake/mooncake.json" <<EOF
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "${MOONCAKE_MASTER_HOST}:${MOONCAKE_MASTER_PORT}",
    "global_segment_size": "${MOONCAKE_GLOBAL_SEGMENT_SIZE}",
    "preferred_segment": false,
    "prefer_alloc_in_same_node": true
}
EOF
        echo "[INFO] 在容器 $P0_CONTAINER 启动 mooncake_master (${MOONCAKE_MASTER_HOST}:${MOONCAKE_MASTER_PORT})..."
        docker exec -d "$P0_CONTAINER" bash /pd/mooncake/run_master.sh
    fi
    for _i in $(seq 1 15); do
        docker exec "$P0_CONTAINER" bash -c "exec 3<>/dev/tcp/127.0.0.1/${MOONCAKE_MASTER_PORT}" 2>/dev/null \
            && { echo "[INFO] mooncake_master 端口就绪"; return 0; }
        sleep 1
    done
    echo "[ERROR] mooncake_master 15s 内未监听 ${MOONCAKE_MASTER_PORT}，查 ${LOG_ROOT}/mooncake/master.log" >&2
    return 1
}

stop_master() {
    # pkill 模式用 [m] 括号写法，避免匹配到承载 pkill 的 shell 自身命令行
    if docker exec "$P0_CONTAINER" pkill -f "[m]ooncake_master" 2>/dev/null; then
        echo "[INFO] mooncake_master 已停止"
    else
        echo "[INFO] mooncake_master 未在运行"
    fi
}

case "$ACTION" in
    start)
        case "$TARGET" in
            decode)
                [ "$node" = "d0" ] || [ "$TARGET" = "decode" ] || true
                echo "[INFO] 在容器 $D0_CONTAINER 启动 decode 引擎 (TP8, MTP=$ENABLE_MTP)..."
                docker exec -d "$D0_CONTAINER" bash /pd/decode/launch_engine.sh
                wait_http "http://${D0_IP}:${DECODE_API_PORT}/health" 2400 "decode 引擎"
                ;;
            prefill)
                echo "[INFO] 在容器 $P0_CONTAINER 启动 prefill 引擎 (TP8, eager)..."
                docker exec -d "$P0_CONTAINER" bash /pd/prefill/launch_engine.sh
                wait_http "http://${P0_IP}:${PREFILL_API_PORT}/health" 2400 "prefill 引擎"
                ;;
            proxy)
                echo "[INFO] 在容器 $P0_CONTAINER 启动 proxy (0.0.0.0:${PROXY_PORT})..."
                docker exec -d "$P0_CONTAINER" bash /pd/run_proxy.sh
                # proxy 进程 import vllm 需要数十秒；example proxy 的探活端点是
                # /healthcheck（/health 404）。3s 首查多半未就绪属正常，另行轮询。
                for _i in $(seq 1 30); do
                    curl -sf -m 2 "http://127.0.0.1:${PROXY_PORT}/healthcheck" >/dev/null \
                        && { echo "[INFO] proxy 就绪"; break; }
                    [ "$_i" = 30 ] && echo "[WARN] proxy 60s 内未响应，查 ${LOG_ROOT}/proxy/proxy.log"
                    sleep 2
                done
                ;;
            master)
                start_master
                ;;
            p0)
                echo "[INFO] ===== P0 侧启动: master(池化) + prefill + proxy ====="
                "$0" start master
                "$0" start prefill
                "$0" start proxy
                ;;
            d0)
                echo "[INFO] ===== D0 侧启动: decode ====="
                "$0" start decode
                ;;
            all)
                # 完整流程（池化顺序 master → decode → prefill → proxy）:
                # 需要能免密 ssh 到两台机器（在 P0 上执行）
                "$0" start master
                ssh "root@${D0_IP}" "bash ${SFS_DEPLOY}/start.sh start decode"
                "$0" start prefill
                "$0" start proxy
                echo "[INFO] 全部就绪。入口: http://${P0_IP}:${PROXY_PORT}"
                ;;
            *) usage ;;
        esac
        ;;
    stop)
        case "$TARGET" in
            proxy)   stop_proxy ;;
            master)  stop_master ;;
            prefill) stop_engine "$P0_CONTAINER" ;;
            decode)  stop_engine "$D0_CONTAINER" ;;
            p0)      stop_master; stop_proxy; stop_engine "$P0_CONTAINER" ;;
            d0)      stop_engine "$D0_CONTAINER" ;;
            auto)
                case "$node" in
                    p0) "$0" stop p0 ;;
                    d0) "$0" stop d0 ;;
                    *)  echo "[ERROR] 无法识别本机节点" >&2; exit 1 ;;
                esac
                ;;
            all)
                stop_master
                stop_proxy
                stop_engine "$P0_CONTAINER"
                ssh "root@${D0_IP}" "bash ${SFS_DEPLOY}/start.sh stop d0" || true
                ;;
            *) usage ;;
        esac
        ;;
    *) usage ;;
esac
