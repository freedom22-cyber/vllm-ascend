#!/usr/bin/env bash
# =====================================================================
# PD 混部 + 图模式(ACL graph) + MTP 一体化启动脚本
#
# 组合能力:
#   PD 混部 : 双机 1P1D PD 分离，Mooncake 实时 KV 传输（P=10.222.43.87 / D=10.222.44.62）
#   图模式  : D 侧 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
#              （decode 批走 ACL graph；P 侧只 prefill，默认仍 eager）
#   MTP     : D 侧 --speculative-config eagle，num_speculative_tokens=5
#
# 用法（宿主机执行，通常在 P0 上）:
#   bash start_pd_graph_mtp.sh plan      # 只打印将要生效的配置与 serve 命令（不启动）
#   bash start_pd_graph_mtp.sh enable    # 写入 env.override.sh（图+MTP 打开），不启动
#   bash start_pd_graph_mtp.sh disable   # 删除 env.override.sh，回到 eager PD 行为
#   bash start_pd_graph_mtp.sh pool      # 打开 KV 池化（env.pool.override.sh），不启动
#   bash start_pd_graph_mtp.sh nopool    # 关闭 KV 池化
#   bash start_pd_graph_mtp.sh start     # 同步 → master(池化时) → D0 decode → P0 prefill → proxy → 健康检查
#   bash start_pd_graph_mtp.sh stop      # 停 proxy / prefill / decode（池化时含 master）
#   bash start_pd_graph_mtp.sh status    # 健康检查（复用 check_pd_health.sh）
#
# 设计要点:
#   - 图/MTP 开关写进部署目录的 env.override.sh，P/D 两侧容器内的 /pd/env.sh
#     都会 source 到它 → 一处开关，两台机器同时生效，无需改两份配置。
#   - serve 参数由 serve_cmd.sh 统一生成，plan 打印 = 实际启动命令，不会漂移。
#   - ENABLE_GRAPH=0（默认）时行为与已验证的 PD 部署一字不差。
#
# 前置:
#   - 两台机器各有一个 PD 容器: bash container.sh [p|d]
#   - SFS 仓库树含图模式 + MTP 修复（uses_fia_v2 / actual_seq_lengths_q）
# =====================================================================
set -Eeuo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"
source "$DEPLOY_DIR/serve_cmd.sh"

ACTION="${1:-plan}"
OVERRIDE_FILE="${DEPLOY_DIR}/env.override.sh"
POOL_OVERRIDE_FILE="${DEPLOY_DIR}/env.pool.override.sh"

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '1,40p'; exit 1; }

log()  { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
err()  { echo "[ERROR] $*" >&2; }

# ---------------------------------------------------------------------
# pool / nopool: 打开/关闭 KV 池化（独立于 env.override.sh 的图模式开关）
# ---------------------------------------------------------------------
do_pool() {
    cat > "$POOL_OVERRIDE_FILE" <<EOF
# 由 start_pd_graph_mtp.sh pool 生成 —— Mooncake KV Cache Pool 池化
# 生成时间: $(date '+%F %T')
# 本文件被 P/D 两侧容器内的 /pd/env.sh source（优先级高于 env.sh 默认值）。
ENABLE_KV_POOL=1
EOF
    log "已写入池化开关: $POOL_OVERRIDE_FILE  (ENABLE_KV_POOL=1)"
    log "提示: 需先起 mooncake_master（bash start.sh start master），再起引擎"
}

do_nopool() {
    if [ -f "$POOL_OVERRIDE_FILE" ]; then
        rm -f "$POOL_OVERRIDE_FILE"
        log "已删除池化开关，回到仅实时 P→D 传输"
    else
        log "池化开关不存在，本来就是非池化模式"
    fi
}

# ---------------------------------------------------------------------
# enable: 写覆盖文件（只写配置，不启动任何东西）
# ---------------------------------------------------------------------
do_enable() {
    cat > "$OVERRIDE_FILE" <<EOF
# 由 start_pd_graph_mtp.sh enable 生成 —— PD 混部 + 图模式 + MTP
# 生成时间: $(date '+%F %T')
# 本文件被 P/D 两侧容器内的 /pd/env.sh 最后 source，优先级最高。
ENABLE_GRAPH=1
PREFILL_ENABLE_GRAPH=0
DECODE_ENABLE_GRAPH=1
ENABLE_MTP=1
EOF
    log "已写入覆盖文件: $OVERRIDE_FILE"
    log "  ENABLE_GRAPH=1  PREFILL_ENABLE_GRAPH=0  DECODE_ENABLE_GRAPH=1  ENABLE_MTP=1"
    log "提示: 图模式下的上下文/并发/显存见 env.sh 的 GRAPH_* 段（图池挤占 KV 预算）"
}

do_disable() {
    if [ -f "$OVERRIDE_FILE" ]; then
        rm -f "$OVERRIDE_FILE"
        log "已删除覆盖文件，回到 eager PD 行为"
    else
        log "覆盖文件不存在，本来就是 eager PD 行为"
    fi
}

# ---------------------------------------------------------------------
# plan: 打印将要生效的配置与命令（不启动）
# ---------------------------------------------------------------------
do_plan() {
    echo "===== 生效配置 ====="
    printf '  %-22s %s\n' "模型"              "$MODEL_PATH"
    printf '  %-22s %s\n' "PD 拓扑"            "P0=${P0_IP}:${PREFILL_API_PORT} → proxy:${PROXY_PORT} → D0=${D0_IP}:${DECODE_API_PORT}"
    if [ "${ENABLE_KV_POOL:-0}" = "1" ]; then
        printf '  %-22s %s\n' "KV 连接器"          "MultiConnector = MooncakeHybrid(实时P→D) + AscendStore(mooncake 池)"
    else
        printf '  %-22s %s\n' "KV 连接器"          "MooncakeHybridConnector (P=kv_producer, D=kv_consumer)"
    fi
    printf '  %-22s %s\n' "图模式总开关"       "ENABLE_GRAPH=${ENABLE_GRAPH:-0}"
    printf '  %-22s %s\n' "P 侧图模式"         "PREFILL_ENABLE_GRAPH=${PREFILL_ENABLE_GRAPH:-0} -> 生效=${EFFECTIVE_PREFILL_GRAPH:-0}"
    printf '  %-22s %s\n' "D 侧图模式"         "DECODE_ENABLE_GRAPH=${DECODE_ENABLE_GRAPH:-0} -> 生效=${EFFECTIVE_DECODE_GRAPH:-0} (${GRAPH_CUDAGRAPH_MODE:-FULL_DECODE_ONLY})"
    printf '  %-22s %s\n' "图档位"             "${GRAPH_CAPTURE_SIZES_JSON:-默认(与单机已验证配置一致)}"
    local mtp_desc="(关)"
    [ "${ENABLE_MTP:-0}" = "1" ] && mtp_desc="$DECODE_SPEC_CONFIG"
    printf '  %-22s %s\n' "MTP"                "ENABLE_MTP=${ENABLE_MTP:-0}  ${mtp_desc}"
    printf '  %-22s %s\n' "上下文(max-len)"    "P/D = ${EFFECTIVE_MAX_MODEL_LEN}"
    printf '  %-22s %s\n' "D 侧并发"           "${EFFECTIVE_DECODE_MAX_NUM_SEQS}"
    printf '  %-22s %s\n' "显存利用率"         "P=${EFFECTIVE_PREFILL_GPU_MEM_UTIL}  D=${EFFECTIVE_DECODE_GPU_MEM_UTIL}"
    printf '  %-22s %s\n' "KV 池化"            "ENABLE_KV_POOL=${ENABLE_KV_POOL:-0}"
    if [ "${ENABLE_KV_POOL:-0}" = "1" ]; then
        printf '  %-22s %s\n' "  mooncake_master" "${MOONCAKE_MASTER_HOST}:${MOONCAKE_MASTER_PORT}"
        printf '  %-22s %s\n' "  池段大小/卡"     "${MOONCAKE_GLOBAL_SEGMENT_SIZE}（每引擎进程）"
        printf '  %-22s %s\n' "  store 子连接器"  "AscendStoreConnector (backend=mooncake, use_layerwise=false)"
        printf '  %-22s %s\n' "  P 侧前缀缓存"    "${POOL_PREFILL_PREFIX_CACHING:-1}"
    fi

    echo
    echo "===== P 侧 serve 命令 ====="
    print_serve_cmd prefill
    echo
    echo "===== D 侧 serve 命令 ====="
    print_serve_cmd decode

    echo
    echo "===== 注意事项 ====="
    if [ "${ENABLE_GRAPH:-0}" = "1" ]; then
        echo "  * 图模式已开启: 图池吃显存，单机实测 util 0.94 时 KV 仅剩 0.11GiB。"
        echo "    因此上下文已下调到 ${EFFECTIVE_MAX_MODEL_LEN}（原 131072），并发 ${EFFECTIVE_DECODE_MAX_NUM_SEQS}。"
        echo "    如需更长上下文：调小 GRAPH_CAPTURE_SIZES_JSON（如 '[1,2,4,8]'）或调高 GRAPH_DECODE_GPU_MEM_UTIL。"
        echo "  * 图模式 + MTP 的互操作修复（uses_fia_v2 / actual_seq_lengths_q）必须在 SFS 仓库里，见 preflight 检查。"
    else
        echo "  * 图模式未开启（ENABLE_GRAPH=0）: 行为与已验证的 PD 部署完全一致。"
        echo "    执行 'bash start_pd_graph_mtp.sh enable' 可生成图+MTP 覆盖配置。"
    fi
    echo "  * 本命令不会启动任何服务。"
}

# ---------------------------------------------------------------------
# preflight: 启动前的静态检查（不占卡）
# ---------------------------------------------------------------------
do_preflight() {
    local rc=0
    echo "===== preflight ====="

    # 1. 覆盖文件
    if [ -f "$OVERRIDE_FILE" ]; then
        printf '  [OK]   覆盖文件存在: %s\n' "$OVERRIDE_FILE"
    else
        printf '  [WARN] 无覆盖文件 -> 当前是 eager PD 行为（start 前先跑 enable）\n'
    fi

    # 2. SFS 仓库树里的图模式 + MTP 修复
    if [ -d "$SFS_REPO/vllm_ascend" ]; then
        local rev
        rev=$(git -C "$SFS_REPO" rev-parse --short=9 HEAD 2>/dev/null || echo '?')
        printf '  [OK]   SFS 仓库: %s @ %s\n' "$SFS_REPO" "$rev"
        if grep -rqs "uses_fia_v2" "$SFS_REPO/vllm_ascend/"; then
            printf '  [OK]   修复1 uses_fia_v2 存在（草稿回放 FIA v2 键名）\n'
        else
            printf '  [FAIL] 修复1 uses_fia_v2 缺失 —— 图模式 + MTP 会失败\n'; rc=1
        fi
        if grep -rqs "actual_seq_lengths_q" "$SFS_REPO/vllm_ascend/models/iquest_q1/"; then
            printf '  [OK]   修复2 actual_seq_lengths_q 存在（MTP 多 token 累计长度）\n'
        else
            printf '  [FAIL] 修复2 actual_seq_lengths_q 缺失 —— 图模式 + MTP 会失败\n'; rc=1
        fi
    else
        printf '  [FAIL] SFS 仓库缺失: %s\n' "$SFS_REPO"; rc=1
    fi

    # 3. 权重与 MTP 草案头
    if [ -d /mnt/sfs_turbo/models/IQuest-Q1 ]; then
        printf '  [OK]   权重目录存在: /mnt/sfs_turbo/models/IQuest-Q1\n'
    else
        printf '  [FAIL] 权重目录缺失: /mnt/sfs_turbo/models/IQuest-Q1\n'; rc=1
    fi
    if [ "${ENABLE_MTP:-0}" = "1" ]; then
        if [ -d /mnt/sfs_turbo/models/IQuest-Q1/mtp ]; then
            printf '  [OK]   MTP 草案头存在: /mnt/sfs_turbo/models/IQuest-Q1/mtp\n'
        else
            printf '  [FAIL] MTP 草案头缺失: /mnt/sfs_turbo/models/IQuest-Q1/mtp\n'; rc=1
        fi
    fi

    # 4. 两台机器的容器
    for pair in "${P0_IP}:${P0_CONTAINER}:prefill" "${D0_IP}:${D0_CONTAINER}:decode"; do
        local ip="${pair%%:*}"
        local rest="${pair#*:}"
        local ct="${rest%%:*}"
        local role="${rest#*:}"
        local st
        st=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "root@${ip}" \
            "docker ps -a --format '{{.Names}} {{.Status}}' | grep -F '${ct}' || true" 2>/dev/null | tr -d '\r')
        if [ -n "$st" ]; then
            printf '  [OK]   %s 容器 @ %s: %s\n' "$role" "$ip" "$st"
        else
            printf '  [FAIL] %s 容器 @ %s 不存在（先跑 bash container.sh [p|d]）\n' "$role" "$ip"; rc=1
        fi
    done

    # 5. KV 池化前提（仅 ENABLE_KV_POOL=1 时检查）
    if [ "${ENABLE_KV_POOL:-0}" = "1" ]; then
        if [ -f "$MOONCAKE_CONFIG_PATH" ]; then
            local maddr
            maddr=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('master_server_address',''))" \
                "$MOONCAKE_CONFIG_PATH" 2>/dev/null || echo "")
            if [ "$maddr" = "${MOONCAKE_MASTER_HOST}:${MOONCAKE_MASTER_PORT}" ]; then
                printf '  [OK]   mooncake.json: %s (master=%s)\n' "$MOONCAKE_CONFIG_PATH" "$maddr"
            else
                printf '  [WARN] mooncake.json 的 master_server_address=%s，与 %s 不一致（start master 时会自动重渲染）\n' \
                    "${maddr:-<解析失败>}" "${MOONCAKE_MASTER_HOST}:${MOONCAKE_MASTER_PORT}"
            fi
        else
            printf '  [FAIL] 缺少 mooncake.json: %s\n' "$MOONCAKE_CONFIG_PATH"; rc=1
        fi
        local has_bin
        has_bin=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "root@${P0_IP}" \
            "docker exec ${P0_CONTAINER} bash -c 'command -v mooncake_master' 2>/dev/null" 2>/dev/null | tr -d '\r')
        if [ -n "$has_bin" ]; then
            printf '  [OK]   P0 容器内 mooncake_master: %s\n' "$has_bin"
        else
            printf '  [FAIL] P0 容器内未找到 mooncake_master\n'; rc=1
        fi
        local mrun
        # 探 TCP 端口而不是 pgrep -f：后者会匹配到自身所在的 shell 命令行（假阳性）
        mrun=$(timeout 3 bash -c "exec 3<>/dev/tcp/${MOONCAKE_MASTER_HOST}/${MOONCAKE_MASTER_PORT}" 2>/dev/null \
            && echo running || echo down)
        if [ "$mrun" = "running" ]; then
            printf '  [OK]   mooncake_master 已在运行\n'
        else
            printf '  [WARN] mooncake_master 未运行 —— start 流程会先拉起它（引擎初始化需连上 master）\n'
        fi
    fi

    echo "  preflight 结束: $([ $rc -eq 0 ] && echo PASS || echo FAIL)"
    return $rc
}

# ---------------------------------------------------------------------
# sync: 把部署目录同步到 SFS（D0 从 SFS 运行），必要时同步到 D0 容器 /pd 真实来源
# ---------------------------------------------------------------------
do_sync() {
    if [ "$(cd "$DEPLOY_DIR" 2>/dev/null && pwd)" = "$(cd "$SFS_DEPLOY" 2>/dev/null && pwd)" ]; then
        log "部署目录已在目标位置（${SFS_DEPLOY}），跳过同步"
        return 0
    fi
    log "同步部署目录 → ${SFS_DEPLOY}"
    mkdir -p "$SFS_ROOT"
    rsync -a --exclude logs --exclude .git "$DEPLOY_DIR"/ "$SFS_DEPLOY"/ || return 1

    # D0 容器内 /pd 的宿主机来源（可能不是 SFS 副本）
    local d0_pd_src
    d0_pd_src=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "root@${D0_IP}" \
        "docker inspect -f '{{range .Mounts}}{{if eq .Destination \"/pd\"}}{{.Source}}{{end}}{{end}}' ${D0_CONTAINER} 2>/dev/null" \
        2>/dev/null | tr -d '\r')
    if [ -n "$d0_pd_src" ] && [ "$d0_pd_src" != "$SFS_DEPLOY" ]; then
        log "D0 容器 /pd 实际来自 ${d0_pd_src}（非 SFS 副本），额外同步一份"
        rsync -a --exclude logs --exclude .git "$DEPLOY_DIR"/ "${d0_pd_src}/" || return 1
    elif [ -z "$d0_pd_src" ]; then
        warn "无法探测 D0 容器 /pd 来源（ssh 或容器不可用），若 D0 不是从 SFS 副本运行需手动同步"
    else
        log "D0 容器 /pd 来自 SFS 副本，一致"
    fi
    log "同步完成"
}

# ---------------------------------------------------------------------
# NPU 占用检查（别人在用卡时防止误启动）
#   判据: npu-smi 空闲卡数 != 8，或容器内已有 vllm 进程
# ---------------------------------------------------------------------
npu_busy_nodes() {
    local busy=""
    for pair in "${P0_IP}:${P0_CONTAINER}" "${D0_IP}:${D0_CONTAINER}"; do
        local ip="${pair%%:*}"
        local ct="${pair#*:}"
        local out n idle vproc
        out=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "root@${ip}" \
            "npu-smi info 2>/dev/null | grep -c 'No running processes' || echo 0; \
             docker exec ${ct} sh -c 'ps -eo comm 2>/dev/null | grep -ci vllm || echo 0' 2>/dev/null || echo 0" \
            2>/dev/null | tr -d '\r')
        idle=$(printf '%s\n' "$out" | sed -n '1p')
        vproc=$(printf '%s\n' "$out" | sed -n '2p')
        idle="${idle:-0}"; vproc="${vproc:-0}"
        if [ "$idle" != "8" ] || [ "$vproc" != "0" ]; then
            busy="${busy} ${ip}(空闲卡=${idle}/8, 容器内vllm进程=${vproc})"
        fi
    done
    printf '%s' "$busy"
}

# ---------------------------------------------------------------------
# start: 同步 → D0 decode → P0 prefill → proxy → 健康检查
# ---------------------------------------------------------------------
do_start() {
    # 本脚本的存在意义就是「PD + 图 + MTP」，没有覆盖文件时自动启用（图+MTP）。
    if [ ! -f "$OVERRIDE_FILE" ]; then
        log "未发现覆盖文件，自动启用图模式 + MTP"
        do_enable
    fi

    echo "===== preflight ====="
    do_preflight || { err "preflight 未通过，放弃启动"; exit 1; }

    local busy
    busy=$(npu_busy_nodes)
    if [ -n "$busy" ]; then
        warn "以下节点 NPU 非全空闲:${busy}"
        if [ "${FORCE:-0}" != "1" ]; then
            err "别人在用卡。确认可以抢占后重跑: FORCE=1 bash start_pd_graph_mtp.sh start"
            exit 1
        fi
        warn "FORCE=1，继续启动（将抢占已占用的卡）"
    fi

    log "同步部署目录（P0 与 D0 生效同一份配置）"
    do_sync

    log "启动顺序: master(池化时) → D0 decode → P0 prefill → proxy"
    bash "$DEPLOY_DIR/start.sh" start all

    echo
    echo "===== 健康检查 ====="
    bash "$DEPLOY_DIR/check_pd_health.sh" || true
    echo
    log "入口: http://${P0_IP}:${PROXY_PORT}"
    log "日志: ${LOG_ROOT}/{decode,prefill,proxy}/"
}

case "$ACTION" in
    plan)      do_plan ;;
    enable)    do_enable ;;
    disable)   do_disable ;;
    pool)      do_pool ;;
    nopool)    do_nopool ;;
    preflight) do_preflight ;;
    sync)      do_sync ;;
    start)     do_start ;;
    stop)      bash "$DEPLOY_DIR/start.sh" stop all ;;
    status)    bash "$DEPLOY_DIR/check_pd_health.sh" ;;
    *)         usage ;;
esac
