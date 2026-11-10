#!/usr/bin/env bash
# =====================================================================
# vllm serve 参数统一生成器（P/D 两侧共用）。
#
#   launch_engine.sh        —— 实际启动时调用，填充数组 SERVE_ARGS
#   start_pd_graph_mtp.sh   —— plan 打印时调用，保证「看到的命令 = 实际启动的命令」
#
# 依赖：env.sh 已 source（提供模型路径/端口/并行/MTP/图模式等变量）。
# 本文件只做参数拼装，不启动任何进程。
# =====================================================================

# 图模式 compilation-config JSON
#   GRAPH_CAPTURE_SIZES_JSON 为空  -> 只带 cudagraph_mode（与单机已验证配置一致：
#                                    FULL_DECODE_ONLY + MTP 捕获 41 档 decode 图）
#   非空（如 '[1,2,4,8,16]'）      -> 追加显式 capture sizes，压缩图池显存
#                                    （大小必须 ≤ 该侧 max_num_seqs）
_compilation_json() {
    if [ -n "${GRAPH_CAPTURE_SIZES_JSON:-}" ]; then
        printf '{"cudagraph_mode":"%s","cudagraph_capture_sizes":%s}' \
            "${GRAPH_CUDAGRAPH_MODE}" "${GRAPH_CAPTURE_SIZES_JSON}"
    else
        printf '{"cudagraph_mode":"%s"}' "${GRAPH_CUDAGRAPH_MODE}"
    fi
}

# ---------------------------------------------------------------------
# P 侧：prefill + Mooncake kv_producer
# ---------------------------------------------------------------------
build_prefill_args() {
    SERVE_ARGS=(
        --host "$P0_IP"
        --port "$PREFILL_API_PORT"
        --served-model-name "$SERVED_MODEL_NAME"
        --tensor-parallel-size 8
        --max-model-len "${PREFILL_MAX_MODEL_LEN:-$EFFECTIVE_MAX_MODEL_LEN}"
        --max-num-batched-tokens "$PREFILL_MAX_NUM_BATCHED_TOKENS"
        --max-num-seqs "$PREFILL_MAX_NUM_SEQS"
        --block-size "$BLOCK_SIZE"
        --trust-remote-code
        --reasoning-parser iquest_q1
        --tool-call-parser iquest_q1
        --enable-auto-tool-choice
        --gpu-memory-utilization "$EFFECTIVE_PREFILL_GPU_MEM_UTIL"
        --disable-hybrid-kv-cache-manager
        --kv-transfer-config "$PREFILL_KV_CONFIG"
    )
    # 池化模式（5.4）: P 侧开启前缀缓存以支持 AscendStoreConnector 的前缀池查找；
    # 非池化保持关闭（原已验证行为）。POOL_PREFILL_PREFIX_CACHING=0 可单独关掉。
    if [ "${ENABLE_KV_POOL:-0}" = "1" ] && [ "${POOL_PREFILL_PREFIX_CACHING:-1}" = "1" ]; then
        SERVE_ARGS+=(--enable-prefix-caching)
    else
        SERVE_ARGS+=(--no-enable-prefix-caching)
    fi
    # 图模式：P 侧默认仍走 eager（FULL_DECODE_ONLY 本身也只对 decode 批生效，
    # 但 P 侧开图会白吃图池显存）；需总开关 ENABLE_GRAPH=1 且本侧开关打开。
    if [ "${EFFECTIVE_PREFILL_GRAPH:-0}" = "1" ]; then
        SERVE_ARGS+=(--compilation-config "$(_compilation_json)")
    else
        SERVE_ARGS+=(--enforce-eager)
    fi
    # QUANT_ARGS 是一串多词参数（"--quantization ascend"），放进数组字面量里
    # 会被拆成两个元素。必须在数组内部追加，不能单独写成一条语句。
    # P 侧 additional-config（M3：P EP8+MC2 等）；"-" 表示消融。镜像 D 侧写法。
    if [ -n "${PREFILL_ADDITIONAL_CONFIG:-}" ] && [ "${PREFILL_ADDITIONAL_CONFIG:-}" != "-" ]; then
        SERVE_ARGS+=(--additional-config "$PREFILL_ADDITIONAL_CONFIG")
    fi
    # 通用透传（profiler 采集等）：PREFILL_EXTRA_ARGS 值内不能有空格（JSON 写紧凑形态）
    if [ -n "${PREFILL_EXTRA_ARGS:-}" ] && [ "${PREFILL_EXTRA_ARGS:-}" != "-" ]; then
        # shellcheck disable=SC2206
        SERVE_ARGS+=(${PREFILL_EXTRA_ARGS})
    fi
    SERVE_ARGS+=(${QUANT_ARGS:-})
    return 0
}

# ---------------------------------------------------------------------
# D 侧：decode + Mooncake kv_consumer + MTP 投机解码
# ---------------------------------------------------------------------
build_decode_args() {
    SERVE_ARGS=(
        --host "$D0_IP"
        --port "$DECODE_API_PORT"
        --served-model-name "$SERVED_MODEL_NAME"
        --tensor-parallel-size 8
        --max-model-len "$EFFECTIVE_MAX_MODEL_LEN"
        --max-num-batched-tokens "$DECODE_MAX_NUM_BATCHED_TOKENS"
        --max-num-seqs "$EFFECTIVE_DECODE_MAX_NUM_SEQS"
        --block-size "$BLOCK_SIZE"
        --no-enable-prefix-caching
        --trust-remote-code
        --reasoning-parser iquest_q1
        --tool-call-parser iquest_q1
        --enable-auto-tool-choice
        --gpu-memory-utilization "$EFFECTIVE_DECODE_GPU_MEM_UTIL"
        --disable-hybrid-kv-cache-manager
        --kv-transfer-config "$DECODE_KV_CONFIG"
    )
    if [ "${ENABLE_MTP:-0}" = "1" ]; then
        SERVE_ARGS+=(--speculative-config "$DECODE_SPEC_CONFIG")
    fi
    if [ "${EFFECTIVE_DECODE_GRAPH:-0}" = "1" ]; then
        SERVE_ARGS+=(--compilation-config "$(_compilation_json)")
    else
        SERVE_ARGS+=(--enforce-eager)
    fi
    # 见 build_prefill_args 的说明：必须放在数组内部追加
    # D 侧时延优化开关（env.sh：DECODE_ASYNC_SCHEDULING / DECODE_ADDITIONAL_CONFIG）
    if [ "${DECODE_ASYNC_SCHEDULING:-0}" = "1" ]; then
        SERVE_ARGS+=(--async-scheduling)
    fi
    # 消融开关：DECODE_ADDITIONAL_CONFIG="-" 表示本次启动不带 additional-config
    if [ -n "${DECODE_ADDITIONAL_CONFIG:-}" ] && [ "${DECODE_ADDITIONAL_CONFIG:-}" != "-" ]; then
        SERVE_ARGS+=(--additional-config "$DECODE_ADDITIONAL_CONFIG")
    fi
    # 通用透传（profiler 采集等）：DECODE_EXTRA_ARGS 值内不能有空格（JSON 写紧凑形态）
    if [ -n "${DECODE_EXTRA_ARGS:-}" ] && [ "${DECODE_EXTRA_ARGS:-}" != "-" ]; then
        # shellcheck disable=SC2206
        SERVE_ARGS+=(${DECODE_EXTRA_ARGS})
    fi
    SERVE_ARGS+=(${QUANT_ARGS:-})
    return 0
}

# ---------------------------------------------------------------------
# 打印（plan 用）：print_serve_cmd <prefill|decode> [vllm_bin]
# ---------------------------------------------------------------------
print_serve_cmd() {
    local role="${1:-}" bin="${2:-vllm}"
    case "$role" in
        prefill) build_prefill_args ;;
        decode)  build_decode_args ;;
        *) echo "[ERROR] 未知 role: ${role}" >&2; return 1 ;;
    esac
    printf '%s serve %s' "$bin" "$MODEL_PATH"
    printf ' %q' "${SERVE_ARGS[@]}"
    printf '\n'
}
