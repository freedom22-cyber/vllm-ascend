# 由 start_pd_graph_mtp.sh pool 生成 —— Mooncake KV Cache Pool 池化
# 生成时间: 2026-11-06 23:11
# 本文件被 P/D 两侧容器内的 /pd/env.sh source（优先级高于 env.sh 默认值）。
ENABLE_KV_POOL=1
# 2026-11-07 按用户要求恢复 P 侧 HBM 前缀缓存（HBM 优先、池兜底）。
# 验证池 GET 时曾临时置 POOL_PREFILL_PREFIX_CACHING=0（HBM 全命中时池查找被设计性跳过）。
# 2026-11-07 01:2x 池 GET 验证中：临时关闭 P 侧 HBM 前缀缓存；验证已通过，文件改回 1。
# 注意：当前运行的 P 引擎仍是 --no-enable-prefix-caching（启动时读到 0），需重启 P 才生效。
POOL_PREFILL_PREFIX_CACHING=1
