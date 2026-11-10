#!/usr/bin/env bash
# =====================================================================
# 容器侧 CANN/ATB 运行时环境加载
# docker exec 不继承镜像 ENTRYPOINT 的环境导出，必须自己 source。
# 顺序与镜像 ENTRYPOINT 一致: ascend-toolkit -> ascendnpu-ir -> nnal/atb。
# AICPU JIT 编译头文件: 漏掉会导致 AICPU 侧找不到编译器。
# =====================================================================
load_cann_env() {
    set +u
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
    if [ -f "/usr/local/Ascend/cann-${CANN_VERSION:-9.2.0}/share/info/ascendnpu-ir/bin/set_env.sh" ]; then
        source "/usr/local/Ascend/cann-${CANN_VERSION:-9.2.0}/share/info/ascendnpu-ir/bin/set_env.sh"
    fi
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
        source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi="${ATB_CXX_ABI:-1}"
    fi
    set -u

    local aicpu_headers=/usr/local/Ascend/cann-9.2.0/tools/hcc/aarch64-target-linux-gnu/include/c++/14.3.0
    export CPLUS_INCLUDE_PATH="${aicpu_headers}:${aicpu_headers}/aarch64-target-linux-gnu:${aicpu_headers}/backward"
}
