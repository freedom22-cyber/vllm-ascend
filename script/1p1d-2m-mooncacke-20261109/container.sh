#!/usr/bin/env bash
# =====================================================================
# 在当前机器创建 PD 角色容器（idle，不起引擎）。
# 用法: bash container.sh [p|d]     （不带参数时按本机 IP 自动识别）
# 仓库与部署目录均挂自 SFS，两台机器一致。
# =====================================================================
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

ROLE="${1:-$(get_node)}"
case "$ROLE" in
    p) NAME="$P0_CONTAINER" ;;
    d) NAME="$D0_CONTAINER" ;;
    *) echo "[ERROR] 无法识别角色: 用 bash container.sh [p|d]" >&2; exit 1 ;;
esac

[ -d "$SFS_REPO/vllm_ascend" ] || {
    echo "[ERROR] 仓库树不在 SFS: $SFS_REPO"; echo "        先在 node11 上执行 bash sync.sh" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "[INFO] 容器 $NAME 已存在，跳过创建（重建请先: docker rm -f $NAME）"
    exit 0
fi

docker run -d --name "$NAME" \
  --privileged --ipc host --network host --shm-size 64g \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 \
  --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/hisi_hdc \
  --device=/dev/ummu --device=/dev/uburma \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/bin/urma_admin:/usr/bin/urma_admin:ro \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
  -v /etc/hixlep:/etc/hixlep:ro \
  -v /mnt/sfs_turbo/models/IQuest-Q1:/models/IQuest-Q1:ro \
  -v /mnt/sfs_turbo/models/IQuest-Q1-W8A8C8-mxfp8:/models/IQuest-Q1-W8A8C8-mxfp8:ro \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /lib/route.conf:/lib/route.conf:ro \
  -v /etc/cloud/topo:/etc/cloud/topo:ro \
  -v "$SFS_REPO":/workspace \
  -v "$DEPLOY_DIR":/pd \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0 \
  -e ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0 \
  -e ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0 \
  -e ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1 \
  -e ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5 \
  -e ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1 \
  -e ATB_COMPARE_TILING_EVERY_KERNEL=0 \
  -e ATB_SHARE_MEMORY_NAME_SUFFIX= \
  -e ATB_MATMUL_SHUFFLE_K_ENABLE=1 \
  -e LCCL_DETERMINISTIC=0 \
  -e LCCL_PARALLEL=0 \
  -e ATB_CXX_ABI=1 \
  -e SOC_VERSION=ascend950dt_9582 \
  -e TASK_QUEUE_ENABLE=1 \
  -e OMP_NUM_THREADS=1 \
  "$VLLM_IMAGE" sleep infinity

echo "[INFO] 容器 $NAME 已创建（${ROLE} 角色，idle）"
echo "       仓库: $SFS_REPO -> /workspace"
echo "       部署: $DEPLOY_DIR -> /pd"
