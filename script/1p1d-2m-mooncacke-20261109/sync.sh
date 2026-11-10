#!/usr/bin/env bash
# =====================================================================
# 在 node11 上执行: 把本部署目录同步到 SFS（D0 侧从 SFS 运行），排除日志。
# 仓库树已随首装同步（rsync -a vllm-ascend → SFS_ROOT/），这里只做在位校验。
# =====================================================================
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

mkdir -p "$SFS_ROOT"
rsync -a --exclude logs --exclude .git "$DEPLOY_DIR"/ "$SFS_DEPLOY"/
echo "[INFO] 部署目录已同步 → ${SFS_DEPLOY}"

if [ -d "$SFS_REPO/vllm_ascend" ]; then
    rev=$(git -C "$SFS_REPO" rev-parse --short=9 HEAD 2>/dev/null || echo '?')
    echo "[INFO] 仓库树在位: ${SFS_REPO} @ ${rev}"
else
    echo "[ERROR] SFS 仓库缺失，执行: rsync -a ${DEPLOY_DIR%/*}/vllm-ascend ${SFS_ROOT}/" >&2
    exit 1
fi
