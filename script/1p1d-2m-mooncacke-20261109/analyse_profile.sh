#!/usr/bin/env bash
# =====================================================================
# 容器内执行: 对 /pd/logs/profile 下最新的 *_ascend_pt 目录跑 torch_npu analyse
# （由宿主机侧 docker exec 调用）。产物: kernel_details.csv / trace_view.json 等。
# 用法: bash /pd/analyse_profile.sh [rank子串，如 rank0]
# =====================================================================
set -Eeuo pipefail
SUBSTR="${1:-rank0}"
PY_BIN=/usr/local/python3.12.16/bin/python3
[ -x "$PY_BIN" ] || PY_BIN=python3

"$PY_BIN" - "$SUBSTR" <<'PYEOF'
import glob, os, sys
from torch_npu.profiler.profiler import analyse

sub = sys.argv[1]
base = "/pd/logs/profile"
dirs = sorted(glob.glob(f"{base}/*{sub}*_ascend_pt"))
if not dirs:
    print(f"NO_PROFILE_DIR matching {sub} under {base}")
    print("have:", glob.glob(f"{base}/*"))
    raise SystemExit(1)
d = dirs[-1]  # 最新
csv_out = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
if os.path.exists(csv_out) and os.path.getsize(csv_out) > 0:
    print(f"ALREADY_ANALYSED: {d}")
else:
    print(f"analysing {d} ...")
    analyse(d)
    print("ANALYSE_DONE", d)
PYEOF
