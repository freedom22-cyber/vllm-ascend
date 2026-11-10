#!/bin/bash
# 2026-11-08: 将 new5:1999 proxy 从 new5/new6(-prof) 切到 node7(39.253)/node6(44.20) PD n67
# 在 new5 宿主机执行; 容器 iquest-pd-p-prof 将 SFS -prof 目录挂为 /pd
echo "== BEFORE: all load_balance proxy procs (host view) =="
pgrep -af "python3.*load_balance_proxy" | grep -v swap_proxy || echo none
echo "== KILL all =="
for p in $(pgrep -f "python3.*load_balance_proxy"); do kill -9 "$p" 2>/dev/null && echo "killed $p"; done
sleep 2
echo "== PORT 1999 after kill =="
ss -tlnp | grep ':1999' || echo PORT-FREE
echo "== START new proxy in container (39.253 / 44.20) =="
mkdir -p /mnt/sfs_turbo/work/chensq/infer/iq-vllm-bf16/deploy/iquest-1p1d-2m-v0.30.0-mooncake-prof/logs/proxy
docker exec -d iquest-pd-p-prof /bin/bash /pd/start_proxy_n67.sh
sleep 4
echo "== AFTER: container proxy procs =="
docker exec iquest-pd-p-prof pgrep -af "python3.*load_balance_proxy" || echo "NO-PROC (check /pd/logs/proxy/proxy-repoint-n67.log)"
echo "== PORT 1999 after start =="
ss -tlnp | grep ':1999'
