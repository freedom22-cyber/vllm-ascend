#!/bin/bash
# PD n67 proxy 启动脚本(容器内 /pd/start_proxy_n67.sh, 由 swap_proxy_n67.sh docker exec -d 调起)
# 目标: prefill=node7 10.222.39.253:7100, decode=node6 10.222.44.20:7100
exec /usr/local/python3.12.16/bin/python3 /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py --host 0.0.0.0 --port 1999 --prefiller-hosts 10.222.39.253 --prefiller-ports 7100 --decoder-hosts 10.222.44.20 --decoder-ports 7100 >> /pd/logs/proxy/proxy-repoint-n67.log 2>&1
