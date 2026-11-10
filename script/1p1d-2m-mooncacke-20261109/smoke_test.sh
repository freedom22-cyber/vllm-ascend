#!/usr/bin/env bash
# =====================================================================
# 端到端冒烟测试（走 proxy 入口）
#   bash smoke_test.sh          # 短问答
#   LONG=1 bash smoke_test.sh   # 追加 ~100k token 长上下文用例（验证 128k）
# =====================================================================
set -uo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DEPLOY_DIR/env.sh"

# 默认走 PD proxy 入口；单机（混部）服务用 BASE_URL 覆盖，例如：
#   BASE_URL=http://10.222.39.33:8002 bash smoke_test.sh
BASE_URL="${BASE_URL:-http://${P0_IP}:${PROXY_PORT}}"

chat() {  # chat <messages_json> [max_tokens]
    curl -s -m 600 "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"
}

show() {  # show <响应json>
    python3 - "$1" <<'PYEOF'
import json, sys
try:
    r = json.loads(sys.argv[1])
    m = r["choices"][0]["message"]
    # vllm 0.30 的 OpenAI 兼容层把思维链放在 reasoning 字段；
    # 老版本/PD 代理侧可能是 reasoning_content，两个都取。
    rc = m.get("reasoning_content") or m.get("reasoning") or ""
    c = m.get("content") or ""
    u = r.get("usage") or {}
    if rc: print("[reasoning]", rc[:200].replace("\n", " "))
    print("[content]", c[:500])
    print(f"[usage] prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")
except Exception as e:
    print("[ERROR] 非法响应:", sys.argv[1][:400], e)
PYEOF
}

echo "===== 用例 1: 短问答（P prefill → KV 传输 → D decode） ====="
resp=$(chat '{"model":"'"${SERVED_MODEL_NAME}"'","messages":[{"role":"user","content":"用一句话介绍你自己。"}],"max_tokens":128,"temperature":0}')
show "$resp"

echo
echo "===== 用例 2: 一次完整生成（验证 D 侧持续解码） ====="
resp=$(chat '{"model":"'"${SERVED_MODEL_NAME}"'","messages":[{"role":"user","content":"从 1 数到 10，用中文顿号分隔，只输出数字。"}],"max_tokens":64,"temperature":0}')
show "$resp"

if [ "${LONG:-0}" = "1" ]; then
    echo
    echo "===== 用例 3: 长上下文（~100k token，验证 128k 能力） ====="
    prompt=$(python3 - <<'PYEOF'
para = ("第{i}段：研究团队在高原地区部署了新型气象观测阵列，"
        "持续记录温度、湿度与辐照度的分钟级变化，口令是{code}。")
lines = [para.format(i=i, code=f"ALPHA-{i:04d}") for i in range(1, 501)]
lines[136] = "第137段：本段包含目标口令 OMEGA-7741，请记住它。"
print("\n".join(lines) + "\n\n问题：只输出第137段中的口令，不要输出其他内容。")
PYEOF
)
    req=$(python3 - "$prompt" <<'PYEOF'
import json, sys
print(json.dumps({"model": "IQuest-Q1",
                  "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": 64, "temperature": 0}, ensure_ascii=False))
PYEOF
)
    resp=$(chat "$req")
    show "$resp"
    echo "$resp" | grep -q "OMEGA-7741" && echo "[PASS] 长上下文口令命中" || echo "[WARN] 口令未命中，检查内容如上"
fi
