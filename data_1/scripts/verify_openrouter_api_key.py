import os
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding='utf-8')

key = os.getenv("OPENROUTER_API_KEY", "").strip()
print("=" * 80, flush=True)
print(" TESTING OPENROUTER API KEY & ACCOUNT BALANCE", flush=True)
print("=" * 80, flush=True)

if not key:
    print("❌ ERROR: OPENROUTER_API_KEY environment variable is missing!", flush=True)
    print("Run: export OPENROUTER_API_KEY=\"sk-or-v1-...\"", flush=True)
    sys.exit(1)

auth_req = urllib.request.Request(
    "https://openrouter.ai/api/v1/auth/key",
    headers={"Authorization": f"Bearer {key}"}
)

try:
    with urllib.request.urlopen(auth_req, timeout=10) as resp:
        info = json.loads(resp.read().decode("utf-8"))
        data = info.get("data", {})
        label = data.get("label", "Key")
        usage = data.get("usage", 0)
        limit = data.get("limit", 0)
        print("  Key Authenticated Successfully!", flush=True)
        print(f"   • Label: {label}", flush=True)
        print(f"   • Usage: ${usage:.4f} USD", flush=True)
        if limit:
            print(f"   • Limit: ${limit:.4f} USD", flush=True)
except Exception as e:
    print(f"  Auth check error: {e}", flush=True)

print("\n Testing Live Qwen 3.7 Flash API Call...", flush=True)
data = {
    "model": "qwen/qwen3.7-flash",
    "messages": [{"role": "user", "content": "مرحبا! أخرج JSON: {\"status\": \"ok\"}"}],
    "response_format": {"type": "json_object"},
}
payload = json.dumps(data).encode("utf-8")
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=payload,
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Abdulaziz-97/RL-ar-",
        "X-Title": "API Key Verification Test",
    }
)

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        content = res["choices"][0]["message"]["content"]
        print("  Live Qwen 3.7 Response Received:", flush=True)
        print(f"   {content}", flush=True)
        print("\n" + "=" * 80, flush=True)
        print(" VERDICT: OPENROUTER API KEY & QWEN 3.7 FLASH ARE 100% OPERATIONAL!", flush=True)
        print("=" * 80, flush=True)
except Exception as e:
    print(f" Qwen 3.7 Call Failed: {e}", flush=True)
    sys.exit(1)
