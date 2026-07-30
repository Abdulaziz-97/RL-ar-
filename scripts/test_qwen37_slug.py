import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

openrouter_key = "YOUR_OPENROUTER_API_KEY_HERE"
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {openrouter_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/Abdulaziz-97/RL-ar-",
    "X-Title": "Arabic RLVR Pipeline"
}

data = {
    "model": "qwen/qwen3.7-flash",
    "messages": [
        {"role": "user", "content": "مرحبا، أجب بكلمة واحدة: تم"}
    ]
}

req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("🟢 [SUCCESS] qwen/qwen3.7-flash WORKS!")
        print("Response:", res["choices"][0]["message"]["content"])
except Exception as e:
    print(f"🔴 [FAILED] qwen/qwen3.7-flash: {e}")
