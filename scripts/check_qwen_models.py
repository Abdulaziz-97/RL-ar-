import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

openrouter_key = "YOUR_OPENROUTER_API_KEY_HERE"
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {openrouter_key}",
    "Content-Type": "application/json"
}

qwen_slugs = [
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen-max",
    "qwen/qwen-plus",
    "qwen/qwen2.5-vl-72b-instruct:free"
]

for slug in qwen_slugs:
    data = {
        "model": slug,
        "messages": [{"role": "user", "content": "مرحبا"}],
        "max_tokens": 10
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            print(f"🟢 [SUCCESS] Model slug '{slug}' WORKS!")
            print("Response:", res["choices"][0]["message"]["content"])
            break
    except Exception as e:
        print(f"🔴 [FAILED] '{slug}': {e}")
