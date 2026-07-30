import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

api_key = "YOUR_API_KEY_HERE"
url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek-reasoner",
    "messages": [
        {"role": "user", "content": "حل المسألة: 15 × 12؟"}
    ]
}

req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("🟢 [SUCCESS] DeepSeek Reasoner (R1) WORKS!")
        print("Thinking:", res["choices"][0]["message"].get("reasoning_content", "")[:100])
        print("Answer:", res["choices"][0]["message"]["content"])
except Exception as e:
    print(f"🔴 [FAILED] DeepSeek Reasoner: {e}")
