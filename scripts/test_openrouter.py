import urllib.request
import json
import os

api_key = "YOUR_OPENROUTER_API_KEY_HERE"

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "qwen/qwen3.7-flash",
    "messages": [
        {"role": "user", "content": "حل المسألة التالية بالتفصيل باللغة العربية الفصحى داخل وسم <think> وضعي الناتج في <answer>: ما هو حاصل ضرب 14 × 15؟"}
    ]
}

req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("[SUCCESS] Response from qwen/qwen3.7-flash:")
        print(res["choices"][0]["message"]["content"])
except Exception as e:
    print(f"[ERROR] {e}")
