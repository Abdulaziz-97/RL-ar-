import urllib.request
import json
import os

api_key = "YOUR_API_KEY_HERE"
url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": "حل المسألة التالية بالتفصيل باللغة العربية الفصحى داخل وسم <think> وضعي الناتج في <answer>: اشترى رجل 12 كتابا بسعر 25 ريالا للكتاب. كم يدفع إجمالا؟"}
    ]
}

req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("[SUCCESS] DeepSeek Response:")
        print(res["choices"][0]["message"]["content"])
except Exception as e:
    print(f"[ERROR] DeepSeek API: {e}")
