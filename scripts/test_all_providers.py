import urllib.request
import urllib.error
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

key = "YOUR_API_KEY_HERE7f0b11efbfbc30bd1e6a16c2c9747d87"

endpoints = [
    ("DashScope / Qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen-plus"),
    ("DashScope / Qwen 2.5", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen2.5-72b-instruct"),
    ("DeepSeek", "https://api.deepseek.com/v1/chat/completions", "deepseek-chat"),
    ("OpenAI", "https://api.openai.com/v1/chat/completions", "gpt-4o-mini"),
    ("OpenRouter", "https://openrouter.ai/api/v1/chat/completions", "qwen/qwen-2.5-72b-instruct")
]

for name, url, model in endpoints:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "مرحبا"}]
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            print(f"[SUCCESS] {name} (Model: {model}) WORKS!")
            print("Response:", res["choices"][0]["message"]["content"])
            sys.exit(0)
    except urllib.error.HTTPError as e:
        print(f"[FAILED] {name} ({model}): {e.code} - {e.read().decode('utf-8')[:100]}")
    except Exception as e:
        print(f"[FAILED] {name} ({model}): {e}")
