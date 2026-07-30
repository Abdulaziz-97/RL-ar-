import urllib.request
import urllib.error
import json
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

key = "YOUR_OPENROUTER_API_KEY_HERE"

models = [
    "qwen/qwen3.7-flash",
    "qwen/qwen-2.5-72b-instruct",
    "deepseek/deepseek-r1",
    "openai/gpt-4o-mini"
]

url = "https://openrouter.ai/api/v1/chat/completions"

print("="*70)
print(f" 🔄 RETRYING OPENROUTER API (5 Retries per Model)")
print("="*70)

for model in models:
    print(f"\n[MODEL]: {model}")
    for attempt in range(1, 6):
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Abdulaziz-97/RL-ar-",
            "X-Title": "Arabic RLVR Pipeline"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": "مرحبا، أجب بكلمة واحدة: تم"}],
            "max_tokens": 10
        }
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                print(f"  🟢 [Attempt {attempt} SUCCESS!] Response:")
                print("  ", res["choices"][0]["message"]["content"])
                sys.exit(0)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")
            print(f"  🔴 [Attempt {attempt} HTTP {e.code}]: {err}")
        except Exception as e:
            print(f"  🔴 [Attempt {attempt} Error]: {e}")
            
        time.sleep(1)

print("\n" + "="*70)
print(" ❌ All 20 retries completed. Key returns 401 User not found.")
print("="*70)
