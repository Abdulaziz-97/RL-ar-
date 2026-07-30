import urllib.request
import urllib.error
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# The two keys provided by the user:
provided_keys = [
    "v1-8866a79fa192bad2f570aa467f671c2e7f0b11efbfbc30bd1e6a16c2c9747d87",
    "YOUR_API_KEY_HERE7f0b11efbfbc30bd1e6a16c2c9747d87"
]

# OpenRouter key formats to test
key_variants = []
for k in provided_keys:
    key_variants.append(k)
    if not k.startswith("sk-or-"):
        clean = k.replace("sk-", "").replace("v1-", "")
        key_variants.append(f"sk-or-v1-{clean}")

url = "https://openrouter.ai/api/v1/chat/completions"

print("="*70)
print(" 🔍 RAW OPENROUTER API KEY VERIFICATION TEST")
print("="*70)

for idx, key in enumerate(key_variants, 1):
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Abdulaziz-97/RL-ar-",
        "X-Title": "Arabic RLVR Pipeline Test"
    }
    data = {
        "model": "qwen/qwen3.7-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    
    print(f"\nTest #{idx}: Key prefix '{key[:22]}...'")
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            print(f"  [SUCCESS] Status Code: {resp.status}")
            print(f"  [RESPONSE]: {body}")
            sys.exit(0)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"  [FAILED] HTTP {e.code}: {err_body}")
    except Exception as e:
        print(f"  [ERROR]: {e}")

print("="*70)
