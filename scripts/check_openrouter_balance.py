import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

openrouter_key = "YOUR_OPENROUTER_API_KEY_HERE"
url = "https://openrouter.ai/api/v1/auth/key"

headers = {
    "Authorization": f"Bearer {openrouter_key}"
}

req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        data = res.get("data", {})
        label = data.get("label", "Default Key")
        usage = data.get("usage", 0.0)
        limit = data.get("limit")
        is_free_tier = data.get("is_free_tier", False)

        print("="*60)
        print(" 💳 OPENROUTER ACCOUNT & KEY BALANCE REPORT")
        print("="*60)
        print(f" Key Label:       {label}")
        print(f" Total Spent:     ${usage:.4f} USD")
        print(f" Key Limit:       ${limit if limit is not None else 'Unlimited'}")
        print(f" Free Tier:       {is_free_tier}")
        print("="*60)
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8")
    print(f"🔴 [HTTP ERROR {e.code}]: {body}")
except Exception as e:
    print(f"🔴 [ERROR]: {e}")
