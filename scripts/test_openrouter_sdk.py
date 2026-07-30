import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

key = "YOUR_OPENROUTER_API_KEY_HERE"

print("[INFO] Initializing OpenAI Client for OpenRouter...")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=key,
    default_headers={
        "HTTP-Referer": "https://github.com/Abdulaziz-97/RL-ar-",
        "X-Title": "Arabic RLVR Pipeline"
    }
)

try:
    response = client.chat.completions.create(
        model="qwen/qwen3.7-flash",
        messages=[{"role": "user", "content": "مرحبا، أجب بكلمة واحدة: تم"}],
        max_tokens=10
    )
    print("🟢 [SUCCESS] OpenRouter OpenAI SDK Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"🔴 [FAILED] OpenRouter OpenAI SDK: {e}")
