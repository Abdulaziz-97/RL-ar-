import json
import urllib.request
import time
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PACK_ROOT = Path(__file__).resolve().parents[1]
prompts_file = PACK_ROOT / "outputs" / "benchmark_50_prompts.jsonl"
out_file = PACK_ROOT / "outputs" / "model_b_deepseek_r1_50.jsonl"

api_key = "YOUR_API_KEY_HERE"
url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def main():
    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    done_ids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    done_ids.add(item.get("id"))

    print(f"[INFO] Generating Model B (DeepSeek-R1 Reasoner) responses ({len(done_ids)}/{len(prompts)} already completed)...")

    with open(out_file, "a", encoding="utf-8") as out_f:
        for idx, p_item in enumerate(prompts, 1):
            if p_item["id"] in done_ids:
                continue
                
            prompt_text = p_item["prompt"]
            user_content = f"أنت معلم رياضيات ومنطق خبير باللغة العربية الفصحى. حل المسألة التالية بالتفصيل باللغة العربية الفصحى:\n\nالمسألة:\n{prompt_text}"
            
            data = {
                "model": "deepseek-reasoner",
                "messages": [{"role": "user", "content": user_content}]
            }

            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    msg = res["choices"][0]["message"]
                    think_text = msg.get("reasoning_content", "") or ""
                    ans_text = msg.get("content", "") or ""
                    
                    full_response = f"<think>\n{think_text.strip()}\n</think>\n<answer>{ans_text.strip()}</answer>"
                    
                    item = dict(p_item)
                    item["model"] = "DeepSeek-R1-Reasoner"
                    item["response"] = full_response
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    out_f.flush()
                    print(f"[DeepSeek-R1 {idx}/50] Generated successfully.")
            except Exception as e:
                print(f"[DeepSeek-R1 {idx}/50] Failed: {e}")
                
            time.sleep(0.3)

    print(f"[SUCCESS] Model B (DeepSeek-R1) dataset complete at: {out_file}")

if __name__ == "__main__":
    main()
