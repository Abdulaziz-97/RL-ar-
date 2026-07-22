"""Quick format probe after SFT — writes UTF-8 report (Windows-safe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rlvr.reward_composer import reward_format
from rlvr_pipeline.config import DEFAULT_SYSTEM_PROMPT
from rlvr_pipeline.trainer import fix_chat_template, load_sft_adapter_strict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/sft_v4")
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 512
OUT = CKPT / "format_probe_results.json"

PROBES = [
    "ما هو 15 + 27؟",
    "ما هو الجذر التربيعي لـ 144؟",
    "إذا كان لديك 5 تفاحات وأعطيت 2، كم تبقى؟",
]


def main() -> int:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    lora_cfg = LoraConfig.from_pretrained(str(CKPT))
    try:
        tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    except OSError:
        tok = AutoTokenizer.from_pretrained(
            lora_cfg.base_model_name_or_path, trust_remote_code=True
        )
    if fix_chat_template(tok):
        print("chat template fixed: removed empty <think> injection")
    lora_cfg.inference_mode = True
    base = AutoModelForCausalLM.from_pretrained(
        lora_cfg.base_model_name_or_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(base, lora_cfg)
    load_sft_adapter_strict(model, str(CKPT))
    model.eval()

    print(f"ckpt={CKPT} max_new_tokens={MAX_NEW}")
    rows = []
    passed = 0
    for q in PROBES:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        has_think = "<think>" in text and "</think>" in text
        has_ans = "<answer>" in text and "</answer>" in text
        fmt = float(reward_format(text)) if has_think and has_ans else 0.0
        ok = bool(has_think and has_ans and fmt > 0)
        passed += int(ok)
        rows.append(
            {
                "prompt": q,
                "think": has_think,
                "answer": has_ans,
                "format": fmt,
                "pass": ok,
                "n_chars": len(text),
                "completion": text,
            }
        )
        print(f"PASS={ok} format={fmt:.2f} think={has_think} answer={has_ans} chars={len(text)}")

    report = {
        "passed": passed,
        "total": len(PROBES),
        "max_new_tokens": MAX_NEW,
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FORMAT PROBE: {passed}/{len(PROBES)} -> {OUT}")
    return 0 if passed >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
