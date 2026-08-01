"""Quick sanity test script using PyTorch + Hugging Face Transformers.
Loads base model + adapter, merges cleanly in memory with zero stripped weights, and generates 1 test output.
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Sanity test GRPO model generation with Hugging Face")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-path", type=str, default="/workspace/RL-ar-/outputs/qwen_4b_2x5090_v4_run/checkpoint-200")
    parser.add_argument("--prompt", type=str, default="السؤال: كم يبلغ حاصل ضرب 12 في 15؟\nA. 150\nB. 180\nC. 200\nD. 160\nالإجابة:")
    args = parser.parse_args()

    print(f"\n[1/3] Loading base model ({args.base_model}) in bfloat16...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"[2/3] Loading GRPO LoRA adapter ({args.adapter_path})...", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter_path)

    # Zero out embed_tokens and lm_head LoRA parameters to prevent embedding distortion during merge
    for name, param in model.named_parameters():
        if "lora_" in name and ("embed_tokens" in name or "lm_head" in name):
            param.data.zero_()

    print(f"[2.5/3] Merging attention & MLP LoRA weights into base model...", flush=True)
    merged_model = model.merge_and_unload()
    merged_model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("\n--- [FORMATTED PROMPT] ---")
    print(formatted_prompt)
    print("--------------------------\n")

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")

    print("[3/3] Generating response with greedy decoding (repetition_penalty=1.10)...", flush=True)
    with torch.no_grad():
        output_ids = merged_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # greedy
            repetition_penalty=1.10,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Cut off the prompt tokens to get only the generated completion
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n==========================================")
    print("      [MODEL OUTPUT GENERATION TEST]      ")
    print("==========================================")
    print(generated_text)
    print("==========================================\n")

if __name__ == "__main__":
    main()
