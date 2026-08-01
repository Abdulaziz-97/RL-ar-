"""Quick sanity test script to verify merged model generation capabilities."""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Quick test of merged model generation")
    parser.add_argument("--model-path", type=str, default="/workspace/RL-ar-/outputs/qwen_4b_2x5090_v4_run/merged_eval_checkpoint-200")
    parser.add_argument("--prompt", type=str, default="السؤال: كم يبلغ حاصل ضرب 12 في 15؟\nA. 150\nB. 180\nC. 200\nD. 160\nالإجابة:")
    args = parser.parse_args()

    model_dir = Path(args.model_path)
    if not model_dir.exists():
        print(f"Error: Model directory {args.model_path} not found!")
        return

    print(f"\n[INFO] Loading vLLM engine for test model: {args.model_path}...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("\n--- [PROMPT PASSED TO MODEL] ---")
    print(formatted_prompt)
    print("--------------------------------\n")

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.0,
        repetition_penalty=1.05,
    )

    print("\nGenerating response from model...", flush=True)
    outputs = llm.generate([formatted_prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text

    print("\n==========================================")
    print("      [MODEL OUTPUT GENERATION TEST]      ")
    print("==========================================")
    print(generated_text)
    print("==========================================\n")

if __name__ == "__main__":
    main()
