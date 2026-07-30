import json
import argparse
from pathlib import Path
from collections import Counter

def evaluate_dataset_quality(jsonl_path: Path) -> dict:
    if not jsonl_path.exists():
        return {"error": f"File not found: {jsonl_path}"}
        
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if not records:
        return {"error": "Empty dataset"}
        
    total = len(records)
    arabic_purity_sum = 0.0
    valid_format_count = 0
    clean_structure_count = 0
    total_steps = 0
    step_counts = []
    
    for r in records:
        text = r.get("response") or r.get("solution") or ""
        
        # 1. Arabic Purity
        if text:
            ar_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
            non_space = max(len(text.replace(" ", "")), 1)
            purity = ar_chars / non_space
        else:
            purity = 0.0
        arabic_purity_sum += purity
        
        # 2. Tag Format Compliance (<think> and <answer>)
        has_think = "<think>" in text and "</think>" in text
        has_answer = "<answer>" in text and "</answer>" in text
        if has_think and has_answer:
            valid_format_count += 1
            
        # 3. Structural Leak Check (text outside tags)
        stripped = text.strip()
        starts_think = stripped.startswith("<think>")
        ends_answer = stripped.endswith("</answer>")
        if starts_think and ends_answer:
            clean_structure_count += 1
            
        # 4. Reasoning Steps
        if "<think>" in text and "</think>" in text:
            think_content = text.split("<think>")[1].split("</think>")[0]
            lines = [l.strip() for l in think_content.split("\n") if l.strip()]
            num_steps = len(lines)
        else:
            num_steps = 0
        step_counts.append(num_steps)
        total_steps += num_steps

    avg_purity = (arabic_purity_sum / total) * 100
    format_rate = (valid_format_count / total) * 100
    clean_structure_rate = (clean_structure_count / total) * 100
    avg_steps = total_steps / total
    
    # Quality Score Formula (Weighted combination of purity, format, structure, and depth)
    depth_score = min((avg_steps / 4.0) * 100, 100)
    overall_quality_score = (0.35 * format_rate) + (0.30 * clean_structure_rate) + (0.25 * avg_purity) + (0.10 * depth_score)

    return {
        "dataset_name": jsonl_path.name,
        "total_samples": total,
        "arabic_purity_pct": round(avg_purity, 2),
        "format_compliance_pct": round(format_rate, 2),
        "clean_structure_pct": round(clean_structure_rate, 2),
        "avg_reasoning_steps": round(avg_steps, 2),
        "overall_quality_score": round(overall_quality_score, 2)
    }

def print_benchmark_comparison(results: list[dict]):
    print("\n" + "="*80)
    print(" 📊 DATA QUALITY GENERATION BENCHMARK COMPARISON ")
    print("="*80)
    
    header = f"{'Metric':<30} | " + " | ".join([f"{r.get('dataset_name', 'Model'):<20}" for r in results])
    print(header)
    print("-" * len(header))
    
    metrics = [
        ("Total Samples", "total_samples"),
        ("Arabic Purity (%)", "arabic_purity_pct"),
        ("Format Compliance (%)", "format_compliance_pct"),
        ("Clean Tag Structure (%)", "clean_structure_pct"),
        ("Avg Reasoning Steps", "avg_reasoning_steps"),
        ("🏆 Overall Quality Score", "overall_quality_score")
    ]
    
    for label, key in metrics:
        row = f"{label:<30} | " + " | ".join([f"{str(r.get(key, 'N/A')):<20}" for r in results])
        print(row)
        
    print("="*80)
    
    # Determine winner
    valid_results = [r for r in results if "overall_quality_score" in r]
    if len(valid_results) >= 2:
        winner = max(valid_results, key=lambda x: x["overall_quality_score"])
        print(f"\n🏆 WINNER: {winner['dataset_name']} with an Overall Quality Score of {winner['overall_quality_score']}/100!\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark synthetic data quality between two datasets/models.")
    parser.add_argument("--file1", type=Path, required=True, help="Path to first dataset JSONL")
    parser.add_argument("--file2", type=Path, required=True, help="Path to second dataset JSONL")
    args = parser.parse_args()

    r1 = evaluate_dataset_quality(args.file1)
    r2 = evaluate_dataset_quality(args.file2)

    print_benchmark_comparison([r1, r2])

if __name__ == "__main__":
    main()
