import json
import os
import shutil
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(PACK_ROOT / "src"))
sys.path.insert(0, str(PACK_ROOT / "vendor"))

from rlvr_synth.orchestrator import SynthConfig, SynthOrchestrator

def run_e2e_pipeline_for_model(model_slug: str, work_dir_name: str, n_families: int = 20) -> dict:
    work_dir = PACK_ROOT / "outputs" / work_dir_name
    if work_dir.exists():
        shutil.rmtree(work_dir)
        
    cfg = SynthConfig(
        mode="pilot",
        n_families=n_families,
        domains=["gsm8k", "math", "math_comp", "logic"],
        traces_per_problem=1,
        seed=890,
        work_dir=str(work_dir),
        backend="external",
        external_module=str(PACK_ROOT / "backends" / "dspy_backend.py"),
        external_config={
            "mode": "live",
            "model": model_slug,
            "gepa_path": str(PACK_ROOT / "assets" / "arabic_teacher_gepa_v2.json"),
            "budget_path": str(work_dir / "budget.json"),
            "budget_usd": 50.0,
            "cache": False,
        },
        partitions={"sft_train": 0.5, "rlvr_train": 0.5},
        max_alternate_methods=0
    )
    
    os.environ["RLVR_PACK_MODE"] = "live"
    print(f"\n[E2E BENCHMARK] Executing full 10-Stage SynthOrchestrator DAG for model: {model_slug}...")
    
    orchestrator = SynthOrchestrator(cfg)
    result = orchestrator.run(resume=False)
    
    stages_dir = work_dir / "stages"
    selected_rows = []
    if (stages_dir / "selected.jsonl").exists():
        selected_rows = [json.loads(l) for l in open(stages_dir / "selected.jsonl", encoding="utf-8") if l.strip()]
        
    quarantine_rows = []
    if (work_dir / "quarantine.jsonl").exists():
        quarantine_rows = [json.loads(l) for l in open(work_dir / "quarantine.jsonl", encoding="utf-8") if l.strip()]
        
    gates_rows = []
    if (stages_dir / "gates.jsonl").exists():
        gates_rows = [json.loads(l) for l in open(stages_dir / "gates.jsonl", encoding="utf-8") if l.strip()]

    total_candidates = len(gates_rows)
    passed_gates = sum(1 for r in gates_rows if r.get("gate_pass"))
    verified_steps = sum(1 for r in gates_rows if r.get("step_verified"))
    
    avg_purity = 0.0
    if gates_rows:
        purities = [r.get("arabic_qa", {}).get("arabic_purity", 0.0) for r in gates_rows]
        avg_purity = (sum(purities) / len(purities)) * 100

    return {
        "model_slug": model_slug,
        "work_dir": str(work_dir),
        "total_families": n_families,
        "total_candidates": total_candidates,
        "step_verified_count": verified_steps,
        "gate_pass_count": passed_gates,
        "quarantined_count": len(quarantine_rows),
        "selected_released_count": len(selected_rows),
        "e2e_yield_pct": round((len(selected_rows) / max(total_candidates, 1)) * 100, 2),
        "avg_arabic_purity_pct": round(avg_purity, 2)
    }

def print_e2e_benchmark_report(report_a: dict, report_b: dict):
    print("\n" + "="*85)
    print(" 🏆 E2E SYNTH ORCHESTRATOR PIPELINE BENCHMARK REPORT ")
    print("="*85)
    
    fmt_str = "{:<35} | {:<22} | {:<22}"
    print(fmt_str.format("Pipeline Benchmark Metric", f"Model A: {report_a['model_slug']}", f"Model B: {report_b['model_slug']}"))
    print("-" * 85)
    
    metrics = [
        ("Total Input Families", "total_families"),
        ("Generated Candidates", "total_candidates"),
        ("Step Verified (Python Contract)", "step_verified_count"),
        ("Passed All 5 Release Gates", "gate_pass_count"),
        ("Quarantined Candidates", "quarantined_count"),
        ("Final Released Corpora Count", "selected_released_count"),
        ("🚀 E2E Pipeline Yield Rate (%)", "e2e_yield_pct"),
        ("🇸🇦 Avg Arabic Purity (%)", "avg_arabic_purity_pct"),
    ]
    
    for label, key in metrics:
        print(fmt_str.format(label, str(report_a.get(key, "N/A")), str(report_b.get(key, "N/A"))))
        
    print("="*85)
    
    if report_a["e2e_yield_pct"] > report_b["e2e_yield_pct"]:
        winner = report_a["model_slug"]
        yield_val = report_a["e2e_yield_pct"]
    else:
        winner = report_b["model_slug"]
        yield_val = report_b["e2e_yield_pct"]
        
    print(f"\n🏆 E2E WINNER: {winner} with a Pipeline Yield Rate of {yield_val}%!\n")

def main():
    print("[INFO] Starting E2E SynthOrchestrator Benchmark Check...")
    
    # Run Model A: DeepSeek Pro
    res_a = run_e2e_pipeline_for_model("deepseek-v4-pro", "benchmark_run_model_a", n_families=10)
    
    # Run Model B: Qwen 3.7 Flash
    res_b = run_e2e_pipeline_for_model("qwen/qwen3.7-flash", "benchmark_run_model_b", n_families=10)
    
    print_e2e_benchmark_report(res_a, res_b)

if __name__ == "__main__":
    main()
