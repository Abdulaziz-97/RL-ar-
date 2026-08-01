"""
EXPERIMENTAL / NON-PRODUCTION.

Demoted from the V4 production path. Free-form Qwen-generated ground truth is not
verifier-first and does not emit canonical family_id / verifier metadata.

Production generation must use `data_1/scripts/run_pipeline.py` with
`configs/full_sft_6500.yaml` / `configs/full_rlvr_8000.yaml` plus
select/calibrate/curate/promote scripts.

Set EXPERIMENTAL_QWEN_DATAGEN=1 to run this script intentionally.
"""

import json
import os
import urllib.request
import time
import sys
import random
import re
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

if os.environ.get("EXPERIMENTAL_QWEN_DATAGEN") != "1":
    print(
        "REFUSED: generate_qwen37_rlvr_live.py is demoted from production.\n"
        "Use run_pipeline.py + calibrate_v4_pass8.py + curate_v4_rlvr.py + promote_v4_release.py.\n"
        "Set EXPERIMENTAL_QWEN_DATAGEN=1 only for experimental staging.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)

SCRIPT_DIR  = Path(__file__).resolve().parent
PACK_ROOT   = SCRIPT_DIR.parent
ROOT_DIR    = PACK_ROOT.parent

OUT_SFT     = ROOT_DIR / "data" / "arabic_reasoning_coldstart_v4.jsonl"
OUT_RLVR    = ROOT_DIR / "data" / "arabic_reasoning_rlvr_v4.jsonl"
BUDGET_FILE = PACK_ROOT / "outputs" / "run" / "budget_qwen37.json"
BUDGET_FILE_MAIN = PACK_ROOT / "outputs" / "run" / "budget.json"

RLVR_TARGET = 4000
WORKERS     = 30

OPENROUTER_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "YOUR_OPENROUTER_API_KEY_HERE",
)
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/Abdulaziz-97/RL-ar-",
    "X-Title": "Arabic RLVR Pipeline V4",
}
MODEL = "qwen/qwen3.7-flash"

COST_PER_PROMPT_TOKEN     = 0.05 / 1_000_000
COST_PER_COMPLETION_TOKEN = 0.15 / 1_000_000

DOMAINS        = ["gsm8k", "math", "math_comp", "logic", "arapro_knowledge", "ifeval_multiconstraint", "aratrust_truth"]
DOMAIN_WEIGHTS = [0.25,    0.20,   0.20,        0.15,    0.08,              0.06,                    0.06]

ANSWER_TYPE = {
    "gsm8k": "integer", "math": "integer", "math_comp": "integer",
    "logic": "logic_json", "arapro_knowledge": "integer",
    "ifeval_multiconstraint": "integer", "aratrust_truth": "integer",
}

MAX_RETRIES   = 3
RETRY_BACKOFF = 1.0

ARABIC_STOPWORDS = {
    "في", "من", "عن", "على", "إلى", "أن", "إن", "كان", "كانت", "إذا", "ما", "هو", "هي", "هذا",
    "هذه", "الذي", "التي", "الذين", "ثم", "أو", "مع", "حيث", "كل", "فما", "كم", "أوجد", "احسب"
}

def super_clean(text: str) -> str:
    t = text.strip().lower()
    t = t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    t = re.sub(r'[^\w\s]', '', t)
    return " ".join(t.split())

def get_content_tokens(text: str) -> set:
    tokens = set(super_clean(text).split())
    return tokens - ARABIC_STOPWORDS

def get_5grams(text: str) -> set:
    words = super_clean(text).split()
    if len(words) < 5:
        return set([" ".join(words)])
    return set(" ".join(words[i:i+5]) for i in range(len(words)-4))

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

# ── Thread-safe Budget Tracker ─────────────────────────────────────────
class BudgetTracker:
    def __init__(self, budget_usd: float = 20.0):
        self._lock = threading.Lock()
        self.spent_usd = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.calls = 0
        self.qwen37_calls = 0
        self.sft_samples_generated = 4000
        self.rlvr_prompts_generated = 0
        self.duplicates_rejected = 0
        self.api_errors = 0
        self.budget_usd = budget_usd
        self.cap_hit = False
        self.start_time = time.time()

    def log_call(self, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self.calls += 1
            self.qwen37_calls += 1
            self.tokens_in += prompt_tokens
            self.tokens_out += completion_tokens
            cost = (prompt_tokens * COST_PER_PROMPT_TOKEN) + (completion_tokens * COST_PER_COMPLETION_TOKEN)
            self.spent_usd += cost
            self.rlvr_prompts_generated += 1
            if self.spent_usd >= self.budget_usd:
                self.cap_hit = True
            self._save_unlocked()

    def log_duplicate(self):
        with self._lock:
            self.duplicates_rejected += 1

    def log_error(self):
        with self._lock:
            self.api_errors += 1

    def _save_unlocked(self):
        elapsed = time.time() - self.start_time
        rate = self.rlvr_prompts_generated / elapsed if elapsed > 0 else 0
        remaining = RLVR_TARGET - self.rlvr_prompts_generated
        eta_seconds = remaining / rate if rate > 0 else 0

        data = {
            "spent_usd": round(self.spent_usd, 6),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "calls": self.calls,
            "qwen37_calls": self.qwen37_calls,
            "sft_samples_generated": self.sft_samples_generated,
            "rlvr_prompts_generated": self.rlvr_prompts_generated,
            "duplicates_rejected": self.duplicates_rejected,
            "api_errors": self.api_errors,
            "samples_per_second": round(rate, 2),
            "eta_minutes": round(eta_seconds / 60, 1),
            "elapsed_minutes": round(elapsed / 60, 1),
            "budget_usd": self.budget_usd,
            "cap_hit": self.cap_hit,
            "workers": WORKERS,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        for path in (BUDGET_FILE, BUDGET_FILE_MAIN):
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=2))

tracker = BudgetTracker(budget_usd=20.0)

# ── Thread-safe Decontamination & Jaccard Registry ───────────────────────
class DecontaminationRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._seen_clean: set[str] = set()
        self._sft_5grams: set[str] = set()
        self._sft_token_sets: list[set] = []
        self._rlvr_token_sets: list[set] = []

    def load_sft(self, sft_prompts: list[str]):
        with self._lock:
            for p in sft_prompts:
                clean_p = super_clean(p)
                self._seen_clean.add(clean_p)
                self._sft_5grams.update(get_5grams(clean_p))
                self._sft_token_sets.append(get_content_tokens(clean_p))

    def is_valid(self, prompt: str) -> bool:
        clean_p = super_clean(prompt)
        cand_tokens = get_content_tokens(clean_p)
        cand_5grams = get_5grams(clean_p)

        with self._lock:
            # 1. Exact string match check
            if clean_p in self._seen_clean:
                return False

            # 2. 5-Gram overlap check against SFT (< 60%)
            if cand_5grams and self._sft_5grams:
                overlap_ratio = len(cand_5grams.intersection(self._sft_5grams)) / len(cand_5grams)
                if overlap_ratio > 0.60:
                    return False

            # 3. Content-Word Jaccard Similarity Guard (< 0.65 limit)
            check_sets = random.sample(self._sft_token_sets, min(60, len(self._sft_token_sets))) + \
                         random.sample(self._rlvr_token_sets, min(60, len(self._rlvr_token_sets)))
            for ts in check_sets:
                if jaccard_similarity(cand_tokens, ts) > 0.65:
                    return False

            self._seen_clean.add(clean_p)
            self._rlvr_token_sets.append(cand_tokens)
            return True

registry = DecontaminationRegistry()

# ── LLM Call ───────────────────────────────────────────────────────────
def call_qwen37(prompt_text: str) -> dict | None:
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }
    payload = json.dumps(data).encode("utf-8")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                usage = body.get("usage", {})
                p_tok = usage.get("prompt_tokens", 350)
                c_tok = usage.get("completion_tokens", 100)
                tracker.log_call(p_tok, c_tok)
                content = body["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception:
            tracker.log_error()
            time.sleep(RETRY_BACKOFF)
    return None

def _generate_one_rlvr(domain: str, idx: int) -> dict | None:
    instruction = f"""أنت معلم رياضيات ومنطق خبير باللغة العربية الفصحى.
أنشئ مسألة فريدة وجديدة تمامًا في مجال ({domain}).

المطلوب:
1. نص المسألة بالعربية الفصحى (prompt) — لا تقل عن جملتين.
2. الإجابة النهائية الرقمية المباشرة فقط بدون وحدات (gold).

ملاحظة مهمة: يجب أن تكون المسألة جديدة تماماً ومبتكرة بأرقام وسياق مختلف.

أخرج النتيجة بصيغة JSON فقط:
{{
  "prompt": "نص المسألة بالعربية الفصحى...",
  "gold": "الرقم فقط"
}}"""
    result = call_qwen37(instruction)
    if not result or not isinstance(result, dict) or not result.get("prompt") or not result.get("gold"):
        return None

    p_raw = result["prompt"]
    if not registry.is_valid(p_raw):
        tracker.log_duplicate()
        return None

    gold = str(result["gold"]).strip()
    atype = ANSWER_TYPE.get(domain, "integer")
    spec = {"type": atype, "canonical": gold}
    try:
        spec["ground_truth_structured"] = int(gold)
    except ValueError:
        spec["ground_truth_structured"] = gold

    return {
        "id": f"rlvr_qwen37_v4_{idx:05d}",
        "domain": domain,
        "partition": "rlvr",
        "prompt": p_raw,
        "response": "",
        "answer_spec": spec,
        "metadata": {
            "ground_truth_answer": gold,
            "teacher_model": MODEL,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

def main():
    print("=" * 80, flush=True)
    print(f" 🚀 GENERATING 100% LIVE QWEN 3.7 FLASH RLVR DATASET WITH CONTENT JACCARD GUARD (< 0.65)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load SFT prompts into Decontamination Registry
    sft_prompts = []
    with open(OUT_SFT, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                sft_prompts.append(json.loads(l).get("prompt", ""))

    registry.load_sft(sft_prompts)
    print(f"[Decontam] Loaded {len(sft_prompts)} SFT prompts into Content-Jaccard & 5-gram decontamination registry.", flush=True)

    # 2. Check if OUT_RLVR already has 4000 valid samples
    existing_samples = []
    if OUT_RLVR.exists():
        with open(OUT_RLVR, encoding="utf-8") as f:
            existing_samples = [l for l in f if l.strip()]
    
    if len(existing_samples) >= RLVR_TARGET:
        print(f"=================================================================", flush=True)
        print(f" 🎯 FOUND {len(existing_samples)} EXISTING RLVR PROMPTS IN {OUT_RLVR}!", flush=True)
        print(f" ⚡ SKIPPING GENERATION AND PROCEEDING IMMEDIATELY TO TRAINING!", flush=True)
        print(f"=================================================================", flush=True)
        return

    # If partial generation, keep existing samples and append
    counter = [len(existing_samples)]
    if len(existing_samples) > 0:
        print(f"[Phase 2 RLVR] Resuming generation from sample #{counter[0] + 1}/{RLVR_TARGET}...", flush=True)
    else:
        with open(OUT_RLVR, "w", encoding="utf-8") as f:
            pass
    write_lock = threading.Lock()

    print(f"[Phase 2 RLVR] Generating {RLVR_TARGET} LIVE Qwen 3.7 Flash RLVR prompts with {WORKERS} workers...", flush=True)

    with open(OUT_RLVR, "a", encoding="utf-8") as f:
        while counter[0] < RLVR_TARGET:
            batch_size = min(WORKERS, RLVR_TARGET - counter[0])
            domains_batch = [random.choices(DOMAINS, weights=DOMAIN_WEIGHTS, k=1)[0] for _ in range(batch_size)]

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = []
                for dom in domains_batch:
                    with write_lock:
                        idx = counter[0] + len(futures) + 1
                    futures.append(pool.submit(_generate_one_rlvr, dom, idx))

                for future in as_completed(futures):
                    item = future.result()
                    if item is not None:
                        with write_lock:
                            counter[0] += 1
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                            f.flush()
                            c = counter[0]
                        print(f"[Phase 2 RLVR] ✓ {c}/{RLVR_TARGET} | ${tracker.spent_usd:.4f} | Workers: {WORKERS}", flush=True)

    tracker._save_unlocked()
    print("\n" + "=" * 80, flush=True)
    print(f" 🏁 PHASE 2 RLVR GENERATION COMPLETE: 4,000 LIVE PROMPTS IN {OUT_RLVR}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
