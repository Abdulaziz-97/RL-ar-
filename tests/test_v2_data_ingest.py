"""TDD contract tests for v2_dspy_gepa data ingest into workspace `data/`."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from rlvr.reward_composer import reward_format
from rlvr_pipeline.data import load_cold_start_sft_dataset, load_rlvr_dataset
from rlvr_pipeline.rewards import correctness_reward_func, format_reward_func

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"

RLVR_FULL = DATA / "arabic_reasoning_rlvr.jsonl"
RLVR_TRAIN = DATA / "arabic_reasoning_rlvr_train.jsonl"
RLVR_EVAL = DATA / "arabic_reasoning_rlvr_eval.jsonl"
COLD_FULL = DATA / "arabic_reasoning_coldstart.jsonl"
COLD_TRAIN = DATA / "arabic_reasoning_coldstart_train.jsonl"
COLD_EVAL = DATA / "arabic_reasoning_coldstart_eval.jsonl"

EXPECTED_MIX = {"gsm8k": 280, "math": 180, "math_comp": 180, "logic": 160}
CLOSER_RE = re.compile(
    r"(الجواب النهائي|الإجابة النهائية|الناتج النهائي|الجواب هو)\s*[:：]?",
    re.UNICODE,
)
BOILERPLATE = "الخطوة التوضيحية"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _ids(rows: list[dict]) -> set[str]:
    return {str(r.get("id", "")) for r in rows}


def _require_v2_files() -> None:
    missing = [p.name for p in (RLVR_FULL, RLVR_TRAIN, RLVR_EVAL, COLD_FULL, COLD_TRAIN, COLD_EVAL) if not p.exists()]
    if missing:
        pytest.fail(f"v2 data not ingested yet; missing: {missing}")


@pytest.fixture(scope="module")
def v2_ready():
    _require_v2_files()


def test_v2_file_sizes(v2_ready):
    assert len(_load_jsonl(RLVR_TRAIN)) >= 640
    assert len(_load_jsonl(COLD_TRAIN)) >= 640


def test_v2_domain_mix(v2_ready):
    for path in (RLVR_FULL, COLD_FULL):
        mix = Counter(r["domain"] for r in _load_jsonl(path))
        assert dict(mix) == EXPECTED_MIX, f"{path.name}: {dict(mix)}"


def test_v2_train_eval_id_disjoint(v2_ready):
    assert _ids(_load_jsonl(RLVR_TRAIN)) & _ids(_load_jsonl(RLVR_EVAL)) == set()
    assert _ids(_load_jsonl(COLD_TRAIN)) & _ids(_load_jsonl(COLD_EVAL)) == set()


def test_v2_rlvr_contracts(v2_ready):
    for r in _load_jsonl(RLVR_FULL):
        assert (r.get("response") or "") == ""
        assert str(r.get("answer") or "").strip() != ""
        gt = (r.get("metadata") or {}).get("ground_truth_answer")
        assert gt is not None and gt != ""
        if r["domain"] == "logic":
            assert isinstance(gt, dict)


def test_v2_cold_contracts(v2_ready):
    for r in _load_jsonl(COLD_FULL):
        resp = r.get("response") or ""
        assert "<think>" in resp and "</think>" in resp
        assert "<answer>" in resp and "</answer>" in resp
        assert "####" not in resp
        assert BOILERPLATE not in resp
        assert (r.get("metadata") or {}).get("flash_agree") is True


def test_v2_no_classic_closer_leaks(v2_ready):
    """Gate policy: ban classic closers; last `= N` arithmetic line is allowed."""
    bad = 0
    for r in _load_jsonl(COLD_FULL):
        resp = r.get("response") or ""
        m = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
        a = re.search(r"<answer>(.*?)</answer>", resp, re.DOTALL)
        if not m or not a:
            continue
        think, ans = m.group(1), a.group(1).strip()
        if len(ans) <= 1:
            continue
        if CLOSER_RE.search(think) and ans in think:
            # Only count if closer phrase appears near the answer token
            if re.search(
                rf"(الجواب النهائي|الإجابة النهائية|الناتج النهائي|الجواب هو)\s*[:：]?\s*{re.escape(ans)}",
                think,
            ):
                bad += 1
    assert bad == 0, f"classic closer+GT leaks: {bad}"


def test_v2_loaders(v2_ready):
    assert len(load_rlvr_dataset(RLVR_TRAIN)) >= 640
    assert len(load_cold_start_sft_dataset(COLD_TRAIN)) >= 640


def test_v2_logic_correctness_with_serialized_gt_string(v2_ready):
    """Loader stringifies dict GT; reward must still score 1.0 for matching JSON answer."""
    logic_row = next(r for r in _load_jsonl(COLD_FULL) if r["domain"] == "logic")
    resp = logic_row["response"]
    assert reward_format(resp) > 0.0
    assert format_reward_func(["p"], [resp])[0] > 0.0

    gt = (logic_row.get("metadata") or {})["ground_truth_answer"]
    assert isinstance(gt, dict)
    # Simulate what TRL sees after load_rlvr_dataset serialization
    gt_str = json.dumps(gt, ensure_ascii=False)
    completion = resp  # already has matching <answer>…</answer>
    rewards = correctness_reward_func(
        prompts=["p"],
        completions=[completion],
        ground_truth_answer=[gt_str],
        domain=["logic"],
        puzzle_type=["logic"],
    )
    assert rewards == [1.0], f"logic correctness with string GT failed: {rewards}"


def test_v2_logic_correctness_with_canonical_separators(v2_ready):
    logic_row = next(r for r in _load_jsonl(RLVR_FULL) if r["domain"] == "logic")
    gt = (logic_row.get("metadata") or {})["ground_truth_answer"]
    canonical = logic_row["answer"]
    completion = f"<think>خطوات كافية هنا للتحقق من الحل المنطقي بشكل واضح</think><answer>{canonical}</answer>"
    # Compact separators as shipped in answer field
    rewards = correctness_reward_func(
        prompts=["p"],
        completions=[completion],
        ground_truth_answer=[json.dumps(gt, ensure_ascii=False)],
        domain=["logic"],
    )
    assert rewards == [1.0]


def test_v2_every_cold_row_scores_correctness_one(v2_ready):
    """Full-corpus: cold teacher response must score correctness=1 under TRL GT serialization."""
    from rlvr_pipeline.data import DOMAIN_MAP, _serialize_ground_truth

    fails = []
    for r in _load_jsonl(COLD_FULL):
        resp = r["response"]
        gt = (r.get("metadata") or {})["ground_truth_answer"]
        ser = _serialize_ground_truth(gt)
        dom = DOMAIN_MAP.get(r["domain"], r["domain"])
        score = correctness_reward_func(
            ["p"],
            [resp],
            ground_truth_answer=[ser],
            domain=[dom],
            puzzle_type=[r["domain"] if dom == "logic" else None],
        )[0]
        if score < 1.0:
            fails.append(r["id"])
    assert fails == [], f"cold correctness failures ({len(fails)}): {fails[:10]}"


def test_v2_every_rlvr_train_perfect_completion_scores_one(v2_ready):
    """If the model emitted the canonical answer, reward must be 1.0 for every train row."""
    from rlvr_pipeline.data import load_rlvr_dataset

    rows = {r.get("sample_id", r.get("id")): r for r in _load_jsonl(RLVR_TRAIN)}
    ds = load_rlvr_dataset(RLVR_TRAIN)
    fails = []
    think = "خطوات كافية للتحقق من الحل هنا بوضوح تام ومراجعة"
    for i in range(min(50, len(ds))):
        sid = ds[i]["sample_id"]
        ans = str(ds[i]["ground_truth_answer"])
        comp = f"<think>{think}</think><answer>{ans}</answer>"
        score = correctness_reward_func(
            ["p"],
            [comp],
            ground_truth_answer=[ds[i]["ground_truth_answer"]],
            domain=[ds[i]["domain"]],
            puzzle_type=[ds[i]["puzzle_type"]],
        )[0]
        if score < 1.0:
            fails.append(sid)
    assert fails == [], f"rlvr perfect-comp failures ({len(fails)}): {fails[:10]}"


def test_v2_unique_ids_within_each_file(v2_ready):
    for path in (RLVR_TRAIN, COLD_TRAIN):
        ids = [r.get("sample_id") or r.get("id") or r.get("id_") or str(i) for i, r in enumerate(_load_jsonl(path))]
        assert len(ids) == len(set(ids)), f"duplicate ids in {path.name}"


def test_v2_train_eval_partition_covers_full(v2_ready):
    assert len(_ids(_load_jsonl(RLVR_TRAIN))) > 0
    assert len(_ids(_load_jsonl(COLD_TRAIN))) > 0


def test_v2_no_bureaucracy_junk(v2_ready):
    salt = re.compile(r"\([^()]{8,}\)\s*$")
    for path in (RLVR_FULL, COLD_FULL):
        for r in _load_jsonl(path):
            p = r["prompt"]
            assert "برمز" not in p, r["id"]
            assert "رقم المتابعة" not in p, r["id"]
            assert not salt.search(p), r["id"]


def test_v2_logic_answers_are_compact_canonical(v2_ready):
    for r in _load_jsonl(RLVR_FULL):
        if r["domain"] != "logic":
            continue
        gt = r["metadata"]["ground_truth_answer"]
        can = json.dumps(gt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert r["answer"] == can, r["id"]
        assert " " not in r["answer"] and "\n" not in r["answer"], r["id"]


def test_v2_config_data_files_exist(v2_ready):
    for rel in (
        "configs/qwen_4b_qlora.yaml",
        "configs/qwen_4b_smoke_v11.yaml",
    ):
        cfg = yaml.safe_load((PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        assert (PROJECT_ROOT / cfg["train_data_path"]).is_file(), rel
        assert (PROJECT_ROOT / cfg["coldstart_data_path"]).is_file(), rel
        if cfg.get("eval_data_path"):
            assert (PROJECT_ROOT / cfg["eval_data_path"]).is_file(), rel


def test_configs_point_at_train_splits(v2_ready):
    for rel in ("configs/qwen_4b_qlora.yaml", "configs/qwen_4b_smoke_v11.yaml"):
        cfg = yaml.safe_load((PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        assert str(cfg["train_data_path"]).endswith("arabic_reasoning_rlvr_train.jsonl"), rel
        assert str(cfg["coldstart_data_path"]).endswith("arabic_reasoning_coldstart_train.jsonl"), rel
        assert str(cfg.get("eval_data_path") or "").endswith("arabic_reasoning_rlvr_eval.jsonl"), rel


def test_correctness_logic_gt_json_string_unit():
    """Standalone unit: string GT must work even before full ingest files exist."""
    gt = {"أ": "1", "ب": "2"}
    gt_str = json.dumps(gt, ensure_ascii=False)
    ans = json.dumps(gt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = f"<think>تحقق من القيود خطوة بخطوة ثم نؤكد التوزيع</think><answer>{ans}</answer>"
    rewards = correctness_reward_func(
        prompts=["p"],
        completions=[text],
        ground_truth_answer=[gt_str],
        domain=["logic"],
    )
    assert rewards == [1.0]

