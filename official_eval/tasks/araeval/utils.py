from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import Iterable
from typing import Any


PUBLIC_TEST_COUNTS = {
    "ien_mcq": 9_990,
    "ien_tf": 5_823,
    "aramath": 605,
    "etec": 1_887,
    "arapro": 5_001,
    "truthfulqa": 536,
    "araifeval": 536,
}

RANDOM_BASELINES = {
    "ien_mcq": 30.77,
    "ien_tf": 50.0,
    "arapro": 25.0,
    "aramath": 25.0,
    "etec": 25.0,
    "truthfulqa": 23.46,
    "araifeval": 0.0,
}

_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_QUOTED_RE = re.compile(r'["“”«](.+?)["“”»]')
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)

_ARABIC_NUMBERS = {
    "واحد": 1,
    "واحدة": 1,
    "الأولى": 1,
    "الاولى": 1,
    "اثنان": 2,
    "اثنين": 2,
    "اثنتان": 2,
    "اثنتين": 2,
    "الثانية": 2,
    "ثلاث": 3,
    "ثلاثة": 3,
    "الثالثة": 3,
    "أربع": 4,
    "اربعة": 4,
    "أربعة": 4,
    "الرابعة": 4,
    "خمس": 5,
    "خمسة": 5,
    "الخامسة": 5,
    "ست": 6,
    "ستة": 6,
    "السادسة": 6,
    "سبع": 7,
    "سبعة": 7,
    "السابعة": 7,
    "ثمان": 8,
    "ثمانية": 8,
    "الثامنة": 8,
    "تسع": 9,
    "تسعة": 9,
    "التاسعة": 9,
    "عشر": 10,
    "عشرة": 10,
    "العاشرة": 10,
}


def _labels(count: int) -> list[str]:
    if not 1 <= count <= len(_LABELS):
        raise ValueError(f"Unsupported choice count: {count}")
    return _LABELS[:count]


def _format_mcq(question: str, choices: list[str], *, context: str = "") -> str:
    lines: list[str] = []
    if context.strip():
        lines.append(f"السياق: {context.strip()}")
    lines.append(f"السؤال: {question.strip()}")
    lines.extend(f"{label}. {choice.strip()}" for label, choice in zip(_labels(len(choices)), choices))
    lines.append("الإجابة:")
    return "\n".join(lines)


def _strip_numbered_choice(choice: str) -> str:
    return re.sub(r"^\s*(?:\d+|[A-Z])[\.\)]\s*", "", str(choice)).strip()


def _strip_parenthesized_choice(choice: str) -> str:
    return re.sub(r"^\s*\([A-Z]\)\s*", "", str(choice)).strip()


def normalize_ien_mcq(row: dict[str, Any]) -> dict[str, Any]:
    choices = [_strip_numbered_choice(choice) for choice in row["Choices"]]
    labels = _labels(len(choices))
    answer = str(row["Answer"]).strip().upper()
    gold = int(answer) - 1 if answer.isdigit() else labels.index(answer)
    return {
        "query": _format_mcq(str(row["Question"]), choices),
        "choices": labels,
        "gold": gold,
    }


def normalize_ien_tf(row: dict[str, Any]) -> dict[str, Any]:
    raw_choices = row["Choices"]
    if isinstance(raw_choices, str):
        raw_choices = ast.literal_eval(raw_choices)
    choices = [_strip_numbered_choice(choice) for choice in raw_choices]
    labels = _labels(len(choices))
    return {
        "query": _format_mcq(str(row["Question"]), choices),
        "choices": labels,
        "gold": int(row["Answer"]) - 1,
    }


def normalize_aramath(row: dict[str, Any]) -> dict[str, Any]:
    choices = [_strip_parenthesized_choice(choice) for choice in row["options"]]
    labels = _labels(len(choices))
    return {
        "query": _format_mcq(
            str(row["question"]), choices, context=str(row.get("passage", ""))
        ),
        "choices": labels,
        "gold": labels.index(str(row["label"]).strip().upper()),
    }


def normalize_etec(row: dict[str, Any]) -> dict[str, Any]:
    choices = [str(choice) for choice in row["choices"]]
    return {
        "query": _format_mcq(str(row["question"]), choices),
        "choices": _labels(len(choices)),
        "gold": int(row["label"]) - 1,
    }


def normalize_arapro(row: dict[str, Any]) -> dict[str, Any]:
    choices = [str(row[f"choice{i}"]) for i in range(1, 5)]
    return {
        "query": _format_mcq(str(row["question"]), choices),
        "choices": _labels(len(choices)),
        "gold": int(row["answer"]) - 1,
    }


def normalize_truthfulqa(row: dict[str, Any]) -> dict[str, Any]:
    choices = [re.sub(r"^[A-Z]\.", "", str(choice)) for choice in row["options"]]
    labels = _labels(len(choices))
    instruction = str(row.get("instruction", "")).strip()
    question = str(row.get("input", "")).strip()
    query = "\n".join(part for part in (instruction, question, "الإجابة:") if part)
    return {
        "query": query,
        "choices": choices,
        "gold": labels.index(str(row["label"]).strip().upper()),
    }


def _sample_dataset(dataset: Any, task: str) -> Any:
    encoded_counts = os.environ.get("ARAEVAL_SAMPLE_COUNTS")
    if not encoded_counts:
        return dataset
    counts = json.loads(encoded_counts)
    if task not in counts:
        raise KeyError(f"Missing sample count for {task}")
    count = int(counts[task])
    if not 0 <= count <= len(dataset):
        raise ValueError(f"Invalid sample count for {task}: {count}")
    seed = int(os.environ.get("ARAEVAL_SAMPLE_SEED", "42"))
    return dataset.shuffle(seed=seed).select(range(count))


def process_ien_mcq(dataset: Any) -> Any:
    return _sample_dataset(dataset, "araeval_ien_mcq").map(normalize_ien_mcq)


def process_ien_tf(dataset: Any) -> Any:
    return _sample_dataset(dataset, "araeval_ien_tf").map(normalize_ien_tf)


def process_aramath(dataset: Any) -> Any:
    return _sample_dataset(dataset, "araeval_aramath").map(normalize_aramath)


def process_etec(dataset: Any) -> Any:
    return _sample_dataset(dataset, "araeval_etec").map(normalize_etec)


def process_arapro(dataset: Any) -> Any:
    return _sample_dataset(dataset, "araeval_arapro").map(normalize_arapro)


def process_truthfulqa(dataset: Any) -> Any:
    return _sample_dataset(dataset, "araeval_truthfulqa").map(normalize_truthfulqa)


def normalize_araifeval(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("instruction_following_prompt", row)
    return {
        "prompt": str(nested["prompt"]),
        "categories": list(nested["categories"]),
    }


def process_araifeval(dataset: Any) -> Any:
    return _sample_dataset(dataset, "araeval_ifeval").map(normalize_araifeval)


def _quoted(text: str) -> list[str]:
    return [match.strip() for match in _QUOTED_RE.findall(text)]


def _number(text: str, default: int | None = None) -> int:
    match = re.search(r"\d+", text)
    if match:
        return int(match.group())
    if re.search(
        r"(?:جملت(?:ان|ين)|كلمت(?:ان|ين)|فقرت(?:ان|ين)|نقطت(?:ان|ين)|قسم(?:ان|ين))",
        text,
    ):
        return 2
    for token, number in _ARABIC_NUMBERS.items():
        if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text):
            return number
    if default is None:
        raise ValueError(f"No number found in instruction: {text}")
    return default


def _instruction_clause(prompt: str, anchor: str) -> str:
    index = prompt.find(anchor)
    if index < 0:
        return prompt
    prior_ends = [prompt.rfind(mark, 0, index) for mark in (".", "!", "؟", "?")]
    start = max(prior_ends) + 1
    following = [
        position
        for mark in (".", "!", "؟", "?")
        if (position := prompt.find(mark, index)) >= 0
    ]
    end = min(following) if following else len(prompt)
    return prompt[start:end].strip()


def _numbered_clause(prompt: str, anchor: str) -> str:
    clauses = re.split(r"(?<=[.!؟?])\s+", prompt)
    numbered: list[str] = []
    for clause in clauses:
        if anchor not in clause:
            continue
        try:
            _number(clause)
        except ValueError:
            continue
        numbered.append(clause)
    return numbered[-1] if numbered else _instruction_clause(prompt, anchor)


def _paragraphs(response: str) -> list[str]:
    normalized = re.sub(r"\n\s*\*\s+\*\s+\*\s*\n", "\n\n", response.strip())
    return [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]


def _sentences(response: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!؟?]+", response.strip()) if part.strip()]


def _words(response: str) -> list[str]:
    return _WORD_RE.findall(response)


def _bullet_lines(response: str) -> list[str]:
    return [
        line
        for line in response.splitlines()
        if re.match(r"^\s*(?:[-*•]|\d+[\.\)])\s+\S", line)
    ]


def _frequency_target(prompt: str, noun: str) -> tuple[str, int]:
    clause = _instruction_clause(prompt, noun)
    quoted = _quoted(clause)
    if not quoted:
        raise ValueError(f"No quoted {noun} target in: {prompt}")
    return quoted[0], _number(clause)


def _count_requirement(prompt: str, anchor: str) -> tuple[int, bool]:
    clause = _numbered_clause(prompt, anchor)
    at_least = any(
        phrase in clause
        for phrase in ("على الأقل", "ما لا يقل", "لا تقل", "كحد أدنى")
    )
    return _number(clause), at_least


def check_instruction(category: str, prompt: str, response: str) -> bool:
    text = response.strip()

    if category == "check_end":
        clause = _instruction_clause(prompt, "أنهي")
        targets = _quoted(clause) or _quoted(prompt)
        return bool(targets) and text.endswith(targets[0])

    if category == "first_word_in_i-th_paragraph":
        clause = _instruction_clause(prompt, "تبدأ الفقرة")
        expected = (_quoted(clause) or _quoted(prompt))[0]
        index = _number(clause) - 1
        paragraphs = _paragraphs(text)
        return 0 <= index < len(paragraphs) and paragraphs[index].startswith(expected)

    if category == "forbidden_words":
        clause = _instruction_clause(prompt, "لا تقم بتضمين")
        targets = _quoted(clause)
        return bool(targets) and all(target not in text for target in targets)

    if category == "include_keywords":
        clause = _instruction_clause(prompt, "قم بتضمين")
        targets = _quoted(clause)
        return bool(targets) and all(target in text for target in targets)

    if category == "json_format":
        try:
            json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return True

    if category == "keyword_frequency":
        target, minimum = _frequency_target(prompt, "الكلمة")
        return text.count(target) >= minimum

    if category == "letter_frequency":
        target, minimum = _frequency_target(prompt, "الحرف")
        return text.count(target) >= minimum

    if category == "minimum_number_highlighted_section":
        minimum, _ = _count_requirement(prompt, "تسليط الضوء")
        highlighted = re.findall(r"(?:\*\*|__)(.+?)(?:\*\*|__)", text, re.DOTALL)
        return len(highlighted) >= minimum

    if category == "multiple_sections":
        marker_index = prompt.find("ضع علامة على بداية كل قسم")
        count_source = prompt[:marker_index] if marker_index >= 0 else prompt
        numbered_clauses = []
        for candidate in re.split(r"(?<=[.!؟?])\s+", count_source):
            try:
                _number(candidate)
            except ValueError:
                continue
            numbered_clauses.append(candidate)
        clause = numbered_clauses[-1] if numbered_clauses else count_source
        minimum = _number(clause)
        marker_clause = _instruction_clause(prompt, "فاصل")
        quoted = _quoted(marker_clause)
        if quoted:
            actual = len([part for part in text.split(quoted[-1]) if part.strip()])
        else:
            named_headers = len(
                re.findall(r"(?m)^\s*(?:#{1,6}\s*)?(?:القسم|قسم)\b", text)
            )
            markdown_sections = len(
                [part for part in re.split(r"(?m)^\s*#{3,}\s*$", text) if part.strip()]
            )
            actual = max(named_headers, markdown_sections)
        return actual >= minimum

    if category == "no_commas":
        return "،" not in text and "," not in text

    if category == "number_bullets":
        count, at_least = _count_requirement(prompt, "نقاط")
        actual = len(_bullet_lines(text))
        return actual >= count if at_least else actual == count

    if category == "number_paragraphs":
        count, at_least = _count_requirement(prompt, "فقر")
        actual = len(_paragraphs(text))
        return actual >= count if at_least else actual == count

    if category == "number_placeholder":
        count, _ = _count_requirement(prompt, "مواضع")
        return len(re.findall(r"\[[^\[\]\n]+\]", text)) >= count

    if category == "number_sentences_at_least":
        count, _ = _count_requirement(prompt, "جمل")
        return len(_sentences(text)) >= count

    if category == "number_sentences_at_most":
        count = _number(_numbered_clause(prompt, "جمل"))
        return len(_sentences(text)) <= count

    if category == "number_words_at_least":
        count = _number(_numbered_clause(prompt, "كلم"))
        return len(_words(text)) >= count

    if category == "number_words_at_most":
        count = _number(_numbered_clause(prompt, "كلم"))
        return len(_words(text)) <= count

    if category == "postscript":
        clause = _instruction_clause(prompt, "ملاحظة")
        markers = _quoted(clause) or _quoted(prompt)
        if not markers:
            return False
        last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
        return last_line.startswith(markers[0])

    if category == "quotation":
        return len(text) >= 2 and text[0] in '"“«' and text[-1] in '"”»'

    if category == "repeat_prompt":
        marker = "أولاً، كرر المدخل دون تغيير، ثم قدم إجابتك"
        before, found, after = prompt.partition(marker)
        if not found:
            marker = "أولا، كرر المدخل دون تغيير، ثم قدم إجابتك"
            before, found, after = prompt.partition(marker)
        original = before.strip()
        if not original and ":" in after:
            original = after.split(":", 1)[1].strip()
        return bool(original) and text.startswith(original)

    if category == "response_language":
        clause = _instruction_clause(prompt, "باللغة")
        if any(word in clause for word in ("الإنجليزية", "الانجليزية", "English")):
            return bool(_LATIN_RE.search(text)) and not _ARABIC_RE.search(text)
        return bool(_ARABIC_RE.search(text)) and not _LATIN_RE.search(text)

    if category == "title":
        return bool(re.search(r"<<[^<>\n]+>>", text))

    if category == "two_responses":
        clause = _instruction_clause(prompt, "افصل")
        quoted = _quoted(clause)
        separator = quoted[-1] if quoted else "******"
        parts = text.split(separator)
        return len(parts) == 2 and all(part.strip() for part in parts)

    raise KeyError(f"Unsupported AraEval category: {category}")


def _loose_candidates(response: str) -> Iterable[str]:
    yield response
    lines = response.splitlines()
    if len(lines) > 1:
        yield "\n".join(lines[1:])
        yield "\n".join(lines[:-1])
    yield response.replace("*", "")


def process_ifeval_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    response = results[0]
    categories = list(doc["categories"])
    strict = [check_instruction(category, doc["prompt"], response) for category in categories]
    loose = [
        any(
            check_instruction(category, doc["prompt"], candidate)
            for candidate in _loose_candidates(response)
        )
        for category in categories
    ]
    return {
        "prompt_level_strict_acc": all(strict),
        "inst_level_strict_acc": strict,
        "prompt_level_loose_acc": all(loose),
        "inst_level_loose_acc": loose,
    }


def aggregate_instruction_accuracy(items: list[list[bool]]) -> float:
    flattened = [value for item in items for value in item]
    return sum(flattened) / len(flattened) if flattened else 0.0


def normalize_against_random(raw_score: float, baseline: float) -> float:
    return 100.0 * (raw_score - baseline) / (100.0 - baseline)


def araeval_overall(raw_scores: dict[str, float]) -> float:
    missing = set(RANDOM_BASELINES) - set(raw_scores)
    if missing:
        raise KeyError(f"Missing AraEval scores: {sorted(missing)}")
    return sum(
        normalize_against_random(raw_scores[name], baseline)
        for name, baseline in RANDOM_BASELINES.items()
    ) / len(RANDOM_BASELINES)
