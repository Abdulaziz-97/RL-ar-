"""
Hack-pattern and clean-pattern fixtures for the reward-hacking regression suite.

HACK_PATTERNS: known reward-hacking / degenerate patterns that must score below 0.2 composite.
CLEAN_PATTERNS: correct, well-formatted Arabic samples that must score above 0.9 composite.

Each score was hand-verified against the formula:
  clamp(0.6*correct + 0.2*format + 0.2*lang - 0.5*answer_leak - 0.3*structural_leak, 0.0, 1.0)
"""

HACK_PATTERNS = [
    {
        "name": "empty_think_tags_wrong_answer",
        "completion": "<think></think><answer>42</answer>",
        "ground_truth": 99,
        "domain": "math",
        "max_composite": 0.2,
    },
    {
        "name": "gibberish_repeated_tokens_wrong_answer",
        "completion": "<think>blah blah blah blah blah blah blah blah</think><answer>42</answer>",
        "ground_truth": 99,
        "domain": "math",
        "max_composite": 0.2,
    },
    {
        "name": "echo_prompt_repeated_wrong_answer",
        "completion": "<think>2 plus 2 2 plus 2 2 plus 2 2 plus 2</think><answer>2 plus 2</answer>",
        "ground_truth": 4,
        "domain": "math",
        "max_composite": 0.2,
    },
    {
        "name": "direct_answer_leak_repeated_correct_answer",
        "completion": "<think>the answer is the answer is the answer is the answer is</think><answer>42</answer>",
        "ground_truth": 42,
        "domain": "math",
        "max_composite": 0.2,
    },
    {
        "name": "direct_answer_leak_arabic_wrong_answer",
        "completion": "<think>\u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0647\u064a \u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0647\u064a \u0627\u0644\u0625\u062c\u0627\u0628\u0629 \u0647\u064a</think><answer>42</answer>",
        "ground_truth": 99,
        "domain": "math",
        "max_composite": 0.2,
    },
    {
        "name": "structural_leak_reasoning_before_tags_wrong_answer",
        "completion": "First I add two and two to get four. <think></think><answer>4</answer>",
        "ground_truth": 99,
        "domain": "math",
        "max_composite": 0.2,
    },
    {
        "name": "structural_leak_reasoning_after_answer_wrong_answer",
        "completion": "<think></think><answer>4</answer> because two plus two equals four as computed",
        "ground_truth": 99,
        "domain": "math",
        "max_composite": 0.2,
    },
    {
        "name": "english_reasoning_short_wrong_answer",
        "completion": "<think>First I add two and two to get four then multiply</think><answer>5</answer>",
        "ground_truth": 4,
        "domain": "math",
        "max_composite": 0.2,
    },
    {
        "name": "partial_logic_solution_wrong_answer",
        "completion": '<think>A is 1 B is 2 C is 9</think><answer>{"A": 1, "B": 2, "C": 9}</answer>',
        "ground_truth": {"A": 1, "B": 2, "C": 3},
        "domain": "logic",
        "max_composite": 0.2,
    },
]

CLEAN_PATTERNS = [
    {
        "name": "clean_arabic_math_addition",
        "completion": "<think>\u0623\u0648\u0644\u0627 \u0623\u062c\u0645\u0639 \u0627\u0644\u0639\u062f\u062f\u064a\u0646 \u0645\u0639\u0627 \u062b\u0645 \u0623\u062d\u0633\u0628 \u0627\u0644\u0646\u0627\u062a\u062c \u0628\u062f\u0642\u0629 \u0639\u0627\u0644\u064a\u0629 \u062c\u062f\u0627 \u0648\u0623\u062a\u062d\u0642\u0642 \u0645\u0646 \u0635\u062d\u0629 \u0627\u0644\u0639\u0645\u0644\u064a\u0629 \u0627\u0644\u062d\u0633\u0627\u0628\u064a\u0629 \u062e\u0637\u0648\u0629 \u0628\u062e\u0637\u0648\u0629</think><answer>4</answer>",
        "ground_truth": 4,
        "domain": "math",
        "min_composite": 0.9,
    },
    {
        "name": "clean_arabic_math_multiplication",
        "completion": "<think>\u0623\u0642\u0648\u0645 \u0628\u0636\u0631\u0628 \u0627\u0644\u0639\u062f\u062f \u0627\u0644\u0623\u0648\u0644 \u0641\u064a \u0627\u0644\u0639\u062f\u062f \u0627\u0644\u062b\u0627\u0646\u064a \u062b\u0645 \u0623\u0631\u0627\u062c\u0639 \u0627\u0644\u0646\u062a\u064a\u062c\u0629 \u0628\u0639\u0646\u0627\u064a\u0629 \u0643\u0628\u064a\u0631\u0629 \u0648\u0623\u062a\u0623\u0643\u062f \u0645\u0646 \u0635\u062d\u0629 \u0627\u0644\u062d\u0633\u0627\u0628 \u0628\u0634\u0643\u0644 \u062f\u0642\u064a\u0642</think><answer>12</answer>",
        "ground_truth": 12,
        "domain": "math",
        "min_composite": 0.9,
    },
    {
        "name": "clean_arabic_math_subtraction",
        "completion": "<think>\u0623\u0637\u0631\u062d \u0627\u0644\u0639\u062f\u062f \u0627\u0644\u0635\u063a\u064a\u0631 \u0645\u0646 \u0627\u0644\u0639\u062f\u062f \u0627\u0644\u0643\u0628\u064a\u0631 \u062b\u0645 \u0623\u062d\u0633\u0628 \u0627\u0644\u0641\u0627\u0631\u0642 \u0628\u062f\u0642\u0629 \u0639\u0627\u0644\u064a\u0629 \u0648\u0623\u0631\u0627\u062c\u0639 \u0627\u0644\u0639\u0645\u0644\u064a\u0629 \u0628\u0639\u0646\u0627\u064a\u0629 \u062a\u0627\u0645\u0629 \u0644\u0644\u0648\u0635\u0648\u0644 \u0644\u0644\u0646\u062a\u064a\u062c\u0629</think><answer>8</answer>",
        "ground_truth": 8,
        "domain": "math",
        "min_composite": 0.9,
    },
]
