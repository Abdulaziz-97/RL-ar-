"""The Qwen3.5 empty-think injection must be stripped from generation prompts."""

from rlvr_sota.trainer import _EMPTY_THINK_INJECTION, strip_think_injection


SNIPPET = (
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is true %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- else %}\n"
    "        " + _EMPTY_THINK_INJECTION + "\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
)


def test_injection_removed():
    fixed = strip_think_injection(SNIPPET)
    assert "<think>\\n\\n</think>" not in fixed
    # The open-think branch (enable_thinking=true) is untouched; we never pass it.
    assert "{{- '<think>\\n' }}" in fixed


def test_noop_when_absent():
    assert strip_think_injection("plain template") == "plain template"
    assert strip_think_injection(None) is None
    assert strip_think_injection("") == ""


def test_rendered_prompt_has_no_preclosed_think():
    import jinja2

    env = jinja2.Environment()
    tpl = env.from_string(strip_think_injection(SNIPPET))
    out = tpl.render(add_generation_prompt=True)
    assert "</think>" not in out
    assert out.strip().endswith("<|im_start|>assistant")
