"""
Utility functions for RLVR Pipeline.
"""

from __future__ import annotations


def format_arabic_terminal_text(text: str) -> str:
    """Reshape Arabic text and apply BiDi algorithm for clean terminal printing."""
    if not text:
        return text
    # Strip ASCII box table border characters that distort BiDi layout
    clean_text = text.replace("│", "").replace("┌", "").replace("┐", "").replace("└", "").replace("┘", "")
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        reshaped = arabic_reshaper.reshape(clean_text)
        return get_display(reshaped)
    except Exception:
        return clean_text
