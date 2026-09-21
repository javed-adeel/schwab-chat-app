"""Sentence-aware chunking.

Chunks are packed from whole sentences up to ``max_chars`` so a retrieved passage never
starts or ends mid-sentence, with a small sentence overlap so facts that straddle a
boundary stay retrievable.
"""

from __future__ import annotations

import re

# A boundary is whitespace after terminal punctuation (optionally followed by a closing
# quote/bracket, which stays attached to the sentence it closes) and before a sentence start.
_SENTENCE_BOUNDARY = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z0-9\"“‘(])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Break a single over-long sentence on spaces (or mid-word as a last resort)."""
    parts: list[str] = []
    while len(sentence) > max_chars:
        cut = sentence.rfind(" ", 0, max_chars)
        cut = cut if cut > 0 else max_chars
        parts.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    if sentence:
        parts.append(sentence)
    return parts


def chunk_text(text: str, max_chars: int, overlap_sentences: int = 1) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")
    sentences: list[str] = []
    for sentence in split_sentences(text):
        sentences.extend(
            _hard_split(sentence, max_chars) if len(sentence) > max_chars else [sentence]
        )

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if current and size + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            size = sum(len(s) + 1 for s in current)
            if size + len(sentence) + 1 > max_chars:  # overlap would overflow: drop it
                current, size = [], 0
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
