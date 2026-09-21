from __future__ import annotations

import pytest

from newschat.ingestion.chunking import chunk_text, split_sentences

TEXT = " ".join(f"Sentence number {i} talks about company results and outlook." for i in range(30))


def test_split_sentences_handles_quotes_and_numbers() -> None:
    parts = split_sentences('He said "Buy." Then 5 analysts agreed. Shares rose 3.5% today.')
    assert parts == ['He said "Buy."', "Then 5 analysts agreed.", "Shares rose 3.5% today."]


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("", 200) == []
    assert chunk_text("   ", 200) == []


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("One short sentence.", 500) == ["One short sentence."]


@pytest.mark.parametrize("max_chars", [200, 350, 600, 1100])
def test_chunks_never_exceed_the_limit_and_are_never_empty(max_chars: int) -> None:
    chunks = chunk_text(TEXT, max_chars, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(0 < len(c) <= max_chars for c in chunks)


def test_no_sentence_is_lost() -> None:
    chunks = chunk_text(TEXT, 300, overlap_sentences=1)
    joined = " ".join(chunks)
    for sentence in split_sentences(TEXT):
        assert sentence in joined


def test_overlap_repeats_the_boundary_sentence() -> None:
    chunks = chunk_text(TEXT, 300, overlap_sentences=1)
    last_of_first = split_sentences(chunks[0])[-1]
    assert last_of_first in chunks[1]


def test_zero_overlap_produces_disjoint_chunks() -> None:
    chunks = chunk_text(TEXT, 300, overlap_sentences=0)
    assert " ".join(chunks) == TEXT


def test_overlong_sentence_is_hard_split() -> None:
    long_sentence = "word " * 200 + "end."
    chunks = chunk_text(long_sentence, 100)
    assert all(len(c) <= 100 for c in chunks) and len(chunks) > 1


def test_unbroken_token_is_split_mid_word_as_last_resort() -> None:
    chunks = chunk_text("x" * 250, 100)
    assert [len(c) for c in chunks] == [100, 100, 50]


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 1), (100, -1)])
def test_invalid_arguments_raise(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="must"):
        chunk_text("text", max_chars, overlap)


def test_overlap_that_would_overflow_is_dropped() -> None:
    # Two 60-char sentences and a 90-char limit: keeping overlap + next sentence can't fit.
    a, b, c = "A" * 59 + ".", "B" * 59 + ".", "C" * 59 + "."
    chunks = chunk_text(f"{a} {b} {c}", 90, overlap_sentences=1)
    assert chunks == [a, b, c]
