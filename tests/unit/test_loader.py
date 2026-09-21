from __future__ import annotations

import json
from pathlib import Path

import pytest

from newschat.ingestion.loader import DataLoadError, load_raw_articles, parse_raw_articles

GOOD = {"title": "T", "link": "https://x.test/a", "ticker": "AAPL", "full_text": "body"}


def test_valid_payload_parses_and_ignores_extra_fields() -> None:
    parsed, invalid = parse_raw_articles({"aapl": [GOOD]})
    assert invalid == 0
    assert list(parsed) == ["AAPL"]  # ticker keys are normalised to upper case
    assert parsed["AAPL"][0].title == "T"


@pytest.mark.parametrize(
    "bad",
    [
        {"link": "https://x.test/a", "full_text": "b"},  # no title
        {"title": "  ", "link": "https://x.test/a"},  # blank title
        {"title": "T", "link": "ftp://x.test/a"},  # not http(s)
        {"title": "T"},  # no link
        "not-a-dict",
    ],
)
def test_bad_records_are_skipped_and_counted_not_fatal(bad: object) -> None:
    parsed, invalid = parse_raw_articles({"AAPL": [GOOD, bad]})
    assert invalid == 1 and len(parsed["AAPL"]) == 1


def test_missing_full_text_defaults_to_empty() -> None:
    parsed, _ = parse_raw_articles({"AAPL": [{"title": "T", "link": "https://x.test/a"}]})
    assert parsed["AAPL"][0].full_text == ""


@pytest.mark.parametrize("payload", [[], "x", 3, None])
def test_non_object_top_level_is_a_hard_error(payload: object) -> None:
    with pytest.raises(DataLoadError):
        parse_raw_articles(payload)


def test_non_list_value_is_a_hard_error() -> None:
    with pytest.raises(DataLoadError, match="AAPL"):
        parse_raw_articles({"AAPL": {"title": "T"}})


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="not found"):
        load_raw_articles(tmp_path / "nope.json")


def test_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DataLoadError, match="not valid JSON"):
        load_raw_articles(path)


def test_round_trip_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    path.write_text(json.dumps({"NVDA": [GOOD]}), encoding="utf-8")
    parsed, invalid = load_raw_articles(path)
    assert invalid == 0 and parsed["NVDA"][0].link == "https://x.test/a"
