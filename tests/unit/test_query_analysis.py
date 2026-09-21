from __future__ import annotations

import pytest

from newschat.domain import ChatTurn, Intent
from newschat.retrieval.query_analysis import QueryAnalyzer


def history(*user_questions: str) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    for q in user_questions:
        turns += [ChatTurn("user", q), ChatTurn("assistant", "An earlier answer [1].")]
    return turns


@pytest.mark.parametrize(
    ("question", "intent", "tickers"),
    [
        ("What is happening with Intel?", Intent.COMPANY, ("INTC",)),
        ("Compare Nvidia and Intel", Intent.COMPARISON, ("NVDA", "INTC")),
        ("Netflix vs Amazon vs Apple", Intent.COMPARISON, ("NFLX", "AMZN", "AAPL")),
        ("Which stocks did Buffett buy?", Intent.MARKET, ()),
        ("How is AWS doing?", Intent.COMPANY, ("AMZN",)),
    ],
)
def test_intent_and_tickers(
    analyzer: QueryAnalyzer, question: str, intent: Intent, tickers: tuple[str, ...]
) -> None:
    plan = analyzer.analyze(question)
    assert plan.intent is intent and plan.tickers == tickers
    assert not plan.is_follow_up and plan.standalone_query == question


def test_question_is_stripped(analyzer: QueryAnalyzer) -> None:
    assert analyzer.analyze("  hello Intel  ").question == "hello Intel"


class TestFollowUps:
    def test_what_about_inherits_the_topic_from_the_previous_question(
        self, analyzer: QueryAnalyzer
    ) -> None:
        plan = analyzer.analyze(
            "What about Amazon?", history("What did Jim Cramer say about Netflix?")
        )
        assert plan.is_follow_up and plan.tickers == ("AMZN",)
        assert "Jim Cramer" in plan.standalone_query
        assert "Netflix" not in plan.standalone_query  # the old company must not leak in

    def test_pronoun_question_inherits_the_company(self, analyzer: QueryAnalyzer) -> None:
        plan = analyzer.analyze(
            "What are analysts saying about its price target?",
            history("Tell me about Intel's breakup"),
        )
        assert plan.is_follow_up and plan.tickers == ("INTC",) and plan.intent is Intent.COMPANY
        assert "Intel" in plan.standalone_query
        assert "breakup" not in plan.standalone_query  # it has its own topic, so no topic leak

    def test_chained_follow_ups_keep_the_original_topic(self, analyzer: QueryAnalyzer) -> None:
        turns = history("What did Jim Cramer say about Netflix?", "What about Amazon?")
        plan = analyzer.analyze("And Nvidia?", turns)
        assert plan.tickers == ("NVDA",) and "Cramer" in plan.standalone_query

    def test_inherits_a_comparison(self, analyzer: QueryAnalyzer) -> None:
        plan = analyzer.analyze("Which of them has more risk?", history("Compare Nvidia and Intel"))
        assert plan.tickers == ("NVDA", "INTC") and plan.intent is Intent.COMPARISON

    def test_no_history_means_no_follow_up(self, analyzer: QueryAnalyzer) -> None:
        plan = analyzer.analyze("What about Amazon?")
        assert not plan.is_follow_up and plan.tickers == ("AMZN",)

    def test_standalone_questions_ignore_history(self, analyzer: QueryAnalyzer) -> None:
        plan = analyzer.analyze(
            "What is IBM doing with Penn State?", history("Tell me about Intel")
        )
        assert not plan.is_follow_up and plan.tickers == ("IBM",)

    def test_only_user_turns_provide_context(self, analyzer: QueryAnalyzer) -> None:
        turns = [ChatTurn("assistant", "Nvidia is great"), ChatTurn("user", "hello")]
        plan = analyzer.analyze("What about its earnings?", turns)
        assert plan.tickers == ()  # nothing to inherit: the assistant's words don't count

    def test_history_without_any_company_leaves_the_question_unanchored(
        self, analyzer: QueryAnalyzer
    ) -> None:
        plan = analyzer.analyze("What about that?", history("What is the market doing?"))
        assert plan.is_follow_up and plan.intent is Intent.MARKET
