"""``newschat-eval``: run the retrieval golden set and (optionally) enforce thresholds."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from newschat.bootstrap import build_corpus, build_retriever
from newschat.config import Settings
from newschat.domain import ChatTurn, Corpus, Intent, QueryPlan
from newschat.evaluation.golden import load_golden
from newschat.evaluation.runner import evaluate, format_report
from newschat.ingestion.entities import EntityRegistry
from newschat.retrieval.query_analysis import QueryAnalyzer
from newschat.retrieval.retriever import HybridRetriever, RetrievalConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("evals/golden_set.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-hit", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument("--min-gate", type=float, default=0.0)
    parser.add_argument(
        "--ablation", action="store_true", help="compare BM25-only, dense-only and hybrid"
    )
    args = parser.parse_args(argv)

    settings = Settings()
    registry = EntityRegistry()
    corpus = build_corpus(settings, registry)
    report = evaluate(
        load_golden(args.golden),
        corpus,
        QueryAnalyzer(registry),
        build_retriever(corpus, settings),
        args.k,
    )
    print(format_report(report))
    if args.ablation:
        print("\n" + _ablation(args.golden, corpus, registry, settings, args.k))

    failed = [
        name
        for name, value, floor in (
            ("hit rate", report.hit_rate, args.min_hit),
            ("MRR", report.mrr, args.min_mrr),
            ("gate accuracy", report.gate_accuracy, args.min_gate),
        )
        if value < floor
    ]
    if failed:
        print(f"\nFAILED thresholds: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


class _NoRoutingAnalyzer(QueryAnalyzer):
    """Ablation: ignore companies and history; every question is a plain market-wide search."""

    def analyze(self, question: str, history: Sequence[ChatTurn] = ()) -> QueryPlan:
        return replace(super().analyze(question, ()), intent=Intent.MARKET, tickers=())


def _ablation(
    golden: Path, corpus: Corpus, registry: EntityRegistry, settings: Settings, k: int
) -> str:
    """Compare retrieval variants on the same golden set."""
    cases = load_golden(golden)
    base = RetrievalConfig.from_settings(settings)
    full = QueryAnalyzer(registry)
    variants: list[tuple[str, QueryAnalyzer, RetrievalConfig]] = [
        ("hybrid (BM25 + LSA)", full, base),
        ("  BM25 only", full, replace(base, semantic_weight=0.0)),
        ("  dense (LSA) only", full, replace(base, lexical_weight=0.0)),
        ("hybrid, no entity routing", _NoRoutingAnalyzer(registry), base),
    ]
    hit_col = f"hit@{k}"
    lines = [
        f"{'variant':<27} {hit_col:>6} {'MRR':>6} {'cmp-cov':>8} {'follow-up hit/MRR':>18}"
        f" {'paraphrase hit/MRR':>19}"
    ]
    for name, analyzer, cfg in variants:
        retriever = HybridRetriever(corpus, config=cfg, lsa_components=settings.lsa_components)
        report = evaluate(cases, corpus, analyzer, retriever, k)
        cats = report.by_category()
        _, f_hit, f_mrr = cats.get("follow_up", (0, float("nan"), float("nan")))
        _, p_hit, p_mrr = cats.get("paraphrase", (0, float("nan"), float("nan")))
        lines.append(
            f"{name:<27} {report.hit_rate:>6.3f} {report.mrr:>6.3f} {report.ticker_coverage:>8.3f}"
            f" {f_hit:>11.2f} /{f_mrr:>5.2f} {p_hit:>12.2f} /{p_mrr:>5.2f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
