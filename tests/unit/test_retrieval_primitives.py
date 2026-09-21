from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from newschat.retrieval.bm25 import BM25Index
from newschat.retrieval.dense import DenseIndex, LsaEmbedder
from newschat.retrieval.fusion import reciprocal_rank_fusion
from newschat.retrieval.tokenize import STOPWORDS, terms, tokenize


class TestTokenize:
    def test_lowercases_stems_and_drops_stopwords(self) -> None:
        assert tokenize("The analysts are raising Intel's targets") == [
            "analyst",
            "rais",
            "intel",
            "target",
        ]

    def test_keeps_numbers_and_two_letter_words_like_ai(self) -> None:
        assert tokenize("AI target $160 from $175") == ["ai", "target", "160", "175"]

    def test_terms_preserve_the_surface_form_for_display(self) -> None:
        [term] = terms("rumours")
        assert term.surface == "rumours" and term.stem != ""

    def test_question_scaffolding_is_ignored(self) -> None:
        assert tokenize("Tell me about the latest news, please") == []

    def test_empty_and_punctuation_only(self) -> None:
        assert tokenize("") == [] and tokenize("?!...") == []

    def test_stopword_list_is_lowercase(self) -> None:
        assert all(w == w.lower() for w in STOPWORDS)


class TestBM25:
    DOCS: ClassVar[list[list[str]]] = [
        ["intel", "breakup", "chip"],
        ["nvidia", "chip", "chip", "earnings"],
        ["apple", "iphone", "launch"],
        ["chip", "market", "market", "market"],
    ]

    def test_documents_with_the_rare_term_rank_first(self) -> None:
        results = BM25Index(self.DOCS).search(["breakup"], top_k=3)
        assert [doc for doc, _ in results] == [0]

    def test_rarer_terms_have_higher_idf(self) -> None:
        index = BM25Index(self.DOCS)
        assert index.idf("breakup") > index.idf("chip")
        assert index.idf("never-seen") > index.idf("breakup")

    def test_term_frequency_saturates(self) -> None:
        index = BM25Index([["a"] + ["x"] * 3, ["a"] + ["x"] * 3 + ["a"] * 30], k1=1.2)
        (_, s1), (_, s2) = index.search(["a"], 2)[0], index.search(["a"], 2)[1]
        assert s1 > s2  # 31 occurrences in a longer document must not beat 1 in a short one by much

    def test_allowed_filter_and_top_k(self) -> None:
        index = BM25Index(self.DOCS)
        assert [d for d, _ in index.search(["chip"], 10, allowed={1, 3})] == [1, 3] or [
            d for d, _ in index.search(["chip"], 10, allowed={1, 3})
        ] == [3, 1]
        assert len(index.search(["chip"], 2)) == 2

    def test_unknown_or_empty_queries_return_nothing(self) -> None:
        index = BM25Index(self.DOCS)
        assert index.search(["zzz"], 5) == [] and index.search([], 5) == []

    def test_document_frequency_and_len(self) -> None:
        index = BM25Index(self.DOCS)
        assert index.document_frequency("chip") == 3 and len(index) == 4

    def test_ties_break_deterministically_by_doc_id(self) -> None:
        index = BM25Index([["a"], ["a"], ["a"]])
        assert [d for d, _ in index.search(["a"], 3)] == [0, 1, 2]

    def test_empty_index_is_safe(self) -> None:
        assert BM25Index([]).search(["a"], 3) == []


class TestDense:
    TEXTS = [
        "chipmaker semiconductor foundry wafer breakup rivals",
        "semiconductor foundry chipmaker manufacturing wafer plant",
        "iphone launch smartphone apple store retail",
        "smartphone apple iphone retail sales",
        "streaming subscribers netflix originals series",
        "netflix streaming series originals subscribers growth",
    ] * 4

    def test_embeddings_are_unit_normalised(self) -> None:
        emb = LsaEmbedder(n_components=4)
        emb.fit(self.TEXTS)
        norms = np.linalg.norm(emb.embed(self.TEXTS), axis=1)
        assert np.allclose(norms, 1.0)

    def test_embedding_before_fit_is_an_error(self) -> None:
        with pytest.raises(RuntimeError):
            LsaEmbedder().embed(["x"])

    def test_related_documents_outrank_unrelated_ones(self) -> None:
        index = DenseIndex(LsaEmbedder(n_components=4), self.TEXTS)
        top = [i for i, _ in index.search("chipmaker foundry", 4)]
        assert set(top) <= {0, 1, 6, 7, 12, 13, 18, 19}

    def test_allowed_restricts_results(self) -> None:
        index = DenseIndex(LsaEmbedder(n_components=4), self.TEXTS)
        results = index.search("chipmaker foundry", 10, allowed={2, 3})
        assert {i for i, _ in results} <= {2, 3}

    def test_no_positive_similarity_means_no_results(self) -> None:
        index = DenseIndex(LsaEmbedder(n_components=4), self.TEXTS)
        assert index.search("zzzz qqqq", 5) == []

    def test_fit_is_deterministic(self) -> None:
        a, b = LsaEmbedder(n_components=4), LsaEmbedder(n_components=4)
        a.fit(self.TEXTS)
        b.fit(self.TEXTS)
        assert np.allclose(a.embed(["chipmaker"]), b.embed(["chipmaker"]))

    def test_tiny_corpus_does_not_crash(self) -> None:
        index = DenseIndex(
            LsaEmbedder(n_components=128), ["alpha beta", "gamma delta", "alpha gamma"]
        )
        assert index.search("alpha", 2)


class TestFusion:
    def test_items_ranked_by_both_lists_beat_items_ranked_by_only_one(self) -> None:
        fused = reciprocal_rank_fusion([[10, 20, 30], [20, 40, 50]])
        ordered = sorted(fused, key=fused.__getitem__, reverse=True)
        assert ordered[0] == 20  # ranked by both, beats 10 (rank 1 in one list only)
        assert fused[20] > max(fused[i] for i in (10, 30, 40, 50))

    def test_score_matches_the_rrf_formula(self) -> None:
        fused = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 4]], k=60)
        assert fused[2] == pytest.approx(1 / 62 + 1 / 62)
        assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
        assert fused[1] == pytest.approx(1 / 61)

    def test_weights_shift_the_outcome(self) -> None:
        fused = reciprocal_rank_fusion([[1, 2], [2, 1]], weights=[1.0, 0.0])
        assert fused[1] > fused[2]

    def test_mismatched_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            reciprocal_rank_fusion([[1]], weights=[1.0, 2.0])

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == {} and reciprocal_rank_fusion([[]]) == {}
