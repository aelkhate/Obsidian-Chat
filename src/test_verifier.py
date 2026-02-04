# tests/test_verifier.py
# Pytest test cases for Verifier (evaluation-ready)

import pytest
from src.verifier import verify_answer, REFUSAL_TEXT


def _src(n=1):
    # minimal "sources" list mock
    return [{"rel_path": "x.md", "heading": "H", "score": 1.0} for _ in range(n)]


def test_pass_simple_two_sentences():
    ans = "It uses FAISS for retrieval (source: [1]). It uses embeddings (source: [1])."
    v = verify_answer(ans, _src(1), "RAG_QA")
    assert v["status"] == "PASS"


def test_fail_uncited_sentence():
    ans = "It uses FAISS for retrieval. It uses embeddings (source: [1])."
    v = verify_answer(ans, _src(1), "RAG_QA")
    assert v["status"] == "FAIL"
    assert "Uncited" in v["reason"]


def test_fail_invalid_citation_id():
    ans = "It uses FAISS for retrieval (source: [2])."
    v = verify_answer(ans, _src(1), "RAG_QA")
    assert v["status"] == "FAIL"
    assert v["reason"] == "Invalid citation ID"


def test_pass_headings_are_exempt():
    ans = "## Results\nIt uses FAISS for retrieval (source: [1])."
    v = verify_answer(ans, _src(1), "RAG_QA")
    assert v["status"] == "PASS"


def test_pass_bullets_must_be_cited_each():
    ans = "- Project A (source: [1])\n- Project B (source: [1])"
    v = verify_answer(ans, _src(1), "RAG_QA")
    assert v["status"] == "PASS"


def test_fail_bullet_without_citation():
    ans = "- Project A\n- Project B (source: [1])"
    v = verify_answer(ans, _src(1), "RAG_QA")
    assert v["status"] == "FAIL"
    assert "Uncited" in v["reason"]


def test_doc_extract_requires_exactly_one_source():
    ans = "Here are the extracted items (source: [1])."
    v = verify_answer(ans, _src(2), "DOC_EXTRACT")
    assert v["status"] == "FAIL"
    assert "must have exactly one source" in v["reason"]


def test_browse_requires_exactly_one_source():
    ans = "Here is the inventory (source: [1])."
    v = verify_answer(ans, _src(2), "BROWSE")
    assert v["status"] == "FAIL"
    assert "must have exactly one source" in v["reason"]


def test_refusal_requires_empty_sources_and_no_citations():
    ans = REFUSAL_TEXT
    v = verify_answer(ans, [], "RAG_QA")
    assert v["status"] == "PASS"

    v2 = verify_answer(ans + " (source: [1])", [], "RAG_QA")
    assert v2["status"] == "FAIL"
    assert "Refusal contains citations" in v2["reason"]

    v3 = verify_answer(ans, _src(1), "RAG_QA")
    assert v3["status"] == "FAIL"
    assert "Refusal with non-empty sources" in v3["reason"]


def test_fail_non_refusal_with_empty_sources():
    ans = "This is an answer (source: [1])."
    v = verify_answer(ans, [], "RAG_QA")
    assert v["status"] == "FAIL"
    assert "empty sources" in v["reason"].lower()
