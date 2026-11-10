"""Tests for the eval harness itself: the golden data is valid, every check can
fail, and the judge never fabricates a score without a key."""

from __future__ import annotations

import json

import pytest

from agent.state import FALLBACK_MESSAGE
from evals.deterministic import (
    CHECKS,
    GoldenCase,
    argument_error,
    contains_phrase,
    evaluate_case,
    load_golden,
    summarize,
)
from evals.llm_judge import NOT_RUN_NO_KEY, run_judge
from tools.mcp_server import mock_data
from tools.mcp_server.rag import load_corpus

GOLDEN = load_golden()
CORPUS = {doc_id: f"{title} {body}" for doc_id, title, body in load_corpus()}
API_TEXT = json.dumps([mock_data.USERS, mock_data.ORDERS, mock_data.WEATHER, mock_data.TRACKING])


# --- golden dataset validity -------------------------------------------------------------


def test_golden_set_shape():
    ids = [c.id for c in GOLDEN]
    assert len(GOLDEN) >= 20 and len(set(ids)) == len(ids)
    assert sum(not c.answerable for c in GOLDEN) >= 5
    assert sum(c.category == "multi_tool" for c in GOLDEN) >= 5
    for case in GOLDEN:
        assert case.answerable == (case.category != "unanswerable")
        assert set(case.expected_tools) <= {"rag_search", "mock_api_get"}
        assert case.answerable == bool(case.expected_key_facts)


def test_golden_doc_ids_exist():
    for case in GOLDEN:
        assert set(case.relevant_doc_ids) <= CORPUS.keys(), case.id


@pytest.mark.parametrize("case", [c for c in GOLDEN if c.answerable], ids=lambda c: c.id)
def test_every_key_fact_is_in_the_corpus_or_api_data(case):
    """A key fact the sources don't contain would make the agent's correct
    behaviour (refusing to guess) fail the eval."""
    sources = [CORPUS[d] for d in case.relevant_doc_ids] + [API_TEXT]
    for fact in case.expected_key_facts:
        assert any(contains_phrase(text, fact) for text in sources), f"{case.id}: {fact!r}"


# --- each check can pass and fail ----------------------------------------------------------


CASE = GoldenCase(
    id="t", category="multi_tool", answerable=True, question="Which city does user 3 live in, and which river?",
    expected_key_facts=["Ottawa", "Ottawa River"], relevant_doc_ids=["capital-canada"],
    expected_tools=["mock_api_get", "rag_search"],
)
UNANSWERABLE = GoldenCase(
    id="u", category="unanswerable", answerable=False, question="What is the capital of Australia?",
    expected_key_facts=[], relevant_doc_ids=[], expected_tools=["rag_search"],
)


def good_state() -> dict:
    return {
        "question": CASE.question,
        "tool_calls": [
            {"tool": "mock_api_get", "args": {"endpoint": "/users/3"}, "attempt": 1, "ok": True},
            {"tool": "rag_search", "args": {"query": "Ottawa river", "top_k": 3}, "attempt": 1, "ok": True},
        ],
        "retrieved_chunks": [
            {"source_id": "api:/users/3", "tool": "mock_api_get", "text": json.dumps(mock_data.USERS[3]), "score": None},
            {"source_id": "capital-canada", "tool": "rag_search", "text": CORPUS["capital-canada"], "score": 0.7},
        ],
        "final": {
            "question": CASE.question, "status": "answered", "answer": "Ottawa; the Ottawa River.",
            "sources": ["api:/users/3", "capital-canada"], "verification_verdict": "PASS", "retries": 0,
            "tools_used": ["mock_api_get", "rag_search"],
        },
    }


def fallback_state(question: str) -> dict:
    return {
        "tool_calls": [{"tool": "rag_search", "args": {"query": question, "top_k": 3}, "attempt": 1, "ok": True}],
        "retrieved_chunks": [],
        "final": {
            "question": question, "status": "fallback", "answer": f"{FALLBACK_MESSAGE} Sources consulted: none.",
            "sources": [], "verification_verdict": "FAIL", "retries": 2, "tools_used": ["rag_search"],
        },
    }


def failed(result) -> set[str]:
    return {c.name for c in result.failed_checks}


def test_good_answer_passes_every_check():
    result = evaluate_case(CASE, good_state())
    assert result.passed, result.failed_checks


def test_missing_key_fact_fails():
    state = good_state()
    state["final"]["answer"] = "Ottawa."
    assert failed(evaluate_case(CASE, state)) == {"key_facts"}


def test_key_fact_absent_from_evidence_fails_groundedness():
    state = good_state()
    state["retrieved_chunks"] = state["retrieved_chunks"][:1]  # drop capital-canada
    assert failed(evaluate_case(CASE, state)) == {"lexical_groundedness", "retrieval"}


def test_wrong_tools_or_invalid_args_fail_tool_calls():
    state = good_state()
    state["tool_calls"][0]["args"] = {"endpoint": "/cities/3"}
    assert failed(evaluate_case(CASE, state)) == {"tool_calls"}
    state = good_state()
    state["tool_calls"] = state["tool_calls"][1:]
    assert failed(evaluate_case(CASE, state)) == {"tool_calls"}


def test_refusing_an_answerable_question_fails():
    state = fallback_state(CASE.question)
    state["tool_calls"] = good_state()["tool_calls"]
    assert {"refusal", "key_facts"} <= failed(evaluate_case(CASE, state))


def test_answering_an_unanswerable_question_fails():
    state = good_state()
    state["tool_calls"] = state["tool_calls"][1:]
    state["final"]["answer"] = "Canberra."
    assert failed(evaluate_case(UNANSWERABLE, state)) == {"refusal"}


def test_correct_refusal_passes():
    assert evaluate_case(UNANSWERABLE, fallback_state(UNANSWERABLE.question)).passed


@pytest.mark.parametrize(
    "final",
    [
        {},
        {"status": "answered", "answer": "x"},
        {**good_state()["final"], "verification_verdict": "FAIL"},
        {**good_state()["final"], "sources": []},
        {**fallback_state("q")["final"], "answer": "I don't know."},
    ],
)
def test_malformed_output_fails_format(final):
    state = good_state()
    state["final"] = final
    assert "format" in failed(evaluate_case(CASE, state))


def test_crashed_case_fails_every_applicable_check():
    result = evaluate_case(CASE, {})
    assert failed(result) == set(CHECKS)


@pytest.mark.parametrize(
    ("tool", "args", "ok"),
    [
        ("rag_search", {"query": "x", "top_k": 3}, True),
        ("rag_search", {"query": "x"}, True),
        ("rag_search", {"query": "{{1.city}} river", "top_k": 3}, False),
        ("rag_search", {"query": "x", "top_k": 50}, False),
        ("mock_api_get", {"endpoint": "/users/{id}", "params": {"id": 3}}, True),
        ("mock_api_get", {"endpoint": "/users/{id}"}, False),
        ("mock_api_get", {"endpoint": "/patents/1"}, False),
        ("web_search", {"q": "x"}, False),
    ],
)
def test_argument_validation(tool, args, ok):
    assert (argument_error(tool, args) is None) == ok


def test_summary_counts_only_applicable_checks():
    results = [evaluate_case(CASE, good_state()), evaluate_case(UNANSWERABLE, fallback_state("q"))]
    summary = summarize(results)
    assert summary["key_facts"] == (1, 1)  # N/A for the unanswerable case
    assert summary["refusal"] == (2, 2)


# --- LLM judge --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {"LLM_PROVIDER": "ollama", "LLM_MODEL": "llama3.2:3b"},
        {"LLM_PROVIDER": "openai", "LLM_MODEL": "some-model"},  # hosted provider but no key
        {"JUDGE_PROVIDER": "anthropic", "JUDGE_MODEL": "some-model"},
    ],
)
def test_judge_does_not_run_without_a_key(monkeypatch, env):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "JUDGE_PROVIDER", "JUDGE_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    report = run_judge([("rag-01", good_state())], threshold=0.7)
    assert not report.ran
    assert report.status == NOT_RUN_NO_KEY
    assert report.results == [] and report.passed
