"""Deterministic eval checks: no LLM, no API key, same run -> same verdict.

Checks per golden case (N/A where a check doesn't apply):

* format               final output parses as `AgentAnswer` and is internally consistent
* tool_calls           tools called == expected_tools, and every call's arguments are valid
* key_facts            the answer contains every expected key fact (answerable cases)
* lexical_groundedness every expected key fact is supported, by token overlap, by a single
                       retrieved evidence item (answerable cases). Lexical, NOT semantic.
* retrieval            every relevant_doc_id was among the retrieved corpus chunks
* refusal              unanswerable -> Fallback response; answerable -> not a Fallback

Normalisation is implemented here rather than imported from the agent's lexical
guard, so a bug in the agent can't silently make its own eval pass.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent.state import FALLBACK_MESSAGE, AgentAnswer
from tools.mcp_server.mock_data import MockAPIError, resolve_path, route_of

GOLDEN_PATH = Path(__file__).with_name("golden.jsonl")

CHECKS: dict[str, str] = {
    "format": "Output format (parses as AgentAnswer)",
    "tool_calls": "Tool-call correctness (tools + valid args)",
    "key_facts": "Answer contains expected key facts",
    "lexical_groundedness": "Lexical groundedness of key facts",
    "retrieval": "Relevant docs retrieved",
    "refusal": "Refusal behaviour (fallback iff unanswerable)",
}

_STOPWORDS = frozenset("a an and at by for from in into is of on or the to was with".split())


@dataclass
class GoldenCase:
    id: str
    category: str
    answerable: bool
    question: str
    expected_key_facts: list[str]
    relevant_doc_ids: list[str]
    expected_tools: list[str]
    note: str = ""


@dataclass
class CheckResult:
    name: str
    passed: bool | None  # None = not applicable to this case
    detail: str = ""


@dataclass
class CaseResult:
    case: GoldenCase
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None  # agent crashed

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed is not False for c in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.passed is False]


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenCase]:
    with path.open(encoding="utf-8") as f:
        return [GoldenCase(**json.loads(line)) for line in f if line.strip()]


# --- text normalisation --------------------------------------------------------------


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 8,849 -> 8849
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", text)  # 17th -> 17
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _content_tokens(text: str) -> set[str]:
    return {t for t in _fold(text).split() if t not in _STOPWORDS}


def contains_phrase(haystack: str, phrase: str) -> bool:
    return f" {_fold(phrase)} " in f" {_fold(haystack)} "


# --- individual checks --------------------------------------------------------------


def check_format(final: Any) -> CheckResult:
    try:
        answer = AgentAnswer.model_validate_json(json.dumps(final))
    except (ValidationError, TypeError) as exc:
        return CheckResult("format", False, f"does not parse as AgentAnswer: {exc}".splitlines()[0])
    if answer.status == "answered":
        if answer.verification_verdict != "PASS" or not answer.sources:
            return CheckResult("format", False, "answered output must have verdict PASS and >=1 source")
    else:
        if not answer.answer.startswith(FALLBACK_MESSAGE) or "Sources consulted:" not in answer.answer:
            return CheckResult("format", False, "fallback output must be the fixed message plus sources consulted")
        if answer.verification_verdict != "FAIL":
            return CheckResult("format", False, "fallback output must carry verdict FAIL")
    return CheckResult("format", True)


def argument_error(tool: str, args: dict[str, Any]) -> str | None:
    """Why a tool call's arguments are invalid, or None if they are valid."""
    if tool == "rag_search":
        if not set(args) <= {"query", "top_k"}:
            return f"unexpected rag_search args {sorted(args)}"
        query, top_k = args.get("query"), args.get("top_k", 3)
        if not isinstance(query, str) or not query.strip() or "{" in query:
            return f"bad query {query!r}"
        if not isinstance(top_k, int) or not 1 <= top_k <= 10:
            return f"bad top_k {top_k!r}"
        return None
    if tool == "mock_api_get":
        if not set(args) <= {"endpoint", "params"}:
            return f"unexpected mock_api_get args {sorted(args)}"
        endpoint, params = args.get("endpoint"), args.get("params")
        if params is not None and not isinstance(params, dict):
            return f"bad params {params!r}"
        if not isinstance(endpoint, str):
            return f"bad endpoint {endpoint!r}"
        try:
            path = resolve_path(endpoint, params)
        except MockAPIError as exc:
            return str(exc)
        if route_of(path) is None:
            return f"endpoint {endpoint!r} is not part of the API"
        return None
    return f"unknown tool {tool!r}"


def check_tool_calls(case: GoldenCase, state: dict[str, Any]) -> CheckResult:
    calls = state.get("tool_calls", [])
    called = sorted({c["tool"] for c in calls})
    expected = sorted(set(case.expected_tools))
    problems = []
    if called != expected:
        problems.append(f"called {called}, expected {expected}")
    for c in calls:
        if err := argument_error(c["tool"], c["args"]):
            problems.append(f"{c['tool']}: {err}")
    return CheckResult("tool_calls", not problems, "; ".join(problems))


def check_key_facts(case: GoldenCase, final: dict[str, Any]) -> CheckResult:
    if not case.answerable:
        return CheckResult("key_facts", None)
    missing = [f for f in case.expected_key_facts if not contains_phrase(final.get("answer", ""), f)]
    return CheckResult("key_facts", not missing, f"missing {missing}" if missing else "")


def check_lexical_groundedness(case: GoldenCase, state: dict[str, Any]) -> CheckResult:
    if not case.answerable:
        return CheckResult("lexical_groundedness", None)
    evidence = [_content_tokens(e["text"]) for e in state.get("retrieved_chunks", [])]
    unsupported = [f for f in case.expected_key_facts if not any(_content_tokens(f) <= ev for ev in evidence)]
    return CheckResult(
        "lexical_groundedness", not unsupported, f"not in any evidence item: {unsupported}" if unsupported else ""
    )


def check_retrieval(case: GoldenCase, state: dict[str, Any]) -> CheckResult:
    if not case.relevant_doc_ids:
        return CheckResult("retrieval", None)
    retrieved = {e["source_id"] for e in state.get("retrieved_chunks", []) if e["tool"] == "rag_search"}
    missing = [d for d in case.relevant_doc_ids if d not in retrieved]
    return CheckResult("retrieval", not missing, f"not retrieved: {missing}" if missing else "")


def check_refusal(case: GoldenCase, final: dict[str, Any]) -> CheckResult:
    status = final.get("status")
    if case.answerable:
        return CheckResult("refusal", status == "answered", "" if status == "answered" else "refused an answerable question")
    ok = status == "fallback" and final.get("answer", "").startswith(FALLBACK_MESSAGE)
    return CheckResult("refusal", ok, "" if ok else f"answered an unanswerable question: {final.get('answer')!r}")


def evaluate_case(case: GoldenCase, state: dict[str, Any]) -> CaseResult:
    final = state.get("final") or {}
    return CaseResult(
        case,
        [
            check_format(final),
            check_tool_calls(case, state),
            check_key_facts(case, final),
            check_lexical_groundedness(case, state),
            check_retrieval(case, state),
            check_refusal(case, final),
        ],
    )


def summarize(results: list[CaseResult]) -> dict[str, tuple[int, int]]:
    """check name -> (passed, applicable). A crashed case is evaluated on an empty
    state by the runner, so it fails every check that applies to it."""
    summary = {}
    for name in CHECKS:
        applicable = [c for r in results for c in r.checks if c.name == name and c.passed is not None]
        summary[name] = (sum(bool(c.passed) for c in applicable), len(applicable))
    return summary
