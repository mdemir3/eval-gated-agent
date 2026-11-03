"""Explicit graph state and the agent's structured output."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

FALLBACK_MESSAGE = "I couldn't verify an answer from the available sources."
MAX_RETRIES = 2

TOOL_NAMES = ("rag_search", "mock_api_get")


class PlanStep(TypedDict):
    tool: str  # "rag_search" | "mock_api_get"
    args: dict[str, Any]
    purpose: str


class EvidenceItem(TypedDict):
    source_id: str  # corpus doc id (e.g. "capital-canada") or "api:/users/3"
    tool: str
    text: str
    score: float | None


class ToolCallRecord(TypedDict):
    tool: str
    args: dict[str, Any]
    attempt: int  # 1 = first try, 2 = the single retry
    ok: bool
    error: str | None
    pass_no: int  # researcher pass: 0 = initial, 1..MAX_RETRIES = verifier retries
    fallback_for: str | None  # set when this call replaces a failed tool


class TraceEvent(TypedDict):
    node: str
    event: str
    detail: dict[str, Any]


class AgentState(TypedDict, total=False):
    # --- fields required by the design -------------------------------------
    question: str
    plan: list[PlanStep]
    retrieved_chunks: list[EvidenceItem]  # all evidence: corpus chunks and API records
    tool_calls: list[ToolCallRecord]
    draft_answer: str
    verification_verdict: Literal["PASS", "FAIL", ""]
    retries: int
    sources_consulted: list[str]
    # --- supporting fields -----------------------------------------------------
    draft_sufficient: bool  # the drafter's own claim that evidence answers the question
    draft_sources: list[str]
    draft_missing: str  # the part of a compound question the drafter found unsupported
    verification_feedback: str  # why the last verdict was FAIL; fed back to the Researcher
    missing_info: str  # what the verifier says is missing; used as the retry search query
    final: dict[str, Any]  # AgentAnswer.model_dump()
    trace: Annotated[list[TraceEvent], operator.add]


class AgentAnswer(BaseModel):
    """The structured output the CLI prints and the evals parse."""

    question: str
    status: Literal["answered", "fallback"]
    answer: str = Field(min_length=1)
    sources: list[str]
    verification_verdict: Literal["PASS", "FAIL"]
    retries: int = Field(ge=0, le=MAX_RETRIES)
    tools_used: list[str]
