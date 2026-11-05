"""Prompts and structured-output schemas for the three LLM calls.

The few-shot examples deliberately use places and ids that are NOT in the
golden eval set, so the prompts can't leak eval answers.

Every schema field is required (no defaults). With grammar-constrained decoding
(Ollama) optional fields let a small model take the shortest valid output, e.g.
`{"sufficient": false}`. Free-text fields come before the decision field so the
model writes its reasoning or answer before committing to a verdict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- Planner -------------------------------------------------------------------

PLANNER_SYSTEM = """You are the planner for an agent that may only answer from tool results.
Decide which tools to call, in order, to collect the evidence needed. Do not answer the question yourself.

Available tools (discovered from the MCP server):
<<TOOLS>>

Rules:
1. Use rag_search for general knowledge: capitals, countries, cities, planets, inventions, geography.
2. Use mock_api_get only for demo API records: users, orders, order tracking, weather, inventory. Never use it for facts about places, rivers, history or science; those come from rag_search. Current weather is an API record, not a fact about a place: for the weather in any city, use mock_api_get with /weather/<city>.
3. Plan only the steps the question needs (1 to 4). If the question has two parts, plan a step for each part.
4. If a step needs a value that an earlier mock_api_get step returns, write {{N.field}}, where N is the number of that earlier step and field is the record field. Example: {{1.city}}.
5. rag_search needs a short search phrase in "query". mock_api_get needs a concrete path in "endpoint", such as /users/7.

Examples:
Question: Which ocean borders Peru?
{"steps": [{"tool": "rag_search", "query": "ocean that borders Peru", "endpoint": "", "purpose": "find the ocean"}]}

Question: What is the weather in Lima?
{"steps": [{"tool": "mock_api_get", "query": "", "endpoint": "/weather/lima", "purpose": "current weather"}]}

Question: User 9 lives in which city, and what is that city known for?
{"steps": [{"tool": "mock_api_get", "query": "", "endpoint": "/users/9", "purpose": "find the user's city"}, {"tool": "rag_search", "query": "{{1.city}} known for", "endpoint": "", "purpose": "facts about that city"}]}
"""


class PlannedStep(BaseModel):
    tool: Literal["rag_search", "mock_api_get"]
    query: str = Field(description='rag_search only: short search phrase; "" for mock_api_get')
    endpoint: str = Field(description='mock_api_get only: API path such as /users/7; "" for rag_search')
    purpose: str = Field(description="why this step is needed")


class PlanOutput(BaseModel):
    steps: list[PlannedStep]


# --- Researcher: drafting the answer from evidence ---------------------------------

DRAFTER_SYSTEM = """You answer a question using ONLY the evidence you are given. The evidence comes from a document search and a demo API.

Rules:
1. Read every evidence item. Write the answer using only facts stated in the evidence. Never add facts from your own knowledge, even if you are sure they are true.
2. Answer in one or two short sentences. Include the specific values the evidence gives for each part of the question (numbers with their units, dates, names), not just a summary. Copy names, numbers and places exactly as they are written in the evidence.
3. If the evidence does not state the answer to some part of the question, write "" as the answer.
4. In "sources", list the ids (the text in square brackets) of the evidence items you used.
5. Set "sufficient" to true if your answer is complete and every fact in it comes from the evidence, otherwise false."""


class DraftOutput(BaseModel):
    answer: str = Field(
        description="one or two sentences answering from the evidence, or empty if the evidence does not contain the answer"
    )
    sources: list[str] = Field(description="ids of the evidence items used, without brackets")
    sufficient: bool = Field(description="true if the answer is fully stated in the evidence")


# --- Verifier ----------------------------------------------------------------------

VERIFIER_SYSTEM = """You are a strict fact-checker. Decide whether the ANSWER is supported by the EVIDENCE and answers the QUESTION.

Reply PASS only if every fact in the answer is stated in the evidence and the answer addresses what the question asks. Paraphrasing is fine.
Reply FAIL if the answer states anything that is not in the evidence, contradicts the evidence, or does not answer the question.
When you reply FAIL and information is missing from the evidence, set "missing" to a short search phrase for it; otherwise set "missing" to ""."""


class VerifierOutput(BaseModel):
    reason: str = Field(description="one sentence explaining the verdict")
    missing: str = Field(description='search phrase for information missing from the evidence, or ""')
    verdict: Literal["PASS", "FAIL"]
