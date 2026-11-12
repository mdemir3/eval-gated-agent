"""Graph tests with a scripted LLM: every routing path, no model required.

`ScriptedLLM` returns canned structured outputs per schema; `FakeTools` returns
canned tool results. Tool specs come from the real MCP server so the planner's
parsing of the tool descriptions is tested against what the server publishes.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable

import pytest
from langgraph.errors import GraphRecursionError

import agent.graph as graph_module
from agent.graph import (
    MAX_GRAPH_STEPS,
    MAX_PLAN_STEPS,
    MAX_TOOL_CALLS,
    RETRY_TOP_K,
    api_resources,
    graph_steps,
    max_llm_calls,
    resolve_placeholders,
    resolve_sources,
    route_after_verifier,
    run_agent,
    validate_plan,
)
from agent.mcp_client import MCPToolClient, ToolResult
from agent.prompts import DraftOutput, PlanOutput, PlannedStep, VerifierOutput
from agent.state import FALLBACK_MESSAGE, MAX_RETRIES, AgentAnswer
from tools.mcp_server.rag import Retriever, load_corpus
from tools.mcp_server.server import create_server

CORPUS = {doc_id: body for doc_id, _title, body in load_corpus()}
BOTH_TOOLS = {"rag_search", "mock_api_get"}


@pytest.fixture(scope="module")
def tool_specs():
    async def go():
        async with MCPToolClient(create_server()) as client:
            return await client.list_tools()

    return asyncio.run(go())


# --- test doubles ----------------------------------------------------------------------


def rag_step(query: str) -> PlannedStep:
    return PlannedStep(tool="rag_search", query=query, endpoint="", purpose="search")


def api_step(endpoint: str) -> PlannedStep:
    return PlannedStep(tool="mock_api_get", query="", endpoint=endpoint, purpose="fetch")


def draft(answer: str, *sources: str) -> DraftOutput:
    return DraftOutput(answer=answer, sources=list(sources), sufficient=True)


INSUFFICIENT = DraftOutput(answer="", sources=[], sufficient=False)
PASS = VerifierOutput(reason="supported", missing="", verdict="PASS")


def fail(missing: str = "") -> VerifierOutput:
    return VerifierOutput(reason="not supported", missing=missing, verdict="FAIL")


class ScriptedLLM:
    """Plan / drafts / verdicts are consumed in order; the last one repeats."""

    def __init__(
        self,
        plan: list[PlannedStep] | Exception,
        drafts: list[DraftOutput] | Callable[[str], DraftOutput],
        verdicts: list[VerifierOutput] = (PASS,),
    ) -> None:
        self.plan = plan
        self.drafts = drafts
        self.verdicts = list(verdicts)
        self.calls = 0
        self.prompts: dict[str, list[str]] = defaultdict(list)

    @staticmethod
    def _next(queue: list[Any]) -> Any:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def generate(self, schema, system: str, user: str):
        self.calls += 1
        self.prompts[schema.__name__].append(user)
        if schema is PlanOutput:
            if isinstance(self.plan, Exception):
                raise self.plan
            return PlanOutput(steps=self.plan)
        if schema is DraftOutput:
            return self.drafts(user) if callable(self.drafts) else self._next(self.drafts)
        if schema is VerifierOutput:
            return self._next(self.verdicts)
        raise AssertionError(f"unexpected schema {schema}")


def rag_result(*doc_ids: str) -> ToolResult:
    hits = [{"doc_id": d, "chunk_id": f"{d}#0", "title": d, "text": CORPUS[d], "score": 0.7} for d in doc_ids]
    return ToolResult(ok=True, data={"query": "q", "results": hits})


def api_result(endpoint: str, record: dict[str, Any]) -> ToolResult:
    return ToolResult(ok=True, data={"endpoint": endpoint, "status": 200, "data": record})


def api_error(status: int = 500) -> ToolResult:
    return ToolResult(ok=False, error=f"GET failed: {status} error")


class FakeTools:
    def __init__(self, specs, handler: Callable[[str, dict[str, Any], int], ToolResult]) -> None:
        self.specs = specs
        self.handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self):
        return self.specs

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((name, arguments))
        return self.handler(name, arguments, len(self.calls))


def run(question: str, llm: ScriptedLLM, tools) -> dict[str, Any]:
    return asyncio.run(run_agent(question, llm, tools))


def verdict_checks(state) -> list[str]:
    return [e["detail"]["check"] for e in state["trace"] if e["event"] == "verdict"]


# --- end-to-end routing ---------------------------------------------------------------------


def test_happy_path_answers_with_sources(tool_specs):
    llm = ScriptedLLM([rag_step("capital of Canada")], [draft("Ottawa is the capital of Canada.", "[capital-canada]")])
    tools = FakeTools(tool_specs, lambda *_: rag_result("capital-canada", "capital-norway"))
    state = run("What is the capital of Canada?", llm, tools)

    final = AgentAnswer.model_validate(state["final"])
    assert final.status == "answered"
    assert final.answer == "Ottawa is the capital of Canada."
    assert final.sources == ["capital-canada"]
    assert (final.verification_verdict, final.retries, final.tools_used) == ("PASS", 0, ["rag_search"])
    assert [e["node"] for e in state["trace"]][:2] == ["planner", "researcher"]
    assert verdict_checks(state) == ["llm"]


def test_verifier_fail_triggers_one_retry_with_wider_search(tool_specs):
    llm = ScriptedLLM(
        [rag_step("capital of Canada")],
        [draft("Ottawa.")],
        verdicts=[fail(missing="Canadian capital city"), PASS],
    )
    tools = FakeTools(tool_specs, lambda *_: rag_result("capital-canada"))
    state = run("What is the capital of Canada?", llm, tools)

    assert state["final"]["status"] == "answered"
    assert state["retries"] == 1
    retry_call = state["tool_calls"][-1]
    assert retry_call["args"] == {"query": "Canadian capital city", "top_k": RETRY_TOP_K}
    assert retry_call["pass_no"] == 1
    assert "A verifier rejected your previous answer" in llm.prompts["DraftOutput"][-1]


def test_fallback_after_max_retries_never_guesses(tool_specs):
    llm = ScriptedLLM([rag_step("capital of Australia")], [INSUFFICIENT])
    tools = FakeTools(tool_specs, lambda *_: rag_result("capital-canada", "capital-brazil"))
    state = run("What is the capital of Australia?", llm, tools)

    final = state["final"]
    assert final["status"] == "fallback"
    assert final["retries"] == MAX_RETRIES
    assert final["verification_verdict"] == "FAIL"
    assert final["answer"].startswith(FALLBACK_MESSAGE)
    assert "Sources consulted: capital-canada, capital-brazil" in final["answer"]
    assert final["sources"] == ["capital-canada", "capital-brazil"]
    assert len(llm.prompts["DraftOutput"]) == 1 + MAX_RETRIES
    assert verdict_checks(state) == ["drafter"] * (1 + MAX_RETRIES)
    assert state["trace"][-1]["node"] == "fallback"


def test_groundedness_guard_rejects_invented_name_without_llm_call(tool_specs):
    llm = ScriptedLLM([rag_step("capital of Australia")], [draft("Canberra is the capital of Australia.")])
    tools = FakeTools(tool_specs, lambda *_: rag_result("capital-canada"))
    state = run("What is the capital of Australia?", llm, tools)

    assert state["final"]["status"] == "fallback"
    assert verdict_checks(state) == ["lexical_guard"] * (1 + MAX_RETRIES)
    assert "Canberra" in state["verification_feedback"]
    assert llm.prompts["VerifierOutput"] == []  # the LLM verifier was never asked


def test_partial_answer_with_invented_email_falls_back(tool_specs):
    """golden none-06: the city is in the record, the email isn't. A draft that
    answers both parts by inventing an email must never reach the user."""
    question = "Which city does user 3 live in, and what is their email address?"
    record = {"id": 3, "name": "Riley Demo", "city": "Ottawa", "member_since": 2020}
    llm = ScriptedLLM(
        [api_step("/users/3")],
        [draft("User 3 lives in Ottawa."), draft("Their email address is riley.demo@example.com.")],
    )
    tools = FakeTools(
        tool_specs, lambda name, *_: api_result("/users/3", record) if name == "mock_api_get" else rag_result("capital-canada")
    )
    state = run(question, llm, tools)

    assert state["final"]["status"] == "fallback"
    assert verdict_checks(state) == ["lexical_guard"] * (1 + MAX_RETRIES)
    assert "riley.demo@example.com" in state["verification_feedback"]
    assert llm.prompts["VerifierOutput"] == []  # caught before the LLM verifier


def test_self_contradicting_verifier_pass_is_treated_as_fail(tool_specs):
    question = "What is the email address of user 2?"
    record = {"id": 2, "name": "Sam Demo", "city": "Tokyo", "member_since": 2021}
    contradictory = VerifierOutput(reason="FAIL", missing="user 2's email address", verdict="PASS")
    llm = ScriptedLLM([api_step("/users/2")], [draft("Sam Demo.", "api:/users/2")], verdicts=[contradictory])
    tools = FakeTools(tool_specs, lambda name, *_: api_result("/users/2", record) if name == "mock_api_get" else rag_result("capital-japan"))
    state = run(question, llm, tools)

    assert state["final"]["status"] == "fallback"
    assert verdict_checks(state) == ["verifier_consistency"] * (1 + MAX_RETRIES)
    assert state["missing_info"] == "user 2's email address"


def test_completeness_guard_rejects_half_answer(tool_specs):
    question = "Which city does user 3 live in, and which river is that city on?"
    record = {"id": 3, "name": "Riley Demo", "city": "Ottawa", "member_since": 2020}

    def tools_handler(name, args, _n):
        return api_result("/users/3", record) if name == "mock_api_get" else rag_result("capital-canada")

    # Each part gets an answer, but the second one names nothing specific.
    llm = ScriptedLLM([api_step("/users/3"), rag_step("{{1.city}} river")], [draft("Ottawa"), draft("a river")])
    state = run(question, llm, FakeTools(tool_specs, tools_handler))

    assert state["final"]["status"] == "fallback"
    assert verdict_checks(state)[0] == "completeness_guard"
    assert state["trace"][-2]["detail"]["missing"] == "which river is that city on"


def test_compound_question_is_drafted_part_by_part(tool_specs):
    question = "Which city does user 3 live in, and which river is that city on?"
    record = {"id": 3, "name": "Riley Demo", "city": "Ottawa", "member_since": 2020}

    def tools_handler(name, args, _n):
        return api_result("/users/3", record) if name == "mock_api_get" else rag_result("capital-canada")

    llm = ScriptedLLM(
        [api_step("/users/3"), rag_step("{{1.city}} river")],
        [draft("User 3 lives in Ottawa."), draft("Ottawa is on the Ottawa River.")],
    )
    tools = FakeTools(tool_specs, tools_handler)
    state = run(question, llm, tools)

    final = state["final"]
    assert final["answer"] == "User 3 lives in Ottawa; Ottawa is on the Ottawa River."
    assert final["sources"] == ["api:/users/3", "capital-canada"]
    prompts = llm.prompts["DraftOutput"]
    assert "Answer only this part of the question, in one complete sentence: Which city does user 3 live in?" in prompts[0]
    assert "which river is that city on?" in prompts[1] and "Already answered: User 3 lives in Ottawa" in prompts[1]
    # the placeholder was filled from step 1's record
    assert ("rag_search", {"query": "Ottawa river", "top_k": 3}) in tools.calls


def test_flaky_tool_succeeds_on_single_retry(tool_specs):
    record = {"order_id": 1001, "carrier": "Demo Freight", "current_city": "Paris", "status": "in transit"}

    def handler(name, args, n):
        return api_error(503) if n == 1 else api_result("/tracking/1001", record)

    llm = ScriptedLLM([api_step("/tracking/1001")], [draft("Order 1001 is in Paris.")])
    state = run("Where is order 1001 according to its tracking?", llm, FakeTools(tool_specs, handler))

    assert [(c["attempt"], c["ok"]) for c in state["tool_calls"]] == [(1, False), (2, True)]
    assert state["final"]["status"] == "answered"
    assert state["final"]["sources"] == ["api:/tracking/1001"]


def test_failed_api_call_is_retried_once_then_falls_back_to_rag(tool_specs):
    def handler(name, args, _n):
        return api_error(500) if name == "mock_api_get" else rag_result("capital-canada")

    llm = ScriptedLLM([api_step("/inventory/SKU-42")], [INSUFFICIENT])
    state = run("How many units of SKU-42 are in stock?", llm, FakeTools(tool_specs, handler))

    first_pass = [c for c in state["tool_calls"] if c["pass_no"] == 0]
    assert [(c["tool"], c["attempt"], c["ok"]) for c in first_pass] == [
        ("mock_api_get", 1, False),
        ("mock_api_get", 2, False),
        ("rag_search", 1, True),
    ]
    assert first_pass[2]["fallback_for"].startswith("mock_api_get")
    assert state["final"]["status"] == "fallback"
    assert "api:/inventory/SKU-42 (failed)" in state["final"]["sources"]
    assert "Failed tool calls" in llm.prompts["DraftOutput"][0]


def test_tool_transport_errors_never_crash_the_run(tool_specs):
    llm = ScriptedLLM([rag_step("x")], [INSUFFICIENT])
    tools = FakeTools(tool_specs, lambda *_: ToolResult(ok=False, error="MCP call failed: BrokenPipeError"))
    state = run("What is the capital of Canada?", llm, tools)
    assert state["final"]["status"] == "fallback"
    assert state["final"]["answer"].endswith("Sources consulted: none.")


def test_malformed_planner_output_defaults_to_rag_search(tool_specs):
    llm = ScriptedLLM(ValueError("not JSON"), [draft("Ottawa.")])
    tools = FakeTools(tool_specs, lambda *_: rag_result("capital-canada"))
    state = run("What is the capital of Canada?", llm, tools)

    assert state["plan"] == [
        {"tool": "rag_search", "args": {"query": "What is the capital of Canada?", "top_k": 3}, "purpose": "default search"}
    ]
    assert "ValueError" in state["trace"][0]["detail"]["llm_error"]
    assert state["final"]["status"] == "answered"


def test_drafter_and_verifier_exceptions_lead_to_fallback(tool_specs):
    class Boom(ScriptedLLM):
        async def generate(self, schema, system, user):
            if schema is VerifierOutput:
                raise TimeoutError("provider timeout")
            return await super().generate(schema, system, user)

    llm = Boom([rag_step("capital of Canada")], [draft("Ottawa.")])
    state = run("What is the capital of Canada?", llm, FakeTools(tool_specs, lambda *_: rag_result("capital-canada")))
    assert state["final"]["status"] == "fallback"
    assert "Verifier error (TimeoutError)" in state["verification_feedback"]


def test_real_mcp_tools_flaky_endpoint_end_to_end(tool_specs, index_dir):
    """Scripted LLM + the real MCP server in-process: the flaky endpoint recovers."""
    llm = ScriptedLLM(
        [api_step("/tracking/1001"), rag_step("{{1.current_city}} river")],
        [draft("Order 1001 is in Paris."), draft("The river Seine flows through Paris.")],
    )

    async def go():
        async with MCPToolClient(create_server(retriever=Retriever(index_dir))) as tools:
            return await run_agent(
                "According to its tracking, which city is order 1001 in right now, and which river flows through that city?",
                llm,
                tools,
            )

    state = asyncio.run(go())
    assert [(c["tool"], c["attempt"], c["ok"]) for c in state["tool_calls"]] == [
        ("mock_api_get", 1, False),
        ("mock_api_get", 2, True),
        ("rag_search", 1, True),
    ]
    assert state["tool_calls"][2]["args"]["query"] == "Paris river"
    assert state["final"]["status"] == "answered"
    assert state["final"]["sources"] == ["api:/tracking/1001", "capital-france"]


# --- limits ----------------------------------------------------------------------------------


def test_bounds_follow_from_the_retry_and_plan_limits():
    assert (MAX_RETRIES, MAX_PLAN_STEPS) == (2, 4)
    assert MAX_GRAPH_STEPS == 8  # planner + 3 x (researcher, verifier) + fallback
    assert MAX_TOOL_CALLS == 20  # 4 plan steps x (2 attempts + 2 fallback) + 2 retries x 2 attempts
    assert max_llm_calls("What is the capital of Canada?") == 7
    assert max_llm_calls("Which city does user 3 live in, and which river is that city on?") == 10


def test_verifier_that_always_fails_stops_at_fallback_within_limits(tool_specs):
    question = "What is the capital of Canada?"
    llm = ScriptedLLM([rag_step("capital of Canada")], [draft("Ottawa is the capital of Canada.")], verdicts=[fail()])
    tools = FakeTools(tool_specs, lambda *_: rag_result("capital-canada"))
    state = run(question, llm, tools)

    assert state["final"]["status"] == "fallback"
    assert state["retries"] == MAX_RETRIES == 2
    assert verdict_checks(state) == ["llm"] * (1 + MAX_RETRIES)  # the LLM verifier said FAIL every time
    assert graph_steps(state["trace"]) == MAX_GRAPH_STEPS  # the longest legal path, and it finishes
    # the planned search, then one wider search; the second retry has nothing new to search
    assert len(state["tool_calls"]) == len(tools.calls) == 2 <= MAX_TOOL_CALLS
    assert llm.calls == max_llm_calls(question) == 7  # the LLM-call bound is tight


def test_tool_calls_stay_bounded_when_every_tool_fails(tool_specs):
    llm = ScriptedLLM([api_step(f"/users/{i}") for i in range(1, 6)], [INSUFFICIENT])
    tools = FakeTools(tool_specs, lambda *_: api_error(503))
    state = run("Where do users 1 to 5 live?", llm, tools)

    assert state["final"]["status"] == "fallback"
    assert state["retries"] == MAX_RETRIES
    assert graph_steps(state["trace"]) == MAX_GRAPH_STEPS
    # 4 plan steps (the 5th is cut) x 2 attempts, one rag_search fallback x 2 (the later
    # fallbacks would repeat the same search), one retry search x 2
    assert len(tools.calls) == 12 <= MAX_TOOL_CALLS


def test_step_limit_stops_a_routing_bug(tool_specs, monkeypatch):
    monkeypatch.setattr(graph_module, "route_after_verifier", lambda state: "researcher")  # never ends
    llm = ScriptedLLM([rag_step("capital of Canada")], [INSUFFICIENT])
    tools = FakeTools(tool_specs, lambda *_: rag_result("capital-canada"))

    with pytest.raises(GraphRecursionError):
        run("What is the capital of Canada?", llm, tools)
    # one draft per researcher run: stopped after 4, not LangGraph's default of 25 steps
    assert len(llm.prompts["DraftOutput"]) == MAX_GRAPH_STEPS // 2


# --- planner validation ------------------------------------------------------------------


def test_api_resources_are_parsed_from_the_tool_description(tool_specs):
    assert api_resources(tool_specs) == {"users", "orders", "weather", "tracking", "inventory"}


def test_unknown_endpoint_becomes_rag_search_with_api_context():
    plan, notes = validate_plan(
        [api_step("/users/3"), api_step("/cities/3")], "Which river is user 3's city on?", BOTH_TOOLS, {"users"}
    )
    assert plan[1] == {"tool": "rag_search", "args": {"query": "Which river is user 3's city on? {{1.*}}", "top_k": 3},
                       "purpose": "fetch"}
    assert "unknown resource 'cities'" in notes[0]


@pytest.mark.parametrize("endpoint", ["/users/{id}", "/inventory/{{1.city}} river", "/users", ""])
def test_malformed_endpoints_are_repaired(endpoint):
    plan, notes = validate_plan([api_step(endpoint)], "What does user 3 own?", BOTH_TOOLS, {"users", "inventory"})
    assert plan[0]["tool"] == "rag_search"
    assert notes


def test_bare_slots_point_at_the_latest_api_step():
    plan, _ = validate_plan(
        [api_step("/users/3"), rag_step("river that {} is on"), rag_step("{city} history")],
        "Where does user 3 live?",
        BOTH_TOOLS,
        {"users"},
    )
    assert [s["args"]["query"] for s in plan[1:]] == ["river that {{1.*}} is on", "{{1.city}} history"]


def test_placeholders_are_renumbered_when_a_step_is_dropped():
    plan, notes = validate_plan(
        [rag_step("x"), PlannedStep(tool="rag_search", query="x", endpoint="", purpose="dup"), api_step("/users/3"),
         rag_step("{{3.city}} facts")],
        "Where does user 3 live?",
        BOTH_TOOLS,
        {"users"},
    )
    assert [s["tool"] for s in plan] == ["rag_search", "mock_api_get", "rag_search"]
    assert plan[2]["args"]["query"] == "{{2.city}} facts"
    assert any("duplicate" in n for n in notes)


def test_api_steps_dropped_for_pure_knowledge_questions():
    plan, notes = validate_plan(
        [rag_step("capital of Brazil"), api_step("/users/1")], "Which city replaced Rio as the capital of Brazil?",
        BOTH_TOOLS, {"users"},
    )
    assert [s["tool"] for s in plan] == ["rag_search"]
    assert "mentions no API record" in notes[0]


def test_plan_is_capped_and_never_empty():
    plan, _ = validate_plan([rag_step(f"q{i}") for i in range(10)], "Q?", BOTH_TOOLS)
    assert len(plan) == MAX_PLAN_STEPS
    plan, notes = validate_plan([], "Q?", BOTH_TOOLS)
    assert plan[0]["args"] == {"query": "Q?", "top_k": 3}
    assert "empty plan" in notes[-1]


def test_resolve_placeholders():
    record = {"city": "Ottawa", "name": "Riley Demo", "id": 3}
    assert resolve_placeholders({"query": "{{1.city}} river"}, {1: record}) == ({"query": "Ottawa river"}, [])
    assert resolve_placeholders({"query": "{{1.*}}"}, {1: record}) == ({"query": "Ottawa Riley Demo"}, [])
    assert resolve_placeholders({"endpoint": "/weather/{{2.city}}"}, {}) == ({"endpoint": "/weather/"}, ["{{2.city}}"])


# --- sources and routing ---------------------------------------------------------------


def _ev(*doc_ids: str) -> list[dict[str, Any]]:
    return [{"source_id": d, "tool": "rag_search", "text": CORPUS[d], "score": 0.5} for d in doc_ids]


def test_sources_come_from_lexical_support_not_model_citations():
    evidence = _ev("capital-france", "geo-everest", "planet-mars")
    assert resolve_sources(["[geo-everest]"], "Paris; 1889.", "When was the tower finished?", evidence) == ["capital-france"]


def test_explanation_answers_pick_the_item_with_most_new_words():
    evidence = _ev("planet-jupiter", "planet-mars", "planet-venus")
    answer = "Mars is called the Red Planet because iron oxide, or rust, on its surface gives it a reddish colour."
    assert resolve_sources([], answer, "Why is Mars called the Red Planet?", evidence) == ["planet-mars"]


def test_citations_are_mapped_when_nothing_matches_lexically():
    evidence = [{"source_id": "api:/users/3", "tool": "mock_api_get", "text": "{}", "score": None}]
    assert resolve_sources(["/api/users/3"], "yes.", "Q?", evidence) == ["api:/users/3"]


@pytest.mark.parametrize(
    ("verdict", "retries", "route"),
    [("PASS", 0, "end"), ("PASS", MAX_RETRIES, "end"), ("FAIL", 0, "researcher"),
     ("FAIL", MAX_RETRIES - 1, "researcher"), ("FAIL", MAX_RETRIES, "fallback")],
)
def test_route_after_verifier(verdict, retries, route):
    assert route_after_verifier({"verification_verdict": verdict, "retries": retries}) == route
