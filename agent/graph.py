"""LangGraph agent: Planner -> Researcher -> Verifier -> (END | Researcher | Fallback).

* Planner     turns the question into tool steps (LLM, structured output), then a
              deterministic validator repairs or drops anything malformed.
* Researcher  executes the steps over MCP. Each failed call is retried once;
              a failed API call then falls back to rag_search. It then drafts an
              answer from the evidence only, one part at a time for compound
              questions. Nothing in here raises.
* Verifier    groundedness guard (no invented names/numbers), completeness guard
              (every part of a compound question answered), then an LLM fact-check
              (a PASS that still names missing information counts as FAIL).
              FAIL with retries left loops back to the Researcher with feedback.
* Fallback    a fixed refusal plus the sources consulted. Never a guess.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from langgraph.graph import END, START, StateGraph

from agent.lexical import (
    COMMON_WORDS,
    answer_fact_terms,
    fact_seeking_parts,
    question_parts,
    salient_terms,
    tokens,
    unsupported_terms,
)
from agent.llm import StructuredLLM
from agent.mcp_client import ToolResult, ToolSpec
from agent.prompts import (
    DRAFTER_SYSTEM,
    PLANNER_SYSTEM,
    VERIFIER_SYSTEM,
    DraftOutput,
    PlanOutput,
    PlannedStep,
    VerifierOutput,
)
from agent.state import (
    FALLBACK_MESSAGE,
    MAX_RETRIES,
    AgentAnswer,
    AgentState,
    EvidenceItem,
    PlanStep,
    ToolCallRecord,
    TraceEvent,
)

MAX_PLAN_STEPS = 4
DEFAULT_TOP_K = 3
RETRY_TOP_K = 5
MAX_EVIDENCE_IN_PROMPT = 8

# Hard bounds on one run. The graph stays within them by construction; run_agent
# enforces the step bound, and the eval report checks every case against all of them.
# Node runs: the planner, a researcher + verifier pair per pass (1 + MAX_RETRIES passes),
# then the fallback.
MAX_GRAPH_STEPS = 1 + 2 * (1 + MAX_RETRIES) + 1
# Tool calls: each plan step gets 2 attempts, plus 2 for the rag_search fallback when an
# API call fails twice; each retry pass runs one search with 2 attempts.
MAX_TOOL_CALLS = MAX_PLAN_STEPS * 4 + MAX_RETRIES * 2
# Each node logs exactly one of these events per run, so they count node runs.
_NODE_EVENTS = frozenset({"plan", "draft", "verdict", "refused"})


def max_llm_calls(question: str) -> int:
    """The planner once, then per pass one draft per question part plus one verifier call."""
    return 1 + (len(question_parts(question)) + 1) * (1 + MAX_RETRIES)


def graph_steps(trace: list[TraceEvent]) -> int:
    """Node runs recorded in a run's trace."""
    return sum(1 for event in trace if event["event"] in _NODE_EVENTS)


# {{N.field}} = a field of step N's API record; {{N.*}} = all its text fields.
_PLACEHOLDER = re.compile(r"\{\{?\s*(\d+)\s*\.\s*(\*|[A-Za-z_]\w*)\s*\}?\}")
# A bare slot the planner sometimes writes instead of {{N.field}}: "{}", "{city}", "{{city}}".
_BARE_SLOT = re.compile(r"\{\{?\s*([A-Za-z_]\w*)?\s*\}?\}")
# Question words that can justify a demo-API call. A purely general-knowledge
# question ("Which city replaced Rio de Janeiro ...?") never needs one, however
# the planner feels about it.
API_HINT_WORDS = frozenset(
    "user users order orders tracking shipment shipped weather temperature inventory stock sku".split()
)
# Endpoint templates as listed in the mock_api_get tool description, e.g. "  /users/{id}: ..."
_ENDPOINT_LINE = re.compile(r"^\s*/(\w+)/\{\w+\}:", re.MULTILINE)


class ToolClient(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


def _event(node: str, event: str, **detail: Any) -> TraceEvent:
    return {"node": node, "event": event, "detail": detail}


# --- pure helpers (unit-tested directly) -------------------------------------------


def api_resources(specs: list[ToolSpec]) -> set[str]:
    """Resource names (users, orders, ...) advertised in the mock_api_get description."""
    return {m for spec in specs if spec.name == "mock_api_get" for m in _ENDPOINT_LINE.findall(spec.description)}


def _endpoint_problem(endpoint: str, resources: set[str] | None) -> str | None:
    concrete = _PLACEHOLDER.sub("x", endpoint)
    if "{" in concrete or "}" in concrete:
        return "unfilled {placeholder}"
    segments = concrete.strip("/").split("/")
    if len(segments) != 2 or not all(segments):
        return "expected /<resource>/<id>"
    if any(ch.isspace() for ch in segments[1]):
        return "id contains spaces"
    if resources is not None and segments[0] not in resources:
        return f"unknown resource {segments[0]!r}"
    return None


def validate_plan(
    steps: list[PlannedStep], question: str, available: set[str], resources: set[str] | None = None
) -> tuple[list[PlanStep], list[str]]:
    """Turn raw planner output into executable steps. Returns (plan, repair notes).

    An invalid mock_api_get step becomes a rag_search on the question plus the
    text fields of the latest earlier API record, so a made-up endpoint costs
    nothing. API steps are dropped when the question mentions no API record
    type. Placeholders are renumbered when earlier steps are dropped.
    """
    wants_api = bool(tokens(question) & API_HINT_WORDS)
    plan: list[PlanStep] = []
    notes: list[str] = []
    seen: set[str] = set()
    new_index: dict[int, int] = {}  # planner step number -> plan step number
    last_api_step: int | None = None

    def renumber(text: str) -> str:
        def sub(m: re.Match[str]) -> str:
            n = new_index.get(int(m.group(1)))
            return "{{%d.%s}}" % (n, m.group(2)) if n else ""

        def bare(m: re.Match[str]) -> str:  # "{city}" -> a field of the latest API record
            return "{{%d.%s}}" % (last_api_step, m.group(1) or "*") if last_api_step else ""

        return " ".join(_BARE_SLOT.sub(bare, _PLACEHOLDER.sub(sub, text)).split())

    for number, step in enumerate(steps, start=1):
        if step.tool not in available:
            notes.append(f"dropped step {number}: unknown tool {step.tool!r}")
            continue
        tool = step.tool
        if tool == "mock_api_get" and not wants_api:
            notes.append(f"dropped step {number}: question mentions no API record (user, order, weather, ...)")
            continue
        if tool == "mock_api_get":
            endpoint = renumber(step.endpoint.strip())
            problem = _endpoint_problem(endpoint, resources) if endpoint else "no endpoint"
            if problem is None:
                args: dict[str, Any] = {"endpoint": endpoint if endpoint.startswith("/") else "/" + endpoint}
            else:
                notes.append(f"step {number}: invalid endpoint {step.endpoint!r} ({problem}); using rag_search")
                tool = "rag_search"
                context = " {{%d.*}}" % last_api_step if last_api_step else ""
                args = {"query": question + context, "top_k": DEFAULT_TOP_K}
        else:
            args = {"query": renumber(step.query.strip()) or question, "top_k": DEFAULT_TOP_K}
        key = json.dumps([tool, args], sort_keys=True)
        if key in seen:
            notes.append(f"dropped step {number}: duplicate {tool}({args})")
            continue
        if len(plan) == MAX_PLAN_STEPS:
            notes.append(f"dropped step {number}: plan is capped at {MAX_PLAN_STEPS} steps")
            continue
        seen.add(key)
        plan.append({"tool": tool, "args": args, "purpose": step.purpose.strip()})
        new_index[number] = len(plan)
        if tool == "mock_api_get":
            last_api_step = len(plan)
    if not plan:
        notes.append("empty plan: defaulting to rag_search on the question")
        plan = [{"tool": "rag_search", "args": {"query": question, "top_k": DEFAULT_TOP_K}, "purpose": "default search"}]
    return plan, notes


def resolve_placeholders(args: dict[str, Any], step_data: dict[int, Any]) -> tuple[dict[str, Any], list[str]]:
    """Replace {{N.field}} with field values from step N's API record."""
    unresolved: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        record = step_data.get(int(match.group(1)))
        field = match.group(2)
        if isinstance(record, dict) and field in record:
            return str(record[field])
        if isinstance(record, dict):  # "*" or a wrong field name: use the record's text values
            return " ".join(str(v) for v in record.values() if isinstance(v, str))
        unresolved.append(match.group(0))
        return ""

    resolved = {k: _PLACEHOLDER.sub(substitute, v).strip() if isinstance(v, str) else v for k, v in args.items()}
    return resolved, unresolved


def api_text_values(evidence: list[EvidenceItem]) -> list[str]:
    """String fields of API records in the evidence (e.g. a user's city), for retry queries."""
    values: list[str] = []
    for item in evidence:
        if item["tool"] != "mock_api_get":
            continue
        try:
            record = json.loads(item["text"])
        except ValueError:
            continue
        values += [v for v in record.values() if isinstance(v, str) and v not in values]
    return values


def format_evidence(evidence: list[EvidenceItem], failed: list[str]) -> str:
    api = [e for e in evidence if e["tool"] == "mock_api_get"]
    docs = sorted((e for e in evidence if e["tool"] != "mock_api_get"), key=lambda e: -(e["score"] or 0))
    lines = [f"[{e['source_id']}] {e['text']}" for e in (api + docs)[:MAX_EVIDENCE_IN_PROMPT]]
    if failed:
        lines.append("Failed tool calls (no data returned):")
        lines += [f"- {f}" for f in failed]
    return "\n".join(lines) if lines else "(no evidence)"


def ordered_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _id_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def resolve_sources(cited: list[str], answer: str, question: str, evidence: list[EvidenceItem]) -> list[str]:
    """The evidence items an answer rests on, decided lexically rather than by
    trusting the model's citations:

    1. items containing one of the answer's names/numbers that isn't in the question;
    2. else (an explanation like "because iron oxide ...") the item(s) sharing the
       most new content words with the answer;
    3. else the drafter's citations ("[geo-nile]", "/api/users/3", ...) mapped onto ids.
    """
    ids = ordered_unique([e["source_id"] for e in evidence])
    asked = tokens(question)
    new_terms = {t.lower() for t in salient_terms(answer)} - asked
    supporting = [e["source_id"] for e in evidence if new_terms & tokens(e["text"])]
    if supporting:
        return ordered_unique(supporting)
    new_words = tokens(answer) - asked - COMMON_WORDS
    overlap = {e["source_id"]: len(new_words & tokens(e["text"])) for e in evidence}
    best = max(overlap.values(), default=0)
    if best:
        return [i for i in ids if overlap.get(i) == best]
    mapped: list[str] = []
    for raw in cited:
        key = _id_key(raw)
        if key:
            mapped += [i for i in ids if _id_key(i) == key] or [i for i in ids if _id_key(i) in key]
    return ordered_unique(mapped) or ids


# --- nodes ---------------------------------------------------------------------------


class AgentNodes:
    def __init__(self, llm: StructuredLLM, tools: ToolClient) -> None:
        self.llm = llm
        self.tools = tools

    # Planner ----------------------------------------------------------------------
    async def planner(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        specs = await self.tools.list_tools()
        catalog = "\n".join(
            f"- {s.name}({', '.join(s.input_schema.get('properties', {}))}): {s.description}" for s in specs
        )
        try:
            out = await self.llm.generate(PlanOutput, PLANNER_SYSTEM.replace("<<TOOLS>>", catalog), f"Question: {question}")
            raw_steps, error = out.steps, None
        except Exception as exc:  # malformed output, provider error
            raw_steps, error = [], f"{type(exc).__name__}: {exc}"
        plan, notes = validate_plan(raw_steps, question, {s.name for s in specs}, api_resources(specs) or None)
        trace = [_event("planner", "plan", steps=plan, repairs=notes, llm_error=error)]
        return {
            "plan": plan,
            "retries": 0,
            "retrieved_chunks": [],
            "tool_calls": [],
            "sources_consulted": [],
            "verification_verdict": "",
            "verification_feedback": "",
            "missing_info": "",
            "trace": trace,
        }

    # Researcher -------------------------------------------------------------------
    async def researcher(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        is_retry = state.get("verification_verdict") == "FAIL"
        retries = state.get("retries", 0) + (1 if is_retry else 0)
        evidence = list(state.get("retrieved_chunks", []))
        calls = list(state.get("tool_calls", []))
        trace: list[TraceEvent] = []

        steps = self._retry_steps(state, evidence, calls) if is_retry else state["plan"]
        if is_retry:
            trace.append(_event("researcher", "retry", retry=retries, feedback=state.get("verification_feedback"), steps=steps))

        step_data: dict[int, Any] = {}
        for number, step in enumerate(steps, start=1):
            args, unresolved = resolve_placeholders(step["args"], step_data)
            if step["tool"] == "rag_search" and not args.get("query"):
                args["query"] = question
            if unresolved:
                trace.append(_event("researcher", "unresolved_placeholder", step=number, placeholders=unresolved))
            if step["tool"] == "mock_api_get" and unresolved:
                result = ToolResult(ok=False, error="endpoint depends on a step that returned no data")
            else:
                result = await self._call_with_retry(step["tool"], args, retries, calls, trace)
            if result.ok:
                step_data[number] = self._add_evidence(step["tool"], result, evidence)
                continue
            if step["tool"] == "mock_api_get":
                # Fall back to the other tool: maybe the corpus covers it. Text fields of
                # records fetched so far (e.g. a user's city) make the search specific.
                context = " ".join(v for v in api_text_values(evidence) if v.lower() not in question.lower())
                fallback_args = {"query": f"{question} {context}".strip(), "top_k": DEFAULT_TOP_K}
                if not any(c["tool"] == "rag_search" and c["args"] == fallback_args for c in calls):
                    trace.append(_event("researcher", "fallback_tool", failed=args, using="rag_search"))
                    fb = await self._call_with_retry(
                        "rag_search", fallback_args, retries, calls, trace, fallback_for=f"mock_api_get {args}"
                    )
                    if fb.ok:
                        self._add_evidence("rag_search", fb, evidence)
            else:
                trace.append(_event("researcher", "partial_evidence", reason="rag_search failed after retry"))

        failed = [
            f"GET {c['args'].get('endpoint')}: {c['error']}"
            for c in calls
            if c["tool"] == "mock_api_get" and not c["ok"] and c["attempt"] == 2
        ]
        draft, unsupported_part = await self._draft(
            question, evidence, failed, state.get("verification_feedback", "") if is_retry else ""
        )
        trace.append(
            _event("researcher", "draft", sufficient=draft.sufficient, answer=draft.answer, sources=draft.sources,
                   unsupported_part=unsupported_part or None)
        )

        consulted = ordered_unique(
            [e["source_id"] for e in evidence]
            + [f"api:{c['args'].get('endpoint')} (failed)" for c in calls if c["tool"] == "mock_api_get" and not c["ok"] and c["attempt"] == 2]
        )
        return {
            "retries": retries,
            "retrieved_chunks": evidence,
            "tool_calls": calls,
            "draft_answer": draft.answer.strip(),
            "draft_sufficient": draft.sufficient,
            "draft_sources": draft.sources,
            "draft_missing": unsupported_part,
            "sources_consulted": consulted,
            "trace": trace,
        }

    def _retry_steps(self, state: AgentState, evidence: list[EvidenceItem], calls: list[ToolCallRecord]) -> list[PlanStep]:
        """On a verifier FAIL: search again, wider, aimed at what the verifier said is missing."""
        base = state.get("missing_info", "").strip() or state["question"]
        extra = " ".join(v for v in api_text_values(evidence) if v.lower() not in base.lower())
        candidates = [f"{base} {extra}".strip(), f"{state['question']} {extra}".strip()]
        for query in candidates:
            args = {"query": query, "top_k": RETRY_TOP_K}
            if not any(c["tool"] == "rag_search" and c["args"] == args for c in calls):
                return [{"tool": "rag_search", "args": args, "purpose": "retry: widen search"}]
        return []  # nothing new to search; just redraft with the feedback

    async def _call_with_retry(
        self,
        tool: str,
        args: dict[str, Any],
        pass_no: int,
        calls: list[ToolCallRecord],
        trace: list[TraceEvent],
        fallback_for: str | None = None,
    ) -> ToolResult:
        result = ToolResult(ok=False, error="not called")
        for attempt in (1, 2):
            result = await self.tools.call(tool, args)
            calls.append(
                {"tool": tool, "args": args, "attempt": attempt, "ok": result.ok, "error": result.error,
                 "pass_no": pass_no, "fallback_for": fallback_for}
            )
            trace.append(_event("researcher", "tool_call", tool=tool, args=args, attempt=attempt, ok=result.ok, error=result.error))
            if result.ok:
                break
        return result

    @staticmethod
    def _add_evidence(tool: str, result: ToolResult, evidence: list[EvidenceItem]) -> Any:
        existing = {(e["source_id"], e["text"]) for e in evidence}
        if tool == "rag_search":
            for hit in result.data.get("results", []):
                if (hit["doc_id"], hit["text"]) not in existing:
                    evidence.append({"source_id": hit["doc_id"], "tool": tool, "text": hit["text"], "score": hit["score"]})
            return None
        record = result.data.get("data", {})
        item: EvidenceItem = {
            "source_id": f"api:{result.data.get('endpoint')}", "tool": tool, "text": json.dumps(record), "score": None
        }
        if (item["source_id"], item["text"]) not in existing:
            evidence.append(item)
        return record

    async def _draft(
        self, question: str, evidence: list[EvidenceItem], failed: list[str], feedback: str
    ) -> tuple[DraftOutput, str]:
        """Draft from evidence only. A compound question is drafted one part at a
        time (small models reliably answer just the first part otherwise) and the
        part answers are joined; if any part is unsupported, the whole draft is.
        Returns (draft, the unsupported part of a compound question or "")."""
        insufficient = DraftOutput(answer="", sources=[], sufficient=False)
        if not evidence:
            return insufficient, ""
        parts = question_parts(question)
        answers: list[str] = []
        sources: list[str] = []
        for part in parts:
            user = f"Question: {question}\n"
            if len(parts) > 1:
                user += f"Answer only this part of the question, in one complete sentence: {part}?\n"
                if answers:
                    user += f"Already answered: {'; '.join(answers)}\n"
            user += f"\nEvidence:\n{format_evidence(evidence, failed)}"
            if feedback:
                user += (
                    f"\n\nA verifier rejected your previous answer: {feedback}\n"
                    "Write a corrected answer that uses only the evidence, or set sufficient to false."
                )
            try:
                draft = await self.llm.generate(DraftOutput, DRAFTER_SYSTEM, user)
            except Exception:
                draft = DraftOutput(answer="", sources=[], sufficient=False)
            if not draft.sufficient or not draft.answer.strip():
                return insufficient, part if len(parts) > 1 else ""
            answer = draft.answer.strip().rstrip(".")
            first_word = answer.split(maxsplit=1)[0]
            if answers and first_word.lower() in COMMON_WORDS:  # "Paris; the river Seine", not "; The river"
                answer = first_word.lower() + answer[len(first_word):]
            answers.append(answer)
            sources += draft.sources
        joined = "; ".join(answers)
        return DraftOutput(answer=joined[0].upper() + joined[1:] + ".", sources=sources, sufficient=True), ""

    # Verifier ---------------------------------------------------------------------
    async def verifier(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        draft = state.get("draft_answer", "")
        evidence = state.get("retrieved_chunks", [])
        missing = ""
        check = "drafter"
        if not state.get("draft_sufficient") or not draft:
            verdict, feedback = "FAIL", "The evidence was judged insufficient to answer the question."
            if missing := state.get("draft_missing", ""):
                feedback = f"The evidence was judged insufficient to answer this part: {missing}."
        elif unsupported := unsupported_terms(draft, " ".join(e["text"] for e in evidence), question):
            check = "lexical_guard"
            verdict = "FAIL"
            feedback = f"The answer uses terms that appear nowhere in the evidence: {', '.join(unsupported)}."
        elif (
            len(parts := question_parts(question)) > 1
            and len(facts := answer_fact_terms(draft, question, api_text_values(evidence)))
            < len(asks := fact_seeking_parts(parts))
        ):
            check = "completeness_guard"
            verdict = "FAIL"
            feedback = (
                f"The question asks for {len(asks)} things ({'; '.join(asks)}) but the answer states only "
                f"{len(facts)} specific fact(s) ({', '.join(facts) or 'none'}). Answer every part."
            )
            missing = asks[-1]
        else:
            check = "llm"
            user = f"QUESTION: {question}\n\nEVIDENCE:\n{format_evidence(evidence, [])}\n\nANSWER: {draft}"
            try:
                out = await self.llm.generate(VerifierOutput, VERIFIER_SYSTEM, user)
                verdict, feedback, missing = out.verdict, out.reason, out.missing.strip()
                if verdict == "PASS" and missing:
                    # A PASS that still names missing information contradicts itself (the prompt
                    # requires missing="" on PASS). Small models mix up the fields; don't guess.
                    check = "verifier_consistency"
                    verdict = "FAIL"
                    feedback = f"The verifier passed the answer but said information is missing: {missing}."
            except Exception as exc:
                verdict, feedback = "FAIL", f"Verifier error ({type(exc).__name__}); treating as unverified."

        update: dict[str, Any] = {
            "verification_verdict": verdict,
            "verification_feedback": "" if verdict == "PASS" else feedback,
            "missing_info": missing,
            "trace": [_event("verifier", "verdict", verdict=verdict, check=check, feedback=feedback, missing=missing)],
        }
        if verdict == "PASS":
            update["final"] = AgentAnswer(
                question=question,
                status="answered",
                answer=draft,
                sources=resolve_sources(state.get("draft_sources", []), draft, question, evidence),
                verification_verdict="PASS",
                retries=state.get("retries", 0),
                tools_used=ordered_unique([c["tool"] for c in state.get("tool_calls", [])]),
            ).model_dump()
        return update

    # Fallback ---------------------------------------------------------------------
    async def fallback(self, state: AgentState) -> dict[str, Any]:
        consulted = state.get("sources_consulted", [])
        answer = f"{FALLBACK_MESSAGE} Sources consulted: {', '.join(consulted) if consulted else 'none'}."
        final = AgentAnswer(
            question=state["question"],
            status="fallback",
            answer=answer,
            sources=consulted,
            verification_verdict="FAIL",
            retries=state.get("retries", 0),
            tools_used=ordered_unique([c["tool"] for c in state.get("tool_calls", [])]),
        )
        return {"final": final.model_dump(), "trace": [_event("fallback", "refused", sources_consulted=consulted)]}


def route_after_verifier(state: AgentState) -> Literal["end", "researcher", "fallback"]:
    if state.get("verification_verdict") == "PASS":
        return "end"
    if state.get("retries", 0) < MAX_RETRIES:
        return "researcher"
    return "fallback"


def build_graph(llm: StructuredLLM, tools: ToolClient):
    nodes = AgentNodes(llm, tools)
    graph = StateGraph(AgentState)
    graph.add_node("planner", nodes.planner)
    graph.add_node("researcher", nodes.researcher)
    graph.add_node("verifier", nodes.verifier)
    graph.add_node("fallback", nodes.fallback)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "verifier")
    graph.add_conditional_edges(
        "verifier", route_after_verifier, {"end": END, "researcher": "researcher", "fallback": "fallback"}
    )
    graph.add_edge("fallback", END)
    return graph.compile()


async def run_agent(question: str, llm: StructuredLLM, tools: ToolClient) -> AgentState:
    """Run one question through the graph and return the final state (incl. `final`).

    recursion_limit=MAX_GRAPH_STEPS is the tightest LangGraph setting that lets the
    longest legal path finish; a routing bug that loops is stopped one node later with
    GraphRecursionError."""
    graph = build_graph(llm, tools)
    return await graph.ainvoke({"question": question, "trace": []}, config={"recursion_limit": MAX_GRAPH_STEPS})
