"""CLI entry point.

    python -m agent.main "What is the capital of Canada?"
    python -m agent.main "..." --trace     # full decision trace
    python -m agent.main "..." --json      # the structured AgentAnswer only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from dotenv import load_dotenv

from agent.graph import run_agent
from agent.llm import LangChainLLM, LLMConfigError
from agent.mcp_client import MCPToolClient
from agent.state import AgentState


async def answer(question: str, llm: LangChainLLM | None = None) -> AgentState:
    llm = llm or LangChainLLM.from_env()
    async with MCPToolClient.stdio() as tools:
        return await run_agent(question, llm, tools)


def _call_sig(tool: str, args: dict[str, Any]) -> str:
    return f"{tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


def format_summary(state: AgentState) -> str:
    final = state["final"]
    lines = [f"Answer: {final['answer']}"]
    if final["status"] == "answered":
        lines.append(f"Sources: {', '.join(final['sources'])}")
    lines += ["", "Plan:"]
    for i, step in enumerate(state.get("plan", []), 1):
        purpose = f"  - {step['purpose']}" if step.get("purpose") else ""
        lines.append(f"  {i}. {_call_sig(step['tool'], step['args'])}{purpose}")

    lines.append("Tools used:")
    grouped: dict[str, list[str]] = {}
    for call in state.get("tool_calls", []):
        outcome = "ok" if call["ok"] else f"failed ({call['error']})"
        note = " [fallback]" if call.get("fallback_for") else ""
        grouped.setdefault(_call_sig(call["tool"], call["args"]) + note, []).append(f"attempt {call['attempt']} {outcome}")
    for sig, attempts in grouped.items():
        lines.append(f"  - {sig}: {'; '.join(attempts)}")
    if not grouped:
        lines.append("  (none)")
    lines.append(f"Verification: {state.get('verification_verdict') or 'n/a'} (retries: {state.get('retries', 0)})")
    if final["status"] == "fallback":
        lines.append(f"Last verifier feedback: {state.get('verification_feedback', '')}")
    return "\n".join(lines)


def format_trace(state: AgentState) -> str:
    lines = ["", "Full decision trace:"]
    for i, ev in enumerate(state.get("trace", []), 1):
        detail = json.dumps(ev["detail"], ensure_ascii=False, default=str)
        lines.append(f"  {i:>2}. [{ev['node']}] {ev['event']}: {detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the eval-gated agent a question.")
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true", help="print the full decision trace")
    parser.add_argument("--json", action="store_true", help="print only the structured answer as JSON")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        state = asyncio.run(answer(args.question))
    except LLMConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(state["final"], indent=2, ensure_ascii=False))
    else:
        print(format_summary(state))
    if args.trace:
        print(format_trace(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
