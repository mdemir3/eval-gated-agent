"""Eval runner.

    python -m evals.run --suite deterministic   # gates CI; exits 1 on any failure
    python -m evals.run --suite all             # + LLM-judge when a hosted key is set
    python -m evals.run --cases rag-01,none-03  # subset while iterating

Every case runs against a fresh MCP server process, so cases are independent
(including the flaky endpoint's state). Writes eval-report.md and
eval-results.json (raw states, for debugging).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.graph import run_agent
from agent.llm import LangChainLLM, LLMConfigError
from agent.mcp_client import MCPToolClient
from evals.deterministic import CHECKS, CaseResult, GoldenCase, evaluate_case, load_golden, summarize
from evals.llm_judge import DEFAULT_THRESHOLD, JudgeReport, run_judge


async def run_cases(cases: list[GoldenCase], llm: LangChainLLM) -> list[dict[str, Any]]:
    runs = []
    for i, case in enumerate(cases, 1):
        calls_before, started = llm.calls, time.perf_counter()
        state: dict[str, Any] = {}
        error = None
        try:
            async with MCPToolClient.stdio() as tools:
                state = dict(await run_agent(case.question, llm, tools))
        except Exception as exc:  # a crash is a failed case, not a crashed eval run
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - started
        result = evaluate_case(case, state)
        result.error = error
        runs.append({"result": result, "state": state, "seconds": seconds, "llm_calls": llm.calls - calls_before})
        failed = ", ".join(c.name for c in result.failed_checks) or (error or "")
        print(
            f"[{i:>2}/{len(cases)}] {'PASS' if result.passed else 'FAIL'} {case.id:<9} "
            f"{seconds:5.1f}s retries={state.get('retries', '-')} status={state.get('final', {}).get('status', '-')}"
            + (f"  failed: {failed}" if not result.passed else ""),
            flush=True,
        )
    return runs


def _git_commit() -> str:
    if sha := os.getenv("GITHUB_SHA"):
        return sha[:7]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pct(passed: int, total: int) -> str:
    return f"{100 * passed / total:.0f}%" if total else "n/a"


def render_report(runs: list[dict[str, Any]], judge: JudgeReport | None, llm: LangChainLLM, suite: str, wall: float) -> str:
    results: list[CaseResult] = [r["result"] for r in runs]
    summary = summarize(results)
    n_pass = sum(r.passed for r in results)
    answerable = sum(r.case.answerable for r in results)
    lines = [
        "# Eval report",
        "",
        f"- **Run:** {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} · commit `{_git_commit()}` · suite `{suite}`",
        f"- **Agent LLM:** `{llm.label}` (temperature {os.getenv('LLM_TEMPERATURE') or 0}, seed {os.getenv('LLM_SEED') or 42})",
        f"- **Environment:** Python {platform.python_version()} on {platform.system()} {platform.machine()}"
        + (" (GitHub Actions)" if os.getenv("GITHUB_ACTIONS") else ""),
        f"- **Cases:** {len(results)} ({answerable} answerable, {len(results) - answerable} unanswerable) · "
        f"wall time {wall / 60:.1f} min · {sum(r['llm_calls'] for r in runs)} LLM calls",
        "",
        "## Deterministic suite",
        "",
        f"**{'PASS' if n_pass == len(results) else 'FAIL'}: {n_pass}/{len(results)} cases pass every check.**",
        "",
        "| Check | Passed | Rate |",
        "|---|---|---|",
    ]
    for name, label in CHECKS.items():
        passed, total = summary[name]
        lines.append(f"| {label} | {passed}/{total} | {_pct(passed, total)} |")

    retried = sum(1 for r in runs if r["state"].get("retries"))
    flaky_recovered = sum(
        1 for r in runs for c in r["state"].get("tool_calls", []) if c["attempt"] == 2 and c["ok"]
    )
    fallback_tool = sum(1 for r in runs for c in r["state"].get("tool_calls", []) if c.get("fallback_for") and c["attempt"] == 1)
    lines += [
        "",
        f"Agent behaviour across the run: {retried} case(s) used a verifier retry; "
        f"{flaky_recovered} tool call(s) failed then succeeded on retry; "
        f"{fallback_tool} fallback tool call(s) after a tool failed twice.",
        "",
        "## LLM-judge suite",
        "",
    ]
    if judge is None:
        lines.append("Not requested (`--suite deterministic`).")
    elif not judge.ran:
        lines.append(judge.status)
    else:
        lines += [
            f"Judge `{judge.judge_model}`, threshold {judge.threshold}. "
            f"**{'PASS' if judge.passed else 'FAIL'}**",
            "",
            "| Case | Faithfulness | Answer relevancy | Pass |",
            "|---|---|---|---|",
        ]
        for jr in judge.results:
            f = "error" if jr.error else f"{jr.faithfulness:.2f}"
            a = "error" if jr.error else f"{jr.answer_relevancy:.2f}"
            lines.append(f"| {jr.case_id} | {f} | {a} | {'yes' if jr.passed else 'no'} |")
        scored = [jr for jr in judge.results if not jr.error]
        if scored:
            mean_f = sum(jr.faithfulness for jr in scored) / len(scored)
            mean_a = sum(jr.answer_relevancy for jr in scored) / len(scored)
            lines += ["", f"Mean faithfulness {mean_f:.2f}, mean answer relevancy {mean_a:.2f} over {len(scored)} answered case(s)."]

    lines += ["", "## Cases", "", "| Case | Category | Result | Status | Retries | Tools called | LLM calls | Seconds |", "|---|---|---|---|---|---|---|---|"]
    for run in runs:
        r, st = run["result"], run["state"]
        tools = ", ".join(sorted({c["tool"] for c in st.get("tool_calls", [])})) or "-"
        lines.append(
            f"| {r.case.id} | {r.case.category} | {'PASS' if r.passed else '**FAIL**'} | "
            f"{st.get('final', {}).get('status', 'crash')} | {st.get('retries', '-')} | {tools} | "
            f"{run['llm_calls']} | {run['seconds']:.1f} |"
        )

    failures = [run for run in runs if not run["result"].passed]
    if failures:
        lines += ["", "## Failures", ""]
        for run in failures:
            r, st = run["result"], run["state"]
            lines.append(f"### {r.case.id}: {r.case.question}")
            if r.error:
                lines.append(f"- agent crashed: `{r.error}`")
            for c in r.failed_checks:
                lines.append(f"- **{c.name}**: {c.detail}")
            lines.append(f"- answer: {st.get('final', {}).get('answer', '(none)')!r}")
            lines.append(f"- last verifier feedback: {st.get('verification_feedback', '')!r}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval suites against the golden dataset.")
    parser.add_argument("--suite", choices=["deterministic", "all"], default="deterministic")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--report", default="eval-report.md")
    parser.add_argument("--results", default="eval-results.json")
    parser.add_argument("--judge-threshold", type=float, default=None)
    args = parser.parse_args(argv)

    load_dotenv()
    cases = load_golden()
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.id in wanted]
        if unknown := wanted - {c.id for c in cases}:
            print(f"Unknown case ids: {sorted(unknown)}", file=sys.stderr)
            return 2
    try:
        llm = LangChainLLM.from_env()
    except LLMConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print(f"Running {len(cases)} golden cases with {llm.label} ...", flush=True)
    started = time.perf_counter()
    runs = asyncio.run(run_cases(cases, llm))
    wall = time.perf_counter() - started

    judge = None
    if args.suite == "all":
        threshold = args.judge_threshold or float(os.getenv("JUDGE_THRESHOLD") or DEFAULT_THRESHOLD)
        answered = [(r["result"].case.id, r["state"]) for r in runs if r["state"].get("final", {}).get("status") == "answered"]
        judge = run_judge(answered, threshold)
        print(judge.status if not judge.ran else f"LLM-judge: {judge.status}, {'PASS' if judge.passed else 'FAIL'}")

    report = render_report(runs, judge, llm, args.suite, wall)
    Path(args.report).write_text(report, encoding="utf-8")
    raw = [{"case": r["result"].case.id, "passed": r["result"].passed, "seconds": round(r["seconds"], 2),
            "checks": [c.__dict__ for c in r["result"].checks], "error": r["result"].error, "state": r["state"]}
           for r in runs]
    Path(args.results).write_text(json.dumps(raw, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    all_pass = all(r["result"].passed for r in runs) and (judge is None or judge.passed)
    n_pass = sum(r["result"].passed for r in runs)
    print(f"\nDeterministic: {n_pass}/{len(runs)} cases passed. Report: {args.report}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
