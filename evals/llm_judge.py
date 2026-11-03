"""Optional LLM-judge suite: DeepEval faithfulness + answer relevancy.

Runs only when the judge provider is a hosted API (openai/anthropic) and its
key is set. Otherwise it reports `LLM-judge: not run (no key)` and scores
nothing - it never fabricates a score and never fails CI for a missing key.

Judge config: JUDGE_PROVIDER / JUDGE_MODEL (default to LLM_PROVIDER / LLM_MODEL),
threshold JUDGE_THRESHOLD or --judge-threshold (default 0.7).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from agent.llm import LangChainLLM, LLMConfigError, get_chat_model, has_api_key, resolve_config

NOT_RUN_NO_KEY = "LLM-judge: not run (no key)"
HOSTED_JUDGES = ("openai", "anthropic")
DEFAULT_THRESHOLD = 0.7


@dataclass
class JudgeCaseResult:
    case_id: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    passed: bool = False
    reasons: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class JudgeReport:
    ran: bool
    status: str  # human-readable: "ran with ..." or why it didn't run
    threshold: float
    judge_model: str = ""
    results: list[JudgeCaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.ran or all(r.passed for r in self.results)


def judge_availability() -> tuple[bool, str]:
    provider = (os.getenv("JUDGE_PROVIDER") or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider not in HOSTED_JUDGES or not has_api_key(provider):
        return False, NOT_RUN_NO_KEY
    try:
        resolve_config("JUDGE")
    except LLMConfigError as exc:
        return False, f"LLM-judge: not run ({exc})"
    return True, ""


def _judge_model():
    """A DeepEval judge backed by our env-configured LangChain chat model."""
    from deepeval.models import DeepEvalBaseLLM

    provider, model = resolve_config("JUDGE")
    chat = get_chat_model("JUDGE")
    method = LangChainLLM._METHOD[provider]

    class LangChainJudge(DeepEvalBaseLLM):
        def __init__(self) -> None:
            super().__init__(model=model)

        def load_model(self, *args: Any, **kwargs: Any):
            return chat

        def generate(self, prompt: str, schema: Any = None, **kwargs: Any):
            if schema is not None:
                return chat.with_structured_output(schema, method=method).invoke(prompt)
            return chat.invoke(prompt).content

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any):
            if schema is not None:
                return await chat.with_structured_output(schema, method=method).ainvoke(prompt)
            return (await chat.ainvoke(prompt)).content

        def get_model_name(self, *args: Any, **kwargs: Any) -> str:
            return f"{provider}:{model}"

    return LangChainJudge()


def run_judge(items: list[tuple[str, dict[str, Any]]], threshold: float) -> JudgeReport:
    """`items` = (case_id, final agent state) for answered cases."""
    available, reason = judge_availability()
    if not available:
        return JudgeReport(ran=False, status=reason, threshold=threshold)

    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    judge = _judge_model()
    report = JudgeReport(ran=True, status=f"ran with {judge.get_model_name()}", threshold=threshold,
                         judge_model=judge.get_model_name())
    for case_id, state in items:
        result = JudgeCaseResult(case_id)
        test_case = LLMTestCase(
            input=state["question"],
            actual_output=state["final"]["answer"],
            retrieval_context=[e["text"] for e in state.get("retrieved_chunks", [])],
        )
        try:
            faith = FaithfulnessMetric(threshold=threshold, model=judge, async_mode=False, verbose_mode=False)
            relevancy = AnswerRelevancyMetric(threshold=threshold, model=judge, async_mode=False, verbose_mode=False)
            faith.measure(test_case)
            relevancy.measure(test_case)
            result.faithfulness, result.answer_relevancy = faith.score, relevancy.score
            result.reasons = {"faithfulness": faith.reason or "", "answer_relevancy": relevancy.reason or ""}
            result.passed = faith.score >= threshold and relevancy.score >= threshold
        except Exception as exc:  # report the failure; never invent a score
            result.error = f"{type(exc).__name__}: {exc}"
        report.results.append(result)
    return report
