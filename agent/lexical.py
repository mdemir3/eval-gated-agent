"""Deterministic lexical guards used by the Verifier.

Groundedness guard: a draft fails if it contains a *salient term* - a number or
a capitalised word that isn't a common English word - that appears in neither
the evidence nor the question. Emails and URLs are checked as whole strings:
"riley.demo@example.com" is invented even when "riley" and "demo" are both in
the record. This is a cheap, model-independent check for the most damaging
hallucinations (made-up names, dates, quantities, contact details).

Completeness guard: a question with N fact-seeking parts ("Which city ..., and
which river ...?") needs at least N distinct *fact terms* in the answer -
salient phrases ("Ottawa", "Ottawa River") or values copied from API records -
that aren't already in the question. "Why"/"how" parts ask for explanations,
not names or numbers, so they don't count. It's a lower bound that catches
half-answers like "Paris"; it cannot prove every part is answered.

Both are lexical, not semantic: they cannot tell "Paris is the capital of
Japan" is wrong when both words occur in the evidence. That's the LLM
verifier's job.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
import unicodedata

# Capitalised words that carry no factual claim on their own (sentence starters,
# discourse words, units). Kept short on purpose: a false FAIL costs a retry,
# a missing entry here never lets a hallucinated name or number through.
COMMON_WORDS = frozenset(
    """
    a about according after also although an and answer any are as at based be because both but by
    can celsius currently degrees did do does during evidence fahrenheit for from has have he her his
    however i if in is it its no not now of on one or our right she so some source sources than that
    the their then there these they this those to today two unfortunately was we were what when where
    which while who why with yes you
    """.split()
)

# ", and which river ..." / " and in what year ..." starts a new question part.
_PART_SPLIT = re.compile(
    r",?\s+and\s+(?=(?:which|what|who|whom|whose|when|where|why|how|in what|in which|near which|at what|on what)\b)",
    re.IGNORECASE,
)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_EXPLANATION_PART = re.compile(r"(?:why|how)\b(?!\s+(?:many|much|far|long|tall|old|big|deep|high)\b)", re.IGNORECASE)
_IDENTIFIER = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+|https?://[^\s,;)\]]+", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_WORD_AND_SEP = re.compile(r"([A-Za-z0-9]+)([^A-Za-z0-9]*)")


def fold(text: str) -> str:
    """Strip accents, join digit groups (8,849 -> 8849), drop ordinal suffixes (17th -> 17)."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _THOUSANDS.sub("", text)
    return _ORDINAL.sub(r"\1", text)


def tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(fold(text))}


def _is_salient(tok: str) -> bool:
    return any(ch.isdigit() for ch in tok) or (tok[0].isupper() and len(tok) > 1 and tok.lower() not in COMMON_WORDS)


def salient_terms(text: str) -> list[str]:
    """Numbers and capitalised non-common words, in order of first appearance."""
    return list(dict.fromkeys(tok for tok in _TOKEN.findall(fold(text)) if _is_salient(tok)))


def salient_phrases(text: str) -> list[str]:
    """Runs of salient words separated only by spaces: "Alexander Graham Bell", "Ottawa River"."""
    phrases: list[str] = []
    run: list[str] = []
    for word, sep in _WORD_AND_SEP.findall(fold(text)):
        if _is_salient(word):
            run.append(word)
            if sep and not sep.strip():
                continue
        if run:
            phrases.append(" ".join(run))
            run = []
    if run:
        phrases.append(" ".join(run))
    return list(dict.fromkeys(phrases))


def identifiers(text: str) -> list[str]:
    """Email addresses and URLs in `text`, lowercased, without trailing punctuation."""
    return list(dict.fromkeys(m.rstrip(".").lower() for m in _IDENTIFIER.findall(text)))


def unsupported_terms(answer: str, evidence: str, question: str = "") -> list[str]:
    """Salient terms and identifiers of `answer` found in neither `evidence` nor `question`."""
    allowed = tokens(evidence) | tokens(question)
    source_text = f"{evidence} {question}".lower()
    words = [t for t in salient_terms(_IDENTIFIER.sub(" ", answer)) if t.lower() not in allowed]
    return words + [i for i in identifiers(answer) if i not in source_text]


def question_parts(question: str) -> list[str]:
    """Split a compound question into its parts; a simple question is one part."""
    return [part.strip(" ?") for part in _PART_SPLIT.split(question) if part.strip(" ?")]


def fact_seeking_parts(parts: Sequence[str]) -> list[str]:
    """Parts that ask for a name, number or value (not a "why"/"how" explanation)."""
    return [p for p in parts if not _EXPLANATION_PART.match(p)]


def answer_fact_terms(answer: str, question: str, api_values: Sequence[str] = ()) -> list[str]:
    """Distinct fact-stating terms in `answer` that aren't in the question:
    salient phrases with a new word, plus API record values copied into the answer."""
    asked = tokens(question)
    facts = [ph for ph in salient_phrases(answer) if tokens(ph) - asked]
    covered = set().union(*(tokens(f) for f in facts)) if facts else set()
    answer_tokens = tokens(answer)
    for value in api_values:
        value_tokens = tokens(value) - asked
        if value_tokens and value_tokens <= answer_tokens and not value_tokens & covered:
            facts.append(value)
            covered |= value_tokens
    return facts
