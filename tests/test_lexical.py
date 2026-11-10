"""The Verifier's deterministic guards."""

from __future__ import annotations

import pytest

from agent.lexical import (
    answer_fact_terms,
    fact_seeking_parts,
    question_parts,
    salient_phrases,
    salient_terms,
    unsupported_terms,
)

EVIDENCE = (
    "Mount Everest is the highest mountain on Earth. Its summit is about 8,849 metres above sea level. "
    "Edmund Hillary and Tenzing Norgay made the first confirmed ascent to the summit in 1953."
)


@pytest.mark.parametrize(
    "answer",
    [
        "Edmund Hillary and Tenzing Norgay.",
        "They reached the summit in 1953.",
        "The summit is 8849 metres high.",  # digit grouping normalised
        "Everest is about 8,849 metres.",
    ],
)
def test_grounded_answers_pass(answer):
    assert unsupported_terms(answer, EVIDENCE) == []


@pytest.mark.parametrize(
    ("answer", "invented"),
    [
        ("George Mallory reached the summit.", ["George", "Mallory"]),
        ("The first ascent was in 1924.", ["1924"]),
        ("Everest is 8,850 metres high.", ["8850"]),
    ],
)
def test_invented_names_and_numbers_are_caught(answer, invented):
    assert unsupported_terms(answer, EVIDENCE) == invented


def test_terms_from_the_question_are_allowed():
    assert unsupported_terms("Kilimanjaro is not mentioned.", EVIDENCE, "Is Kilimanjaro taller?") == []


USER_RECORD = '{"id": 3, "name": "Riley Demo", "city": "Ottawa", "member_since": 2020}'
USER_QUESTION = "Which city does user 3 live in, and what is their email address?"


@pytest.mark.parametrize(
    ("answer", "invented"),
    [
        # lowercase, and every word of it ("riley", "demo") is in the record: only a
        # whole-string check catches it
        ("Ottawa; riley.demo@example.com.", ["riley.demo@example.com"]),
        ("User 3 lives in Ottawa; their email address is Riley.Demo@Example.com", ["riley.demo@example.com"]),
        ("Ottawa; see https://example.com/users/3.", ["https://example.com/users/3"]),
    ],
)
def test_invented_emails_and_urls_are_caught(answer, invented):
    assert unsupported_terms(answer, USER_RECORD, USER_QUESTION) == invented


def test_emails_and_urls_present_in_the_evidence_pass():
    evidence = '{"id": 7, "email": "help@demo.example", "site": "https://demo.example/help"}'
    answer = "You can write to help@demo.example or visit https://demo.example/help."
    assert unsupported_terms(answer, evidence) == []


def test_accents_are_folded():
    assert unsupported_terms("Brasília.", "Brasilia became the capital in 1960.") == []


def test_salient_terms_skip_sentence_starters():
    assert salient_terms("The answer is Ottawa. However, it is cold.") == ["Ottawa"]


def test_salient_phrases_group_adjacent_names():
    assert salient_phrases("Alexander Graham Bell, in 1876; the Ottawa River.") == [
        "Alexander Graham Bell",
        "1876",
        "Ottawa River",
    ]


@pytest.mark.parametrize(
    ("question", "parts"),
    [
        ("What is the capital of Canada?", 1),
        ("In which city do the White Nile and the Blue Nile meet?", 1),
        ("Which city does user 3 live in, and which river is that city on?", 2),
        ("Who was granted the patent, and in what year?", 2),
    ],
)
def test_question_parts(question, parts):
    assert len(question_parts(question)) == parts


def test_explanation_parts_do_not_need_a_fact_term():
    parts = question_parts("Why is Mars red, and how many moons does it have?")
    assert fact_seeking_parts(parts) == ["how many moons does it have"]


def test_answer_fact_terms_count_distinct_new_facts():
    question = "Which city is order 1001 in, and which river flows through that city?"
    assert answer_fact_terms("Paris.", question) == ["Paris"]
    assert answer_fact_terms("Order 1001 is in Paris; the river Seine.", question) == ["Paris", "Seine"]
    # a lowercase API value copied into the answer counts as a fact
    assert answer_fact_terms("It is sunny; Japan.", "What is the weather, and which country?", ["sunny"]) == [
        "Japan",
        "sunny",
    ]
