"""Unit tests for the MCP tools.

Tools are exercised through MCP `call_tool` (an in-process server, plus one
real stdio subprocess test) so schema validation and error signalling are
covered, not just the Python functions behind them.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.mcp_client import MCPToolClient
from tools.mcp_server.mock_data import MockAPI, MockAPIError
from tools.mcp_server.rag import Retriever, chunk_document, load_corpus
from tools.mcp_server.server import create_server


def call(server, name: str, args: dict):
    async def go():
        async with MCPToolClient(server) as client:
            return await client.call(name, args)

    return asyncio.run(go())


def call_many(server, calls: list[tuple[str, dict]]):
    """Several calls in one session, so per-session state (the flaky endpoint) is shared."""

    async def go():
        async with MCPToolClient(server) as client:
            return [await client.call(name, args) for name, args in calls]

    return asyncio.run(go())


# --- corpus / chunking ----------------------------------------------------


def test_corpus_has_about_twenty_short_docs():
    docs = load_corpus()
    assert 18 <= len(docs) <= 25
    for doc_id, _title, body in docs:
        sentences = [s for s in body.replace("?", ".").split(". ") if s.strip()]
        assert 2 <= len(sentences) <= 5, doc_id


def test_short_doc_is_one_chunk_and_long_doc_splits_on_sentences():
    assert len(chunk_document("d", "T", "One. Two. Three.")) == 1
    long_body = " ".join(f"Sentence number {i} has five words." for i in range(60))
    chunks = chunk_document("d", "T", long_body, max_words=50)
    assert len(chunks) > 1
    assert all(c.text.endswith(".") for c in chunks)
    assert [c.chunk_id for c in chunks][:2] == ["d#0", "d#1"]


# --- rag_search -----------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("What is the capital of Canada?", "capital-canada"),
        ("Why is Mars called the Red Planet?", "planet-mars"),
        ("Who got the first patent for the telephone?", "invention-telephone"),
        ("Where did the Wright brothers fly?", "invention-airplane"),
        ("Where do the White Nile and Blue Nile meet?", "geo-nile"),
        ("Who first climbed Mount Everest?", "geo-everest"),
    ],
)
def test_rag_search_returns_relevant_doc_first(index_dir, query, expected_doc):
    result = call(create_server(retriever=Retriever(index_dir)), "rag_search", {"query": query, "top_k": 3})
    assert result.ok, result.error
    hits = result.data["results"]
    assert hits[0]["doc_id"] == expected_doc
    assert {"doc_id", "chunk_id", "text", "score"} <= hits[0].keys()


def test_rag_search_respects_top_k_and_orders_by_score(index_dir):
    result = call(create_server(retriever=Retriever(index_dir)), "rag_search", {"query": "planets", "top_k": 5})
    scores = [h["score"] for h in result.data["results"]]
    assert len(scores) == 5
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("args", [{"query": "x", "top_k": 0}, {"query": "x", "top_k": 11}, {"query": ""}, {}])
def test_rag_search_rejects_invalid_arguments(index_dir, args):
    result = call(create_server(retriever=Retriever(index_dir)), "rag_search", args)
    assert not result.ok
    assert "validation error" in result.error


def test_rag_search_reports_missing_index_as_tool_error(tmp_path):
    result = call(create_server(retriever=Retriever(tmp_path / "nope")), "rag_search", {"query": "x"})
    assert not result.ok
    assert "build_index" in result.error


# --- mock_api_get: normal endpoints ----------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"endpoint": "/users/3"}, {"id": 3, "name": "Riley Demo", "city": "Ottawa", "member_since": 2020}),
        ({"endpoint": "/users/{id}", "params": {"id": 1}}, {"id": 1, "name": "Alex Demo", "city": "Paris", "member_since": 2019}),
        ({"endpoint": "/orders/1002"}, {"id": 1002, "user_id": 1, "item": "telephone", "status": "processing"}),
        ({"endpoint": "/weather/Tokyo"}, {"city": "Tokyo", "condition": "sunny", "temperature_c": 22}),
        ({"endpoint": "weather/cairo"}, {"city": "Cairo", "condition": "clear", "temperature_c": 31}),
    ],
)
def test_mock_api_returns_expected_records(args, expected):
    result = call(create_server(), "mock_api_get", args)
    assert result.ok, result.error
    assert result.data["status"] == 200
    assert result.data["data"] == expected


@pytest.mark.parametrize(
    ("endpoint", "status"),
    [("/users/99", "404"), ("/weather/Atlantis", "404"), ("/users/abc", "400"), ("/payments/1", "404")],
)
def test_mock_api_errors_are_tool_errors(endpoint, status):
    result = call(create_server(), "mock_api_get", {"endpoint": endpoint})
    assert not result.ok
    assert status in result.error


def test_mock_api_is_deterministic():
    api = MockAPI()
    assert api.get("/orders/1001") == api.get("/orders/1001") == MockAPI().get("/orders/1001")


# --- mock_api_get: flaky and always-error endpoints ---------------------------


def test_flaky_endpoint_fails_first_then_succeeds_on_retry():
    first, retry = call_many(create_server(), [("mock_api_get", {"endpoint": "/tracking/1001"})] * 2)
    assert not first.ok and "503" in first.error
    assert retry.ok
    assert retry.data["data"]["current_city"] == "Paris"


def test_flaky_state_is_per_resource_and_per_session():
    a1, b1, a2 = call_many(
        create_server(),
        [
            ("mock_api_get", {"endpoint": "/tracking/1001"}),
            ("mock_api_get", {"endpoint": "/tracking/1002"}),
            ("mock_api_get", {"endpoint": "/tracking/1001"}),
        ],
    )
    assert (a1.ok, b1.ok, a2.ok) == (False, False, True)
    # a fresh server starts cold again
    assert not call(create_server(), "mock_api_get", {"endpoint": "/tracking/1001"}).ok


def test_always_error_endpoint_is_handled_without_crashing():
    results = call_many(
        create_server(),
        [("mock_api_get", {"endpoint": "/inventory/SKU-42"})] * 3 + [("mock_api_get", {"endpoint": "/users/2"})],
    )
    assert all(not r.ok and "500" in r.error for r in results[:3])
    assert results[3].ok, "server must keep serving after repeated tool errors"


def test_mock_api_error_type():
    with pytest.raises(MockAPIError) as exc:
        MockAPI().get("/inventory/X")
    assert exc.value.status == 500


# --- real stdio transport ------------------------------------------------------


def test_tools_are_served_over_mcp_stdio(index_dir, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(index_dir))

    async def go():
        async with MCPToolClient.stdio() as client:
            names = sorted(t.name for t in await client.list_tools())
            rag = await client.call("rag_search", {"query": "largest planet", "top_k": 1})
            api = await client.call("mock_api_get", {"endpoint": "/orders/1003"})
            return names, rag, api

    names, rag, api = asyncio.run(go())
    assert names == ["mock_api_get", "rag_search"]
    assert rag.ok and rag.data["results"][0]["doc_id"] == "planet-jupiter"
    assert api.ok and api.data["data"]["status"] == "delivered"
