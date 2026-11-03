"""Shared fixtures. Nothing here needs an LLM or network: the embedding model is
read from the local Hugging Face cache populated by `python -m evals.build_index`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.mcp_server.rag import build_index


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh ChromaDB index of the corpus, built once per test session."""
    path = tmp_path_factory.mktemp("chroma")
    build_index(persist_dir=path)
    return path
