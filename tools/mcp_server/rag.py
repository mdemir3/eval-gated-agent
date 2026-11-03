"""Corpus loading, chunking, indexing and search for the `rag_search` tool.

Embeddings come from sentence-transformers `all-MiniLM-L6-v2`, run locally on
CPU (no API key). Vectors live in a local persistent ChromaDB collection.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus"
COLLECTION_NAME = "corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 truncates input at 256 word pieces. Corpus docs are 2-5
# sentences (~40-60 words), so a whole doc fits in one chunk. Longer docs are
# split on sentence boundaries so nothing is silently truncated.
MAX_CHUNK_WORDS = 120

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def chroma_dir() -> Path:
    """Index location: $CHROMA_DIR (relative paths resolve against the repo root)."""
    path = Path(os.getenv("CHROMA_DIR") or ".chroma")
    return path if path.is_absolute() else REPO_ROOT / path


def load_corpus(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[tuple[str, str, str]]:
    """Return (doc_id, title, body) for every Markdown file in the corpus."""
    docs = []
    for path in sorted(corpus_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else path.stem
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        docs.append((path.stem, title, body))
    if not docs:
        raise FileNotFoundError(f"No .md documents found in {corpus_dir}")
    return docs


def chunk_document(doc_id: str, title: str, body: str, max_words: int = MAX_CHUNK_WORDS) -> list[Chunk]:
    """One chunk per doc unless it exceeds `max_words`; then greedy sentence packing."""
    sentences = [s for s in _SENTENCE_END.split(body) if s]
    chunks: list[list[str]] = [[]]
    for sentence in sentences:
        current = chunks[-1]
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            chunks.append([sentence])
        else:
            current.append(sentence)
    return [
        Chunk(chunk_id=f"{doc_id}#{i}", doc_id=doc_id, title=title, text=" ".join(parts))
        for i, parts in enumerate(chunks)
        if parts
    ]


def _client(persist_dir: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))


def _embedding_function():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL, device="cpu")


def build_index(corpus_dir: Path = DEFAULT_CORPUS_DIR, persist_dir: Path | None = None) -> int:
    """(Re)build the Chroma collection from the corpus. Returns the number of chunks."""
    persist_dir = persist_dir or chroma_dir()
    client = _client(persist_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # collection didn't exist yet
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    chunks = [c for doc in load_corpus(corpus_dir) for c in chunk_document(*doc)]
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"doc_id": c.doc_id, "title": c.title} for c in chunks],
    )
    return len(chunks)


class Retriever:
    """Thin wrapper over the persisted collection; the embedding model loads on first use."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self.persist_dir = persist_dir or chroma_dir()
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            if not self.persist_dir.exists():
                raise FileNotFoundError(
                    f"No ChromaDB index at {self.persist_dir}. Run `python -m evals.build_index` first."
                )
            self._collection = _client(self.persist_dir).get_collection(
                COLLECTION_NAME, embedding_function=_embedding_function()
            )
        return self._collection

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        res = self._get_collection().query(query_texts=[query], n_results=top_k)
        hits = []
        for chunk_id, text, meta, distance in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                {
                    "doc_id": meta["doc_id"],
                    "chunk_id": chunk_id,
                    "title": meta["title"],
                    "text": text,
                    "score": round(1.0 - float(distance), 4),  # cosine similarity
                }
            )
        return hits
