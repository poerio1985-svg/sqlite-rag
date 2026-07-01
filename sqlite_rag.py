"""
sqlite-rag — a minimal Retrieval-Augmented Generation store.

Embeddings come from a local Ollama server; vectors and text live in a single
SQLite file; similarity search is plain cosine in Python. No vector database,
no external dependencies — standard library only.

Quickstart:
    ollama pull nomic-embed-text
    python example.py
"""

import json
import math
import os
import re
import sqlite3
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def _embed(text, prefix):
    """Return the embedding vector for `prefix + text` from Ollama.

    nomic-embed-text expects task prefixes: "search_document: " for stored
    chunks and "search_query: " for queries. Skipping them hurts recall.
    """
    body = json.dumps({"model": EMBED_MODEL, "prompt": prefix + text}).encode()
    req = urllib.request.Request(
        OLLAMA_URL + "/api/embeddings", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["embedding"]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class RAG:
    def __init__(self, db_path="rag.db"):
        self.db = sqlite3.connect(db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks "
            "(id INTEGER PRIMARY KEY, source TEXT, text TEXT, embedding TEXT)"
        )
        self.db.commit()

    @staticmethod
    def _chunk(text, size=800, overlap=100):
        text = re.sub(r"\s+", " ", text).strip()
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i : i + size])
            i += size - overlap
        return chunks

    def ingest(self, text, source="doc"):
        """Chunk `text`, embed each chunk, store it. Returns the chunk count."""
        n = 0
        for ch in self._chunk(text):
            emb = _embed(ch, "search_document: ")
            self.db.execute(
                "INSERT INTO chunks (source, text, embedding) VALUES (?,?,?)",
                (source, ch, json.dumps(emb)),
            )
            n += 1
        self.db.commit()
        return n

    def ingest_file(self, path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return self.ingest(f.read(), source=os.path.basename(path))

    def search(self, query, k=4):
        """Return the top-k chunks most similar to `query`."""
        q = _embed(query, "search_query: ")
        rows = self.db.execute("SELECT source, text, embedding FROM chunks").fetchall()
        scored = [(_cosine(q, json.loads(e)), s, t) for s, t, e in rows]
        scored.sort(reverse=True)
        return [
            {"score": round(sc, 3), "source": s, "text": t}
            for sc, s, t in scored[:k]
        ]

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
