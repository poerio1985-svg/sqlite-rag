"""Tiny end-to-end demo for sqlite-rag.

    ollama pull nomic-embed-text
    ollama pull qwen2.5:7b        # optional, only for the generated answer
    python example.py
"""

import json
import os
import urllib.request

from sqlite_rag import RAG, OLLAMA_URL

CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:7b")

SAMPLE = """
The James Webb Space Telescope launched on 25 December 2021 from French Guiana.
It observes primarily in the infrared and orbits the Sun at the second Lagrange
point (L2), about 1.5 million kilometres from Earth. Its primary mirror is 6.5
metres across and is made of 18 gold-coated beryllium segments.
"""


def answer(question, context):
    prompt = (
        "Answer the question using ONLY the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    body = json.dumps(
        {"model": CHAT_MODEL, "prompt": prompt, "stream": False}
    ).encode()
    req = urllib.request.Request(
        OLLAMA_URL + "/api/generate", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["response"].strip()


if __name__ == "__main__":
    rag = RAG("demo.db")
    if rag.count() == 0:
        print("Ingesting sample text...")
        print("  chunks stored:", rag.ingest(SAMPLE, source="jwst"))

    q = "How far from Earth does the telescope orbit, and at which point?"
    print("\nQuery:", q)
    hits = rag.search(q, k=2)
    for h in hits:
        print(f"  [{h['score']}] {h['source']}: {h['text'][:80]}...")

    context = "\n".join(h["text"] for h in hits)
    try:
        print("\nGenerated answer:\n ", answer(q, context))
    except Exception as e:
        print("\n(Skipping generation — is a chat model pulled?)", e)
