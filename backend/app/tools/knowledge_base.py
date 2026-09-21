"""Local knowledge base with BM25 retrieval.

Zero external services: the KB is a bundled JSON file of worked problems,
indexed in memory with a small BM25 implementation. Swap this module for a
vector store (Qdrant, pgvector) behind the same `search()` signature if the
corpus outgrows keyword retrieval.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from app.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokenize(d) for d in documents]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = (sum(self.doc_len) / len(self.docs)) if self.docs else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str) -> list[float]:
        q_terms = _tokenize(query)
        scores = [0.0] * len(self.docs)
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


class KnowledgeBase:
    def __init__(self, path: Path):
        self.entries: list[dict] = []
        self.index: BM25Index | None = None
        if path.exists():
            self.entries = json.loads(path.read_text(encoding="utf-8"))
            corpus = [
                f"{e['question']} {e.get('topic', '')} {e.get('solution', '')}"
                for e in self.entries
            ]
            self.index = BM25Index(corpus)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top_k KB entries with normalized relevance scores."""
        if not self.index or not self.entries:
            return []
        top_k = top_k or settings.KB_TOP_K
        scores = self.index.score(query)
        best = max(scores) or 1.0
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            entry = dict(self.entries[i])
            entry["score"] = round(scores[i] / best, 4)
            results.append(entry)
        return results


knowledge_base = KnowledgeBase(settings.KB_PATH)


def search_knowledge_base(query: str) -> str:
    """Tool entry point: format KB hits as text for a tool_result block."""
    results = knowledge_base.search(query)
    if not results:
        return "No relevant problems found in the knowledge base."
    parts = []
    for r in results:
        parts.append(
            f"[{r['id']}] (topic: {r.get('topic', '?')}, relevance: {r['score']})\n"
            f"Q: {r['question']}\nWorked solution: {r.get('solution', 'n/a')}\n"
            f"Answer: {r.get('answer', 'n/a')}"
        )
    return "\n\n".join(parts)


KB_TOOL_SCHEMA = {
    "name": "search_knowledge_base",
    "description": "Search the curated knowledge base of worked math problems "
                   "(algebra, calculus, geometry, trigonometry, probability, statistics). "
                   "Check here FIRST for standard textbook-style problems before reasoning "
                   "from scratch — a matching worked solution is the best reference.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The math question or key phrases"}},
        "required": ["query"],
    },
}
