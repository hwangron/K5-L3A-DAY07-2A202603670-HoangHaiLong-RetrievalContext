"""Four retrieval methods over the same 144 Day 6 strategy-C chunks.

dense          pgvector cosine with text-embedding-3-small (Day 6 index)
bm25           Okapi BM25 over the same chunk texts (own implementation, English tokens)
hybrid_rrf     dense top-N + BM25 top-N fused by Reciprocal Rank Fusion (ranks, not raw scores)
hybrid_rerank  hybrid_rrf top-N candidates re-scored by a local cross-encoder

Every result has id, text, source_id, source, page, metadata, score, rank, method.
Filters only come from the user-provided `filter` field of a question; never from gold."""
import math
import re
from collections import Counter
from typing import Any

from .common import CONFIG

STOPWORDS = set("""a an and are as at be by can do does for from how if in is it its of on or that the
then there this to was what when where which why with we our you your""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def match_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    return all(chunk["metadata"].get(k) == v for k, v in (filters or {}).items())


def ranked(rows: list[dict[str, Any]], method: str, top_k: int) -> list[dict[str, Any]]:
    return [{**r, "rank": i + 1, "method": method} for i, r in enumerate(rows[:top_k])]


class BM25:
    def __init__(self, chunks: list[dict[str, Any]], k1: float | None = None, b: float | None = None) -> None:
        cfg = CONFIG["bm25"]
        self.k1, self.b = k1 or cfg["k1"], b or cfg["b"]
        self.chunks = chunks
        self.tf = [Counter(tokenize(c["text"])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q = tokenize(query)
        rows = []
        for c, tf, ln in zip(self.chunks, self.tf, self.len):
            if not match_filter(c, filters):
                continue
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * ln / self.avg))
                    for t in q if t in tf)
            if s > 0:
                rows.append({**c, "score": s})
        rows.sort(key=lambda r: (-r["score"], r["id"]))
        return ranked(rows, "bm25", top_k)


class Dense:
    def __init__(self, store, embedder) -> None:
        self.store, self.embedder = store, embedder

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        return ranked(self.store.search(self.embedder.embed(query), top_k, filters), "dense", top_k)


class HybridRRF:
    def __init__(self, dense: Dense, bm25: BM25, candidates: int | None = None, k: int | None = None) -> None:
        self.dense, self.bm25 = dense, bm25
        self.n, self.k = candidates or CONFIG["candidates"], k or CONFIG["rrf_k"]

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for name, retr in (("dense", self.dense), ("bm25", self.bm25)):
            for r in retr.search(query, filters=filters, top_k=self.n):
                row = fused.setdefault(r["id"], {**{k: v for k, v in r.items() if k not in ("score", "rank", "method")},
                                                 "score": 0.0, "sources": {}})
                row["score"] += 1 / (self.k + r["rank"])
                row["sources"][name] = r["rank"]
        rows = sorted(fused.values(), key=lambda r: (-r["score"], r["id"]))
        return ranked(rows, "hybrid_rrf", top_k)


class Reranker:
    """Cross-encoder rerank of hybrid candidates. Runs locally on CPU, no API cost."""

    def __init__(self, base: HybridRRF, model_name: str | None = None, candidates: int | None = None) -> None:
        from sentence_transformers import CrossEncoder
        self.base, self.n = base, candidates or CONFIG["candidates"]
        self.model = CrossEncoder(model_name or CONFIG["reranker"])

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        cands = self.base.search(query, filters=filters, top_k=self.n)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands])
        rows = [{**c, "score": float(s), "rrf_rank": c["rank"]} for c, s in zip(cands, scores)]
        rows.sort(key=lambda r: (-r["score"], r["id"]))
        return ranked(rows, "hybrid_rerank", top_k)


def build(names: list[str]) -> tuple[dict[str, Any], Any]:
    """Create the requested retrievers. Returns (retrievers, embedder) so callers can read cost."""
    from .common import load_chunks, load_env
    from .store import Day6Store, QueryEmbedder
    load_env()
    store, emb = Day6Store(), QueryEmbedder()
    dense, bm25 = Dense(store, emb), BM25(load_chunks())
    out: dict[str, Any] = {"dense": dense, "bm25": bm25}
    if {"hybrid_rrf", "hybrid_rerank"} & set(names):
        out["hybrid_rrf"] = HybridRRF(dense, bm25)
    if "hybrid_rerank" in names:
        out["hybrid_rerank"] = Reranker(out["hybrid_rrf"])
    return {n: out[n] for n in names}, emb
