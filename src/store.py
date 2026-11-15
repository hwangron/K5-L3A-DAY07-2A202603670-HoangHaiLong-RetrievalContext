"""Read-only access to the Day 6 pgvector store and the matching query embedder.

Errors (database down, API failure) are raised; there is no mock fallback."""
import hashlib
import json
import os
import time
from typing import Any
from urllib.parse import urlparse

import pg8000.native
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from .common import CACHE, CONFIG

DAY6 = CONFIG["day6"]
FILTER_FIELDS = {"chapter", "section", "page_type"}
COLS = ["chunk_id", "document_id", "source", "page", "printed_page", "chapter", "section",
        "section_title", "page_type", "char_start", "char_end", "text"]


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def to_day7(row: dict[str, Any]) -> dict[str, Any]:
    """Day 6 row -> Day 7 chunk contract (id, text, source_id, source, page, metadata)."""
    meta = {k: row[k] for k in ("printed_page", "chapter", "section", "section_title", "page_type",
                                "char_start", "char_end")}
    meta["day6_strategy"] = DAY6["strategy"]
    return {"id": row["chunk_id"], "text": row["text"], "source_id": row["document_id"],
            "source": row["source"], "page": row["page"], "metadata": meta}


class Day6Store:
    def __init__(self) -> None:
        u = urlparse(os.environ["DATABASE_URL"])
        self.conn = pg8000.native.Connection(user=u.username, password=u.password, host=u.hostname,
                                             port=u.port or 5432, database=u.path.lstrip("/"))
        self.table, self.strategy = DAY6["table"], DAY6["strategy"]
        row = self.conn.run(f"SELECT count(*), min(embedding_model), max(embedding_model), "
                            f"min(vector_dims(embedding)) FROM {self.table} WHERE strategy = :s", s=self.strategy)[0]
        if row[0] == 0:
            raise RuntimeError(f"No rows for strategy {self.strategy}; run Day 6 ingest first")
        if row[1] != DAY6["embedding_model"] or row[2] != DAY6["embedding_model"] or row[3] != DAY6["dimensions"]:
            raise RuntimeError(f"Index model/dimension mismatch: {row}")
        self.n_chunks = row[0]

    def versions(self) -> dict[str, Any]:
        r = self.conn.run("SELECT current_setting('server_version'), "
                          "(SELECT extversion FROM pg_extension WHERE extname='vector')")[0]
        return {"postgres": r[0], "pgvector": r[1], "strategy": self.strategy, "chunks": self.n_chunks,
                "embedding_model": DAY6["embedding_model"], "dimensions": DAY6["dimensions"]}

    def all_chunks(self) -> list[dict[str, Any]]:
        rows = self.conn.run(f"SELECT {', '.join(COLS)} FROM {self.table} WHERE strategy = :s "
                             "ORDER BY page, char_start, chunk_id", s=self.strategy)
        return [to_day7(dict(zip(COLS, r))) for r in rows]

    def search(self, qvec: list[float], top_k: int, filters: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Exact cosine search; filter fields are whitelisted and applied in SQL before ranking."""
        where, params = ["strategy = :s"], {"s": self.strategy, "q": _vec(qvec), "k": top_k}
        for i, (field, value) in enumerate((filters or {}).items()):
            if field not in FILTER_FIELDS:
                raise ValueError(f"filter field not allowed: {field}")
            where.append(f"{field} = :f{i}")
            params[f"f{i}"] = value
        rows = self.conn.run(
            f"SELECT {', '.join(COLS)}, 1 - (embedding <=> CAST(:q AS vector)) FROM {self.table} "
            f"WHERE {' AND '.join(where)} ORDER BY embedding <=> CAST(:q AS vector), chunk_id LIMIT :k", **params)
        return [{**to_day7(dict(zip(COLS, r[:-1]))), "score": float(r[-1])} for r in rows]


class QueryEmbedder:
    """Same model as the Day 6 index. Vectors are cached on disk (.cache/, gitignored)
    so re-running an experiment does not pay twice; cache hits are counted separately."""

    def __init__(self) -> None:
        self.model = DAY6["embedding_model"]
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip(), timeout=60)
        self.path = CACHE / f"query_emb_{self.model}.json"
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.tokens = self.calls = self.cache_hits = 0

    def embed(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        for attempt in range(4):
            try:
                resp = self.client.embeddings.create(model=self.model, input=[text])
                break
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        vec = resp.data[0].embedding
        if len(vec) != DAY6["dimensions"]:
            raise RuntimeError("query embedding has wrong dimension")
        self.calls += 1
        self.tokens += resp.usage.total_tokens
        self.cache[key] = vec
        CACHE.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.cache), encoding="utf-8")
        return vec

    def cost_usd(self) -> float:
        return self.tokens / 1e6 * DAY6["embedding_price_per_mtok_usd"]
