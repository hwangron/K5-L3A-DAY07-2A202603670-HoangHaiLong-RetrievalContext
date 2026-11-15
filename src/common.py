"""Shared paths, config, data loading and evidence matching for Day 7."""
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
CHUNKS_FILE = ROOT / "data" / "day6" / "chunks.json"
QUESTIONS_FILE = ROOT / "data" / "questions.json"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"


def load_env() -> None:
    """Read .env in the repo root (gitignored). Values already in the environment win."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    for key in ("DATABASE_URL", "OPENAI_API_KEY"):
        if not (os.getenv(key) or "").strip():
            raise RuntimeError(f"{key} is missing; copy .env.example to .env and fill it")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_chunks() -> list[dict[str, Any]]:
    return read_json(CHUNKS_FILE)


def load_questions(split: str) -> list[dict[str, Any]]:
    """Only the requested split is returned, so dev work never touches test rows."""
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    return [q for q in read_json(QUESTIONS_FILE) if q["split"] == split]


def squash(text: str) -> str:
    """Same normalisation as Day 6: collapse whitespace, lower case."""
    return re.sub(r"\s+", " ", text).strip().lower()


def is_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """A chunk is a hit if it is from a gold (source_id, page) and contains the whole evidence."""
    text = squash(chunk["text"])
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and squash(g["evidence"]) in text for g in gold)


def block(chunk: dict[str, Any]) -> str:
    m = chunk.get("metadata", {})
    return (f"[{chunk['id']} | {chunk['source']} | page {chunk['page']} | section {m.get('section')}]\n"
            f"{chunk['text']}")


def pack_context(chunks: list[dict[str, Any]], budget: int) -> tuple[str, list[str]]:
    """Greedy packing in the given order; skips duplicates and chunks that do not fit.
    Budget unit: characters of the final context string, headers included (as in starter)."""
    blocks, ids, seen = [], [], set()
    for c in chunks:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        b = block(c)
        if len("\n\n".join(blocks + [b])) <= budget:
            blocks.append(b)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids
