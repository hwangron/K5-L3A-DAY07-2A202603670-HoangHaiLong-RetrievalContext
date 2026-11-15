"""Export Day 6 strategy-C chunks from pgvector and build the Day 7 benchmark.

    python -m src.export_day6 --day6 ../Lab6/K5-L3A-Day06-DataIngestionEngineering

Writes data/day6/chunks.json, data/questions.json, data/cases.json, data/day6/export_info.json.
Gold evidence is only converted (field rename) and checked; nothing is guessed."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from .common import CHUNKS_FILE, QUESTIONS_FILE, ROOT, is_hit, load_env, read_json, write_json
from .store import Day6Store


def convert_question(q: dict) -> dict:
    gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
    for alt in q.get("alternatives", []):
        gold.append({"source_id": alt.get("source_id", q["gold_source_id"]), "page": alt["page"],
                     "evidence": alt["quote"]})
    return {"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
            "expected_answer": q["expected_answer"], "gold": gold, "filter": q.get("filter"),
            "answerable": True, "expected_behavior": "search_and_answer"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day6", required=True, type=Path, help="Day 6 repo folder")
    args = ap.parse_args()
    load_env()
    store = Day6Store()
    chunks = store.all_chunks()
    write_json(CHUNKS_FILE, chunks)

    day6_q = read_json(args.day6 / "benchmark" / "questions.json")["questions"]
    questions = [convert_question(q) for q in day6_q] + read_json(ROOT / "data" / "unanswerable.json")
    questions.sort(key=lambda q: (q["split"], q["id"]))
    problems = []
    for q in questions:
        if q["answerable"]:
            for g in q["gold"]:
                if not any(is_hit(c, [g]) for c in chunks):
                    problems.append(f"{q['id']}: evidence on page {g['page']} not inside one chunk")
    write_json(QUESTIONS_FILE, questions)
    write_json(ROOT / "data" / "cases.json",
               [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in questions])
    counts = {s: {"answerable": sum(q["split"] == s and q["answerable"] for q in questions),
                  "unanswerable": sum(q["split"] == s and not q["answerable"] for q in questions)}
              for s in ("dev", "test")}
    write_json(CHUNKS_FILE.parent / "export_info.json", {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "store": store.versions(), "chunks": len(chunks), "questions": counts,
        "gold_problems": problems,
        "license": "OpenStax Calculus Volume 1, CC BY-NC-SA 4.0. Download for free at "
                   "https://openstax.org/details/books/calculus-volume-1."})
    print(len(chunks), "chunks;", counts, "problems:", problems or "none")


if __name__ == "__main__":
    main()
