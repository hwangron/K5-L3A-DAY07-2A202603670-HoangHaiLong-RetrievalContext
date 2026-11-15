"""Retrieval-only benchmark with the ORIGINAL question as query (no rewriting).

    python -m src.eval_retrieval --split dev [--methods dense bm25 hybrid_rrf hybrid_rerank] [--budgets 800 3000]

Hit@1 / Evidence Hit@3: evidence chunk (gold source + page + full quote) at rank 1 / in top-3.
Context Evidence Rate: evidence chunk inside the context packed from top-3 under the char budget.
Unanswerable questions are not in the denominators. Output: results/retrieval/<split>/."""
import argparse
import statistics
import time

from .common import CONFIG, RESULTS, is_hit, load_questions, pack_context, write_json
from .retrievers import build

TOP_K = CONFIG["retrieval_eval"]["top_k"]


def slim(r: dict) -> dict:
    return {"rank": r["rank"], "id": r["id"], "page": r["page"], "section": r["metadata"]["section"],
            "score": round(r["score"], 5), "hit": r.get("hit"), **({"sources": r["sources"]} if "sources" in r else {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid_rrf", "hybrid_rerank"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[800, 3000])
    args = ap.parse_args()
    qs = [q for q in load_questions(args.split) if q["answerable"]]
    retrievers, emb = build(args.methods)
    out_dir = RESULTS / "retrieval" / args.split
    summary = {"split": args.split, "n": len(qs), "top_k": TOP_K, "query": "original question",
               "filter": "question.filter (user-provided scope) for all methods", "methods": {}}
    for name, retr in retrievers.items():
        tokens0 = emb.tokens
        retr.search("warm up", top_k=1)  # model load / connection outside timing
        traces, times = [], []
        for q in qs:
            t = time.perf_counter()
            res = retr.search(q["question"], filters=q.get("filter"), top_k=TOP_K)
            times.append((time.perf_counter() - t) * 1000)
            for r in res:
                r["hit"] = is_hit(r, q["gold"])
            ctx = {}
            for b in args.budgets:
                _, ids = pack_context(res, b)
                ctx[str(b)] = {"ids": ids, "evidence": any(r["hit"] and r["id"] in ids for r in res)}
            traces.append({"query_id": q["id"], "question": q["question"], "filter": q.get("filter"),
                           "results": [slim(r) for r in res], "hit1": bool(res and res[0]["hit"]),
                           "hit3": any(r["hit"] for r in res), "context": ctx,
                           "elapsed_ms": round(times[-1], 1)})
        n = len(qs)
        s = {"hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces), "N": n,
             "context_evidence": {b: sum(t["context"][b]["evidence"] for t in traces) for b in map(str, args.budgets)},
             "median_ms": round(statistics.median(times), 1), "mean_ms": round(statistics.mean(times), 1),
             "embedding_tokens": emb.tokens - tokens0,
             "missed_at3": [t["query_id"] for t in traces if not t["hit3"]],
             "missed_at1": [t["query_id"] for t in traces if not t["hit1"]]}
        s["embedding_cost_usd"] = round(s["embedding_tokens"] / 1e6 * CONFIG["day6"]["embedding_price_per_mtok_usd"], 6)
        summary["methods"][name] = s
        write_json(out_dir / f"{name}.json", traces)
        ce = ", ".join(f"ctx@{b}={v}/{n}" for b, v in s["context_evidence"].items())
        print(f"{name:14s} Hit@1={s['hit1']}/{n} Hit@3={s['hit3']}/{n} {ce} median={s['median_ms']}ms "
              f"miss@3={s['missed_at3']}")
    summary["embedding_cache_hits"] = emb.cache_hits
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
