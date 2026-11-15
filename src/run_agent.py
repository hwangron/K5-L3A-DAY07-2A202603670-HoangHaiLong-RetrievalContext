"""Run the real agent for one config on one split and save traces.

    python -m src.run_agent --config A1_dense_budget --split dev

Output results/agent/<config>_<split>/traces.json and auto_summary.json.
Gold is used only AFTER the run, to compute Context Evidence Rate; labels for answer quality,
citation support and abstention are written by hand in reviews.json next to the traces."""
import argparse
import statistics
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

from .agent import Agent
from .common import CONFIG, RESULTS, is_hit, load_chunks, load_questions, write_json
from .retrievers import build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIG["agent_configs"]))
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--only", nargs="*", help="optional query IDs (debug)")
    ap.add_argument("--agent-version", type=int, default=3, choices=(1, 2, 3),
                    help="1 = first dev version, 2 = cited parts, 3 = parts + verifier (see src/agent.py)")
    args = ap.parse_args()
    cfg = CONFIG["agent_configs"][args.config]
    retrievers, emb = build([cfg["retriever"]])
    retr = retrievers[cfg["retriever"]]
    retr.search("warm up", top_k=1)
    agent = Agent(retr, top_k=cfg["top_k"], budget_chars=cfg["budget_chars"],
                  context_order=cfg["context_order"], max_searches=cfg["max_searches"], version=args.agent_version)
    chunks = {c["id"]: c for c in load_chunks()}
    qs = load_questions(args.split)
    if args.only:
        qs = [q for q in qs if q["id"] in args.only]
    traces = []
    emb_tokens0 = emb.tokens
    for q in qs:
        t = time.perf_counter()
        try:
            r = agent.run(q["question"], q.get("filter"))  # no gold passed
        except Exception as e:  # provider / database error: record, never fake
            r = {"status": "error", "answer": None, "citations": [], "events": [{"event": "error", "error": repr(e),
                 "traceback": traceback.format_exc(limit=3)}], "final_context": "", "final_context_ids": [],
                 "searches": 0, "usage": {}}
        r["elapsed_ms"] = round((time.perf_counter() - t) * 1000, 1)
        ev = None
        if q["answerable"]:
            ev = any(is_hit(chunks[i], q["gold"]) for i in r["final_context_ids"])
        traces.append({"query_id": q["id"], "split": q["split"], "answerable": q["answerable"],
                       "question": q["question"], "filter": q.get("filter"), "config": args.config,
                       "model": agent.model, "agent_version": args.agent_version, **r, "post_hoc": {"context_evidence": ev}})
        print(f"{q['id']} {r['status']:8s} searches={r['searches']} ctx_evidence={ev} "
              f"{(r['answer'] or '')[:110]!r}")
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    llm_cost = sum(t["usage"].get("llm_cost_usd", 0) for t in ok)
    emb_cost = (emb.tokens - emb_tokens0) / 1e6 * CONFIG["day6"]["embedding_price_per_mtok_usd"]
    summary = {
        "config": args.config, **cfg, "split": args.split, "model": agent.model, "agent_version": args.agent_version,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": len(traces), "errors": len(traces) - len(ok),
        "status": dict(Counter(t["status"] for t in traces)),
        "tool_use": {"n": sum(t["searches"] > 0 for t in ok), "N": len(ok)},
        "searches_distribution": dict(sorted(Counter(t["searches"] for t in ok).items())),
        "context_evidence": {"n": sum(bool(t["post_hoc"]["context_evidence"]) for t in ans), "N": len(ans)},
        "false_abstention_auto": {"n": sum(t["status"] == "abstain" for t in ans), "N": len(ans)},
        "unanswerable_abstained_auto": {"n": sum(t["status"] == "abstain" for t in ok if not t["answerable"]),
                                        "N": sum(not t["answerable"] for t in ok)},
        "avg_context_chars": round(statistics.mean(len(t["final_context"]) for t in ok), 1) if ok else None,
        "input_tokens": sum(t["usage"].get("input_tokens", 0) for t in ok),
        "output_tokens": sum(t["usage"].get("output_tokens", 0) for t in ok),
        "llm_cost_usd": round(llm_cost, 5), "embedding_cost_usd": round(emb_cost, 6),
        "median_latency_s": round(statistics.median(t["elapsed_ms"] for t in ok) / 1000, 2) if ok else None,
        "total_latency_s": round(sum(t["elapsed_ms"] for t in traces) / 1000, 1),
    }
    out = RESULTS / "agent" / f"v{args.agent_version}" / f"{args.config}_{args.split}"
    write_json(out / "traces.json", traces)
    if not args.only:
        write_json(out / "auto_summary.json", summary)
    print({k: summary[k] for k in ("status", "tool_use", "searches_distribution", "context_evidence",
                                   "llm_cost_usd", "median_latency_s")})


if __name__ == "__main__":
    main()
