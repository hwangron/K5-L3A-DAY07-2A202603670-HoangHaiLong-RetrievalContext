"""Turn hand labels into reviews.json for starter/agent_metrics.py.

    python -m src.make_reviews --config A1_dense_budget --split dev

Input results/agent/<config>_<split>/labels.json, written by a human after reading each trace:
  {"q01": {"grade": 2, "unsupported": [], "note": "..."},          # answerable
   "u01": {"abstention_correct": true, "note": "..."}}             # unanswerable
`unsupported` lists cited chunk IDs that do NOT support the answer; all other citations count as
supporting. Abstained answerable questions must have grade 0. Error runs need no label."""
import argparse

from .common import RESULTS, read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--agent-version", type=int, default=3, choices=(1, 2, 3))
    args = ap.parse_args()
    d = RESULTS / "agent" / f"v{args.agent_version}" / f"{args.config}_{args.split}"
    traces, labels = read_json(d / "traces.json"), read_json(d / "labels.json")
    reviews = []
    for t in traces:
        if t["status"] == "error":
            continue
        lab = labels[t["query_id"]]
        r = {"query_id": t["query_id"], "citation_support": [c not in lab.get("unsupported", []) for c in t["citations"]],
             "note": lab.get("note", "")}
        if t["answerable"]:
            r["answer_grade"] = lab["grade"]
        else:
            r["abstention_correct"] = lab["abstention_correct"]
        reviews.append(r)
    write_json(d / "reviews.json", reviews)
    print(f"{len(reviews)} reviews written to {d / 'reviews.json'}")


if __name__ == "__main__":
    main()
