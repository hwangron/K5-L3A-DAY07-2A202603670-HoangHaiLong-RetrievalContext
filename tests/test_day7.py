"""Offline checks for the Day 7 code (BM25, RRF, context packing, agent loop).
Fake model/retriever objects only test control flow; they never measure quality."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import Agent, pack  # noqa: E402
from src.common import is_hit  # noqa: E402
from src.retrievers import BM25, HybridRRF  # noqa: E402


def chunk(i, text, page=1, section="2.3"):
    return {"id": i, "text": text, "source_id": "doc", "source": "doc.pdf", "page": page,
            "metadata": {"section": section, "chapter": 2}}


CHUNKS = [chunk("c1", "The squeeze theorem bounds a function between two others."),
          chunk("c2", "Multiply by the conjugate when a square root appears.", 2),
          chunk("c3", "Continuity needs three conditions.", 3, "2.4")]


class FakeDense:
    def __init__(self, order): self.order = order
    def search(self, query, *, filters=None, top_k=3):
        rows = [c for c in CHUNKS if c["id"] in self.order]
        rows.sort(key=lambda c: self.order.index(c["id"]))
        return [{**c, "score": 1.0 - i / 10, "rank": i + 1, "method": "dense"} for i, c in enumerate(rows[:top_k])]


class FakeClient:
    """Returns scripted tool calls in order."""
    def __init__(self, calls):
        self.calls, self.seen = iter(calls), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.seen.append(kw)
        name, args = next(self.calls)
        if name == "verify":  # verifier call returns JSON content, not a tool call
            msg = SimpleNamespace(content=json.dumps(args), tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                                   usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))
        tc = SimpleNamespace(id=f"t{len(self.seen)}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        msg = SimpleNamespace(content=None, tool_calls=[tc])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def agent(calls, order=("c2", "c1"), **kw):
    opts = dict(top_k=2, budget_chars=3000, context_order="rank_interleave", max_searches=2, version=1)
    opts.update(kw)
    client = FakeClient(calls)
    return Agent(FakeDense(list(order)), client=client, model="fake", **opts), client


class RetrievalTests(unittest.TestCase):
    def test_bm25_prefers_exact_terms_and_filters(self):
        bm = BM25(CHUNKS)
        self.assertEqual(bm.search("conjugate square root")[0]["id"], "c2")
        self.assertEqual([r["id"] for r in bm.search("continuity conditions", filters={"section": "2.3"})], [])

    def test_rrf_uses_ranks_of_both_lists(self):
        h = HybridRRF(FakeDense(["c1", "c3"]), BM25(CHUNKS), candidates=3, k=60)
        top = h.search("conjugate square root", top_k=3)
        self.assertEqual({r["id"] for r in top}, {"c1", "c2", "c3"})
        self.assertIn("bm25", next(r for r in top if r["id"] == "c2")["sources"])

    def test_hit_needs_page_and_full_quote(self):
        gold = [{"source_id": "doc", "page": 2, "evidence": "multiply  by the CONJUGATE"}]
        self.assertTrue(is_hit(CHUNKS[1], gold))
        self.assertFalse(is_hit(CHUNKS[0], gold))


class ContextTests(unittest.TestCase):
    def test_policies_and_budget(self):
        cands = [{**CHUNKS[0], "rank": 1, "_search": 0, "score": 1}, {**CHUNKS[1], "rank": 2, "_search": 0, "score": 1},
                 {**CHUNKS[2], "rank": 1, "_search": 1, "score": 1}]
        _, first, _ = pack(cands, 3000, "keep_first")
        _, inter, _ = pack(cands, 3000, "rank_interleave")
        self.assertEqual(first, ["c1", "c2", "c3"])
        self.assertEqual(inter, ["c1", "c3", "c2"])
        _, small, dropped = pack(cands, 120, "keep_first")
        self.assertEqual(len(small), 1)
        self.assertEqual(len(dropped), 2)


class AgentLoopTests(unittest.TestCase):
    def test_answer_with_valid_citation(self):
        a, client = agent([("search", {"query": "conjugate"}),
                           ("answer", {"answer": "Use the conjugate [c2].", "citations": ["c2"], "abstain": False})])
        r = a.run("square root trick?")
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["citations"], ["c2"])
        self.assertEqual(client.seen[0]["tool_choice"]["function"]["name"], "search")

    def test_citation_outside_context_is_repaired_then_abstains(self):
        bad = ("answer", {"answer": "Answer [zzz].", "citations": ["zzz"], "abstain": False})
        a, _ = agent([("search", {"query": "x"}), bad, bad])
        r = a.run("q")
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(sum(e["event"] == "invalid_or_missing_citations" for e in r["events"]), 2)

    def test_search_limit_only_offers_answer(self):
        a, client = agent([("search", {"query": "a"}), ("search", {"query": "b"}),
                           ("answer", {"answer": "missing", "citations": [], "abstain": True})])
        r = a.run("q")
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(r["searches"], 2)
        self.assertEqual([t["function"]["name"] for t in client.seen[-1]["tools"]], ["answer"])

    def test_old_context_is_not_repeated(self):
        a, client = agent([("search", {"query": "a"}), ("search", {"query": "b"}),
                           ("answer", {"answer": "x [c2]", "citations": ["c2"], "abstain": False})])
        a.run("q")
        tool_msgs = [m for m in client.seen[-1]["messages"] if m["role"] == "tool"]
        self.assertTrue(tool_msgs[0]["content"].startswith("[superseded"))


class AgentV2Tests(unittest.TestCase):
    def test_parts_are_rendered_with_inline_ids(self):
        a, _ = agent([("search", {"query": "x"}),
                      ("answer", {"parts": [{"text": "Use the conjugate.", "citations": ["c2"]}],
                                  "abstain": False, "missing": ""})], version=2)
        r = a.run("q")
        self.assertEqual((r["status"], r["answer"], r["citations"]), ("answered", "Use the conjugate. [c2]", ["c2"]))

    def test_part_without_citation_is_rejected(self):
        bad = ("answer", {"parts": [{"text": "From memory.", "citations": []}], "abstain": False, "missing": ""})
        a, _ = agent([("search", {"query": "x"}), bad, bad], version=2)
        self.assertEqual(a.run("q")["status"], "abstain")

    def test_abstain_hides_model_text(self):
        a, _ = agent([("search", {"query": "x"}),
                      ("answer", {"parts": [{"text": "The chain rule says ...", "citations": []}],
                                  "abstain": True, "missing": "chain rule not in excerpts"})], version=2)
        r = a.run("q")
        self.assertEqual(r["status"], "abstain")
        self.assertNotIn("chain rule", r["answer"])
        self.assertEqual(r["events"][-1]["model_text"], "chain rule not in excerpts")


class AgentV3Tests(unittest.TestCase):
    ANS = ("answer", {"parts": [{"text": "The chain rule is f'(g)g'.", "citations": ["c1"]}],
                      "abstain": False, "missing": ""})

    def test_unsupported_claim_is_repaired_then_abstains(self):
        no = ("verify", {"parts": [{"index": 1, "supported": False, "reason": "not in text"}]})
        a, _ = agent([("search", {"query": "x"}), self.ANS, no, self.ANS, no], version=3)
        r = a.run("q")
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(sum(e["event"] == "unsupported_claims" for e in r["events"]), 2)

    def test_supported_claim_is_answered(self):
        yes = ("verify", {"parts": [{"index": 1, "supported": True, "reason": "stated"}]})
        a, _ = agent([("search", {"query": "x"}), self.ANS, yes], version=3)
        r = a.run("q")
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["usage"]["verify_calls"], 1)

    def test_missing_verdict_counts_as_unsupported(self):
        empty = ("verify", {"parts": []})
        a, _ = agent([("search", {"query": "x"}), self.ANS, empty, self.ANS, empty], version=3)
        self.assertEqual(a.run("q")["status"], "abstain")


if __name__ == "__main__":
    unittest.main()
