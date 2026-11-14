#!/usr/bin/env python3
"""Top-k view: is a usable passage among the top k, for a pipeline where an LLM reads k passages?

For each probe and arm, rank all passages of all videos (BM25 or dense) and report, for k in 1, 3, 5, 10, 20, 40:
  right_passage_at_k   a passage that holds the probe cue (same video, c0 <= cue <= c1) is in the top k
                       (ceiling for a pipeline that stores each cue's start second inside its passages)
  cite30_at_k          some passage in the top k cites a second within 30 s of the right second, either side
  play30_at_k          some passage in the top k cites from 30 s before the right second up to 1 s after it
                       (pressing play reaches the moment; the best an LLM choosing among the k could do)
Usage:
  scripts/s06_topk.py --retriever bm25|dense [--queries out/queries_paraphrase.jsonl] --arms a,b --out out/10_x.json
Also writes <out>.ranks.jsonl: per probe and arm, the 0-based rank of the first right passage and of the
first playable start within the top 40 (null if none), for budget-matched comparisons (s07_budget.py).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from arms import build_arms  # noqa: E402
from common import BM25, load, make_probes  # noqa: E402
from s01_run import DenseIndex  # noqa: E402

KS = [1, 3, 5, 10, 20, 40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--queries", default=None)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = load()
    probes = make_probes(rows, 60.0, jitter=True)
    if a.queries:
        q = {}
        for line in open(a.queries, encoding="utf-8"):
            if line.strip():
                o = json.loads(line)
                q[o["pid"]] = o["query"]
        probes = [dict(p, query=q[p["pid"]]) for p in probes if p["pid"] in q]
    keep = a.arms.split(",")
    arms = {k: v for k, v in build_arms(rows).items() if k in keep}

    model, qv = None, None
    if a.retriever == "dense":
        import numpy as np
        from fastembed import TextEmbedding
        model = TextEmbedding("BAAI/bge-small-en-v1.5", threads=4)
        qv = np.array(list(model.embed([p["query"] for p in probes], batch_size=64)), dtype="float32")
        qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9

    all_ranks = []
    res = {"probes": len(probes), "retriever": a.retriever, "queries": a.queries or "verbatim", "arms": {}}
    for name in keep:
        docs = arms[name]
        texts = [d["text"] for d in docs]
        idx = DenseIndex(texts, model) if model else BM25(texts)
        hit = {k: 0 for k in KS}
        c30 = {k: 0 for k in KS}
        p30 = {k: 0 for k in KS}
        ranks = []
        for i, p in enumerate(probes):
            if model:
                s = idx.m @ qv[i]
                order = list(np.argsort(-s)[:max(KS)])
            else:
                acc = idx.scores(p["query"])
                order = sorted(acc, key=lambda j: (-acc[j], j))[:max(KS)]
            first_hit = first_c30 = first_p30 = None
            for r, j in enumerate(order):
                d = docs[j]
                if d["vid"] != p["vid"]:
                    continue
                if first_hit is None and d["c0"] <= p["cue"] <= d["c1"]:
                    first_hit = r
                if first_c30 is None and abs(d["start"] - p["true"]) <= 30:
                    first_c30 = r
                if first_p30 is None and -30 <= d["start"] - p["true"] <= 1:
                    first_p30 = r
            ranks.append({"arm": name, "pid": p["pid"], "hit": first_hit, "p30": first_p30})
            for k in KS:
                hit[k] += first_hit is not None and first_hit < k
                c30[k] += first_c30 is not None and first_c30 < k
                p30[k] += first_p30 is not None and first_p30 < k
        n = len(probes)
        res["arms"][name] = {f"right_passage_at_{k}": round(100 * hit[k] / n, 1) for k in KS}
        res["arms"][name].update({f"cite30_at_{k}": round(100 * c30[k] / n, 1) for k in KS})
        res["arms"][name].update({f"play30_at_{k}": round(100 * p30[k] / n, 1) for k in KS})
        print(name, res["arms"][name], flush=True)
        all_ranks.extend(ranks)
    json.dump(res, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", ".ranks.jsonl"), "w") as f:
        for r in all_ranks:
            f.write(json.dumps({k: (int(v) if hasattr(v, "item") else v) for k, v in r.items()}) + "\n")


if __name__ == "__main__":
    main()
