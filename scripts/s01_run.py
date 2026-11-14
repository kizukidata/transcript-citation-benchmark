#!/usr/bin/env python3
"""Run every arm against one probe/query set and one retriever.

  python scripts/s01_run.py --retriever bm25 --out out/01_bm25_verbatim.json
  python scripts/s01_run.py --retriever bm25 --queries out/queries_paraphrase.jsonl --out ...
  python scripts/s01_run.py --retriever dense --out ...

--queries: JSONL of {"pid", "query"}; only those probes are scored, with that text.
Small-to-big arms (s2b_*): retrieve with a big passage, then cite the best cue
(or Actor paragraph) inside it, ranked by the same retriever.
"""

import argparse
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, __import__("os").path.dirname(__file__))
from arms import build_arms  # noqa: E402
from common import BM25, load, make_probes, rng, summarize  # noqa: E402

S2B_BIG = ["chunk1500", "win120", "cue1500", "win60"]
S2B_SMALL = ["cue1", "para"]


class DenseIndex:
    def __init__(self, texts, model, cache_dir="emb_cache"):
        import hashlib
        import numpy as np
        self.np = np
        self.model = model
        h = hashlib.sha256("\x1e".join(texts).encode("utf-8")).hexdigest()[:24]
        path = os.path.join(cache_dir, f"bge-small-{h}.npy")
        if os.path.exists(path):
            self.m = np.load(path)
        else:
            self.m = self._embed(texts)
            os.makedirs(cache_dir, exist_ok=True)
            np.save(path, self.m)

    def _embed(self, texts):
        np = self.np
        v = np.array(list(self.model.embed(texts, batch_size=64)), dtype="float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v

    def top(self, query, restrict=None, qv=None):
        np = self.np
        if qv is None:
            qv = self._embed([query])[0]
        if restrict:
            idx = sorted(restrict)
            s = self.m[idx] @ qv
            return idx[int(np.argmax(s))]
        return int(np.argmax(self.m @ qv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.json")
    ap.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--queries", default=None)
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--caption", choices=["manual", "auto"])
    ap.add_argument("--jitter", action="store_true", help="probe times off the 60-second grid")
    ap.add_argument("--arms", default=None, help="comma list; default all")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-probe", default=None, help="write per-probe cited seconds (JSONL)")
    a = ap.parse_args()

    rows = load(a.data)
    if a.caption:
        rows = [r for r in rows if r.get("captionKind") == a.caption]
    probes = make_probes(rows, a.every, jitter=a.jitter)
    if a.queries:
        q = {}
        for line in open(a.queries, encoding="utf-8"):
            if line.strip():
                o = json.loads(line)
                q[o["pid"]] = o["query"]
        probes = [dict(p, query=q[p["pid"]]) for p in probes if p["pid"] in q]
    arms = build_arms(rows)
    if a.arms:
        keep = set(a.arms.split(","))
        arms = {k: v for k, v in arms.items() if k in keep or k in S2B_SMALL or k in S2B_BIG}

    model = None
    qcache = {}
    if a.retriever == "dense":
        from fastembed import TextEmbedding
        model = TextEmbedding("BAAI/bge-small-en-v1.5", threads=4)  # keep to 4 CPU threads
        import numpy as np
        qs = [p["query"] for p in probes]
        qv = np.array(list(model.embed(qs, batch_size=64)), dtype="float32")
        qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9
        qcache = {p["pid"]: qv[i] for i, p in enumerate(probes)}

    def make_index(docs):
        texts = [d["text"] for d in docs]
        return DenseIndex(texts, model) if model else BM25(texts)

    def top(idx, p, restrict=None):
        if model:
            return idx.top(p["query"], restrict, qcache[p["pid"]])
        return idx.top(p["query"], restrict)

    indexes = {name: make_index(docs) for name, docs in arms.items()}
    # lookup: (vid, cue) -> passage ids of each small arm
    small_by_vid_cue = {}
    for sname in S2B_SMALL:
        if sname not in arms:
            continue
        m = defaultdict(set)
        for i, d in enumerate(arms[sname]):
            for k in range(d["c0"], d["c1"] + 1):
                m[(d["vid"], k)].add(i)
        small_by_vid_cue[sname] = m

    vids = sorted({p["vid"] for p in probes})
    plan = [(name, None) for name in arms]
    plan += [(f"s2b_{b}_to_{s}", (b, s)) for b in S2B_BIG for s in S2B_SMALL if b in arms and s in arms]

    res = {"probes": len(probes), "jitter": a.jitter, "every": a.every, "videos": len(vids), "retriever": a.retriever,
           "queries": a.queries or "verbatim (cue + two neighbours, v1 rule)", "arms": {}}
    per_probe = []
    R = rng()
    for name, s2b in plan:
        signed, per_vid = [], defaultdict(list)
        for p in probes:
            if s2b is None:
                d = arms[name][top(indexes[name], p)]
            else:
                big, small = s2b
                bd = arms[big][top(indexes[big], p)]
                cand = set()
                for k in range(bd["c0"], bd["c1"] + 1):
                    cand |= small_by_vid_cue[small].get((bd["vid"], k), set())
                d = arms[small][top(indexes[small], p, cand)] if cand else bd
            e = (d["start"] - p["true"]) if d["vid"] == p["vid"] else None
            signed.append(e)
            per_vid[p["vid"]].append(e)
            per_probe.append({"arm": name, "pid": p["pid"], "cited": d["start"], "true": p["true"],
                              "right_video": d["vid"] == p["vid"]})
        out = summarize(signed, len(probes) or 1)
        docs = arms[name] if s2b is None else arms[s2b[1]]
        secs = [max(0.0, d["end"] - d["start"]) for d in docs]
        out["passages"] = len(arms[name]) if s2b is None else f"{len(arms[s2b[0]])} -> {len(arms[s2b[1]])}"
        out["median_passage_s"] = round(statistics.median(secs), 1)
        out["avg_chars"] = round(statistics.mean(len(d["text"]) for d in docs))
        # per video
        pv = {}
        for v in vids:
            es = per_vid[v]
            pv[v] = round(100 * sum(1 for e in es if e is not None and abs(e) <= 30) / max(len(es), 1), 1)
        out["within_30s_by_video"] = pv
        # bootstrap over videos
        boots = []
        for _ in range(a.boot):
            pick = [R.choice(vids) for _ in vids]
            hit = sum(sum(1 for e in per_vid[v] if e is not None and abs(e) <= 30) for v in pick)
            n = sum(len(per_vid[v]) for v in pick)
            boots.append(100 * hit / max(n, 1))
        boots.sort()
        out["within_30s_ci95_video_bootstrap"] = [round(boots[int(0.025 * len(boots))], 1),
                                                  round(boots[int(0.975 * len(boots)) - 1], 1)]
        res["arms"][name] = out
        print(f"{name:28s} n={out['passages']!s:>12} med_s={out['median_passage_s']:>6} "
              f"rv={out['right_video_pct']:>5} w30={out['within_30s_pct']:>5} "
              f"ci={out['within_30s_ci95_video_bootstrap']} med={out['median_abs_error_s']} "
              f"play={out['playable_30s_pct']} after={out['lands_after_pct']}", flush=True)

    open(a.out, "w", encoding="utf-8").write(json.dumps(res, indent=1) + "\n")
    if a.per_probe:
        with open(a.per_probe, "w", encoding="utf-8") as f:
            for x in per_probe:
                f.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    main()
