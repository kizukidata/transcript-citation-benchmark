#!/usr/bin/env python3
"""Pick a seeded sample of jittered probes and write the context a question writer sees.

Output: out/paraphrase_sample.jsonl, one line per probe:
  {pid, vid, title, context_before, target, context_after}
target = the three cues of the v1 query; the writer sees ~30 s on each side.
"""

import json
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import load, make_probes, rng  # noqa: E402

PER_VIDEO = 25

rows = load()
by = {r["videoId"]: r for r in rows}
probes = make_probes(rows, 60.0, jitter=True)
R = rng(7)
out = []
for vid in sorted({p["vid"] for p in probes}):
    ps = [p for p in probes if p["vid"] == vid]
    pick = sorted(R.sample(ps, min(PER_VIDEO, len(ps))), key=lambda p: p["true"])
    segs = by[vid]["segments"]
    for p in pick:
        k = p["cue"]
        a, b = max(0, k - 1), k + 2
        before = [s for s in segs[:a] if s["start"] >= segs[a]["start"] - 30]
        after = [s for s in segs[b:] if s["start"] <= segs[min(b, len(segs) - 1)]["start"] + 30]
        out.append({"pid": p["pid"], "vid": vid, "title": by[vid]["title"],
                    "context_before": " ".join(s["text"].strip() for s in before),
                    "target": " ".join(s["text"].strip() for s in segs[a:b]),
                    "context_after": " ".join(s["text"].strip() for s in after)})
with open("out/paraphrase_sample.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(len(out), "probes from", len({o["vid"] for o in out}), "videos")
