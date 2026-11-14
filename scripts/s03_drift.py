#!/usr/bin/env python3
"""How far the proportional estimate (char offset / chars * duration) drifts from
each caption cue's real start, per video. No retrieval involved.

Out: out/05_drift.json and a table on stdout.
"""

import json
import statistics
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import cue_offsets, duration, load  # noqa: E402

rows = load()
res = []
for r in rows:
    offs, n, _ = cue_offsets(r)
    dur = duration(r)
    errs = [o / n * dur - float(s["start"]) for o, s in zip(offs, r["segments"])]
    ab = sorted(abs(e) for e in errs)
    # speaking-rate spread: chars per second over 60-second windows
    segs = r["segments"]
    rate = []
    w = 60.0
    t = 0.0
    while t + w <= dur:
        ch = sum(len(s["text"]) + 1 for s in segs if t <= float(s["start"]) < t + w)
        rate.append(ch / w)
        t += w
    res.append({
        "vid": r["videoId"], "caption": r.get("captionKind"), "title": r["title"][:50],
        "duration_min": round(dur / 60, 1),
        "median_abs_drift_s": round(statistics.median(ab), 1),
        "p90_abs_drift_s": round(ab[int(0.9 * len(ab))], 1),
        "max_abs_drift_s": round(ab[-1], 1),
        "share_cues_drift_over_30s_pct": round(100 * sum(1 for e in ab if e > 30) / len(ab), 1),
        "early_share_pct": round(100 * sum(1 for e in errs if e < -1) / len(errs), 1),
        "chars_per_s_p10_p90": [round(sorted(rate)[int(0.1 * len(rate))], 1), round(sorted(rate)[int(0.9 * len(rate))], 1)] if rate else None,
    })
res.sort(key=lambda x: x["duration_min"])
for x in res:
    print(f"{x['vid']} {x['caption']:6s} {x['duration_min']:6.1f}min med={x['median_abs_drift_s']:6.1f} "
          f"p90={x['p90_abs_drift_s']:6.1f} max={x['max_abs_drift_s']:6.1f} >30s={x['share_cues_drift_over_30s_pct']:5.1f}% "
          f"rate={x['chars_per_s_p10_p90']}  {x['title']}")
# rank correlation duration vs median drift (Spearman, no ties expected)
d = [x["duration_min"] for x in res]
m = [x["median_abs_drift_s"] for x in res]
def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0] * len(v)
    for r_, i in enumerate(o):
        rk[i] = r_
    return rk
rd, rm = ranks(d), ranks(m)
nn = len(d)
rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rd, rm)) / (nn * (nn * nn - 1))
print("spearman(duration, median drift) =", round(rho, 3))
json.dump({"videos": res, "spearman_duration_vs_median_drift": round(rho, 3)},
          open("out/05_drift.json", "w"), indent=1)
