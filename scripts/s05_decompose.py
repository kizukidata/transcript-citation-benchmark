#!/usr/bin/env python3
"""Split each top-1 citation into stages, and compare with the closed-form prediction.

Arms whose second is an estimate (*_prop) or a two-stage pick (s2b_*) are skipped. LangChain
(lc*) arms are included; their second is the window boundary.
The retrieved passage is looked up again by (video, cited second) and classified:
  wrong_video      top-1 passage from another video
  right_passage    the passage holds the probe cue (c0 <= cue <= c1). For lc* this differs from
                   the time range, because a cue that runs past a boundary is put in the next chunk.
  wrong_passage    right video, passage elsewhere in it
Metrics ("playable" = cited from 30 s before the right second up to 1 s after it, so pressing
play reaches the moment; "two-sided" = |cited - true| <= 30):
  pred                     closed form sum(L*min(1,30/L))/sum(L) over the arm's passages
  playable, two_sided      over all probes
  playable_if_right_passage, two_sided_if_right_passage
  mid_two_sided_if_right_passage   citing the passage midpoint instead (two-sided)
  wrong_passage_median_gap_s
Usage: scripts/s05_decompose.py <name> <out/NN_x.probes.jsonl> [...] -> out/09_decompose.json, .md
"""

import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from arms import build_arms  # noqa: E402
from common import load, make_probes, rng  # noqa: E402

SKIP = ("_prop", "s2b_")


def playable(e):
    return -30 <= e <= 1


def pct(x, n):
    return round(100 * x / n, 1) if n else None


def main():
    pairs = sys.argv[1:]
    rows = load()
    cue_of = {p["pid"]: p["cue"] for p in make_probes(rows, 60.0, jitter=True)}
    arms = build_arms(rows)
    index = {a: {(p["vid"], round(p["start"], 3)): p for p in ps} for a, ps in arms.items()}
    pred = {}
    for a, ps in arms.items():
        L = [max(0.0, p["end"] - p["start"]) for p in ps]
        pred[a] = round(100 * sum(x * min(1, 30 / x) for x in L if x > 0) / sum(L), 1)

    out = {}
    lines = ["| setting | arm | probes | formula | playable | two-sided | wrong video | wrong passage | right passage "
             "| playable if right passage | two-sided if right passage | midpoint two-sided if right passage "
             "| wrong-passage median gap s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i in range(0, len(pairs), 2):
        name, path = pairs[i], pairs[i + 1]
        recs = defaultdict(list)
        for line in open(path, encoding="utf-8"):
            o = json.loads(line)
            recs[o["arm"]].append(o)
        out[name] = {}
        for a, rs in recs.items():
            if a.startswith(SKIP) or a.endswith(SKIP) or a not in index:
                continue
            n = len(rs)
            c = defaultdict(int)
            gaps = []
            per_vid = defaultdict(list)
            for o in rs:
                e = o["cited"] - o["true"]
                per_vid[o["pid"].split("@")[0]].append(1 if o["right_video"] and playable(e) else 0)
                if o["right_video"]:
                    c["play"] += playable(e)
                    c["two"] += abs(e) <= 30
                else:
                    c["wrong_video"] += 1
                    continue
                v = o["pid"].split("@")[0]
                p = index[a].get((v, round(o["cited"], 3)))
                if p is None:
                    c["unmatched"] += 1
                    continue
                if p["c0"] <= cue_of[o["pid"]] <= p["c1"]:
                    c["right_passage"] += 1
                    c["rp_play"] += playable(e)
                    c["rp_two"] += abs(e) <= 30
                    c["rp_mid"] += abs((p["start"] + p["end"]) / 2 - o["true"]) <= 30
                else:
                    c["wrong_passage"] += 1
                    gaps.append(abs(e))
            rp = c["right_passage"]
            R = rng()
            vids = sorted(per_vid)
            boots = []
            for _ in range(2000):
                pick = [per_vid[vids[R.randrange(len(vids))]] for _ in vids]
                tot = sum(len(x) for x in pick)
                boots.append(100 * sum(sum(x) for x in pick) / tot if tot else 0.0)
            boots.sort()
            ci = [round(boots[int(0.025 * len(boots))], 1), round(boots[int(0.975 * len(boots)) - 1], 1)]
            res = {
                "probes": n, "pred_within30": pred[a],
                "playable": pct(c["play"], n), "playable_ci95_video_bootstrap": ci, "two_sided": pct(c["two"], n),
                "wrong_video_pct": pct(c["wrong_video"], n), "wrong_passage_pct": pct(c["wrong_passage"], n),
                "right_passage_pct": pct(rp, n), "right_passage_n": rp, "unmatched": c["unmatched"],
                "playable_if_right_passage": pct(c["rp_play"], rp),
                "two_sided_if_right_passage": pct(c["rp_two"], rp),
                "mid_two_sided_if_right_passage": pct(c["rp_mid"], rp),
                "wrong_passage_median_gap_s": round(statistics.median(gaps), 1) if gaps else None,
            }
            out[name][a] = res
            lines.append("| " + " | ".join(str(x) for x in [
                name, a, n, res["pred_within30"], f'{res["playable"]} {ci}', res["two_sided"], res["wrong_video_pct"],
                res["wrong_passage_pct"], res["right_passage_pct"], res["playable_if_right_passage"],
                res["two_sided_if_right_passage"], res["mid_two_sided_if_right_passage"],
                res["wrong_passage_median_gap_s"]]) + " |")
    json.dump(out, open("out/09_decompose.json", "w"), indent=1)
    open("out/09_decompose.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
