#!/usr/bin/env python3
"""Budget-matched comparison: the same seconds of transcript handed to the LLM.

Reads out/10_topk_<retriever>_paraphrase.ranks.jsonl (from s06_topk.py). For a budget B seconds and a
passage length T, k = B / T passages are handed over. Reports the share of questions whose right
passage (the one holding the probe cue) is among them, and paired differences with a video-level
bootstrap (4,000 resamples, seed 20261113).
Writes out/11_budget.json and out/11_budget.md.
"""

import json
import random
from collections import defaultdict

ARMS = {"win30": 30, "win60": 60, "win120": 120, "lc120": 120}
BUDGETS = [150, 300, 600, 1200]
PAIRS = [("lc120", 5, "win30", 20), ("win120", 5, "win30", 20), ("win60", 10, "win30", 20),
         ("lc120", 5, "win30", 5), ("lc120", 10, "win30", 40)]


def load(ret):
    by = defaultdict(dict)
    for line in open(f"out/10_topk_{ret}_paraphrase.ranks.jsonl"):
        o = json.loads(line)
        by[o["arm"]][o["pid"]] = o["hit"]
    return by


def share(by, arm, k, pids):
    return 100 * sum(1 for p in pids if by[arm][p] is not None and by[arm][p] < k) / len(pids)


def main():
    out, lines = {}, []
    for ret in ["bm25", "dense"]:
        by = load(ret)
        pids = sorted(by["win30"])
        vids = defaultdict(list)
        for p in pids:
            vids[p.split("@")[0]].append(p)
        vl = sorted(vids)
        table = {}
        for b in BUDGETS:
            for arm, t in ARMS.items():
                k = b // t
                if k >= 1 and k <= 40 and b % t == 0:
                    table[f"{arm}@{b}"] = {"k": k, "right_passage_pct": round(share(by, arm, k, pids), 1)}
        R = random.Random(20261113)
        diffs = {}
        for a, ka, c, kc in PAIRS:
            point = share(by, a, ka, pids) - share(by, c, kc, pids)
            boots = []
            for _ in range(4000):
                pick = [p for _ in vl for p in vids[vl[R.randrange(len(vl))]]]
                boots.append(share(by, a, ka, pick) - share(by, c, kc, pick))
            boots.sort()
            diffs[f"{a}_k{ka}-{c}_k{kc}"] = {"diff": round(point, 1),
                                             "ci95": [round(boots[100], 1), round(boots[3899], 1)]}
        out[ret] = {"table": table, "diffs": diffs}
        lines.append(f"## {ret}")
        lines.append("| budget s | " + " | ".join(ARMS) + " |")
        lines.append("|---:|" + "---:|" * len(ARMS))
        for b in BUDGETS:
            cells = [f"k={table[f'{a}@{b}']['k']}: {table[f'{a}@{b}']['right_passage_pct']}" if f"{a}@{b}" in table else "—"
                     for a in ARMS]
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        for name, d in diffs.items():
            lines.append(f"- {name}: {d['diff']} [{d['ci95'][0]}, {d['ci95'][1]}]")
    json.dump(out, open("out/11_budget.json", "w"), indent=1)
    open("out/11_budget.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
