#!/usr/bin/env python3
"""Tables and figures from out/0*_*.json and the per-probe files.

Adds, per arm and setting, the share within 30 s *given the right video*
(separates "found the video" from "found the second").
Writes out/08_tables.md, out/08_tables.json, fig/fig1_size_curve.png, fig/fig2_grid_artifact.png
"""

import json
import os
import sys
from collections import defaultdict

SETTINGS = [
    ("bm25_grid", "BM25, verbatim, v1 grid", "out/01_bm25_verbatim"),
    ("bm25_verb", "BM25, verbatim", "out/02_bm25_verbatim_jitter"),
    ("dense_verb", "Dense, verbatim", "out/03_dense_verbatim_jitter"),
    ("bm25_para", "BM25, paraphrased", "out/06_bm25_paraphrase"),
    ("dense_para", "Dense, paraphrased", "out/07_dense_paraphrase"),
]


def cond(path):
    """within-30 given right video, per arm, from the per-probe JSONL."""
    agg = defaultdict(lambda: [0, 0])
    if not os.path.exists(path):
        return {}
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        if o["right_video"]:
            a = agg[o["arm"]]
            a[0] += 1
            a[1] += 1 if abs(o["cited"] - o["true"]) <= 30 else 0
    return {k: round(100 * h / n, 1) for k, (n, h) in agg.items() if n}


def main():
    data = {}
    for key, label, base in SETTINGS:
        if not os.path.exists(base + ".json"):
            print("missing", base, file=sys.stderr)
            continue
        j = json.load(open(base + ".json"))
        c = cond(base + ".probes.jsonl")
        for arm, v in j["arms"].items():
            v["within_30s_given_right_video_pct"] = c.get(arm)
        data[key] = {"label": label, "probes": j["probes"], "arms": j["arms"]}
    json.dump(data, open("out/08_tables.json", "w"), indent=1)

    arms_order = ["char500_prop", "char500_true", "char1500_prop", "char1500_true", "cue250", "cue500", "cue1000",
                  "cue1500", "cue3000", "win30", "win60", "win120", "lc30", "lc60", "lc120", "para", "chunk1500",
                  "cue1", "s2b_win60_to_cue1", "s2b_chunk1500_to_cue1", "s2b_cue1500_to_cue1"]
    lines = ["| arm | median passage s | " + " | ".join(
        f"{data[k]['label']} (n={data[k]['probes']}) playable / two-sided within 30 s [CI] / right video / two-sided given right video"
        for k in data) + " |", "|---|---:|" + "---:|" * len(data)]
    for arm in arms_order:
        row = [arm]
        med = None
        cells = []
        for k in data:
            v = data[k]["arms"].get(arm)
            if v:
                med = v.get("median_passage_s")
                cells.append(f"{v['playable_30s_pct']} / {v['within_30s_pct']} [{v['within_30s_ci95_video_bootstrap'][0]}, "
                             f"{v['within_30s_ci95_video_bootstrap'][1]}] / {v['right_video_pct']} / "
                             f"{v['within_30s_given_right_video_pct']}")
            else:
                cells.append("—")
        lines.append("| " + arm + f" | {med} | " + " | ".join(cells) + " |")
    open("out/08_tables.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = {"cues packed to N chars": ["cue250", "cue500", "cue1000", "cue1500", "cue3000"],
            "N-char pieces, second of the cue they start in": ["char250_true", "char500_true", "char1000_true", "char1500_true", "char3000_true"],
            "N-char pieces, second estimated from the offset": ["char250_prop", "char500_prop", "char1000_prop", "char1500_prop", "char3000_prop"],
            "time windows": ["win15", "win30", "win60", "win120", "win300"]}
    singles = {"transcript-tool paragraphs": "para", "LangChain 120 s chunks (default)": "lc120"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    import numpy as np
    xs_f = np.geomspace(10, 350, 200)
    axes[0].plot(xs_f, [100 * min(1, 30 / x) for x in xs_f], "k--", lw=1, label="formula min(1, 30/L)")
    for ax, key in zip(axes[:1], ["bm25_verb"]):
        if key not in data:
            continue
        A = data[key]["arms"]
        for name, arms in fams.items():
            xs = [A[a]["median_passage_s"] for a in arms if a in A]
            ys = [A[a]["playable_30s_pct"] for a in arms if a in A]
            if xs:
                ax.plot(xs, ys, marker="o", label=name)
        for name, a in singles.items():
            if a in A:
                ax.scatter([A[a]["median_passage_s"]], [A[a]["playable_30s_pct"]], marker="*", s=140, zorder=5, label=name)
        ax.set_xscale("log")
        ax.set_xticks([15, 30, 60, 120, 300])
        ax.set_xticklabels(["15", "30", "60", "120", "300"])
        ax.set_xlim(10, 350)
        ax.minorticks_off()
        ax.set_xlabel("median passage length (seconds, log)")
        ax.set_title(data[key]["label"] + f" ({data[key]['probes']} probes)")
        ax.grid(alpha=.3)
    ax = axes[1]
    if os.path.exists("out/11_budget.json"):
        B = json.load(open("out/11_budget.json"))
        styles = {"bm25": "-", "dense": "--"}
        names = {"win30": "30 s windows", "win60": "60 s windows", "win120": "120 s windows",
                 "lc120": "LangChain 120 s chunks"}
        colors = {"win30": "tab:red", "win60": "tab:orange", "win120": "tab:blue", "lc120": "tab:green"}
        for ret, ls in styles.items():
            for arm in names:
                pts = sorted((int(k.split("@")[1]), v["right_passage_pct"]) for k, v in B[ret]["table"].items()
                             if k.split("@")[0] == arm)
                if pts:
                    ax.plot([x for x, _ in pts], [y for _, y in pts], ls, marker="o", color=colors[arm],
                            label=f"{names[arm]}, {'BM25' if ret == 'bm25' else 'dense'}")
        ax.set_xscale("log")
        ax.set_xticks([150, 300, 600, 1200])
        ax.set_xticklabels(["150", "300", "600", "1,200"])
        ax.minorticks_off()
        ax.set_xlabel("seconds of transcript handed to the LLM (log)")
        ax.set_ylabel("right passage among them (%)")
        ax.set_title("Rewritten questions: same transcript budget (322)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    axes[0].set_title("Verbatim queries, BM25 (1,257 probes)")
    axes[0].set_ylabel("cited 0 to 30 s before the moment (%)")
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    fig.savefig("fig/fig1_size_curve.png", dpi=150)

    if "bm25_grid" in data and "bm25_verb" in data:
        arms = ["win30", "lc30", "win60", "lc60", "win120", "lc120", "para"]
        labels = ["30 s\nwindows", "LangChain\n30 s", "60 s\nwindows", "LangChain\n60 s", "120 s\nwindows",
                  "LangChain\n120 s", "paragraphs"]
        g = [data["bm25_grid"]["arms"][a]["playable_30s_pct"] for a in arms]
        j = [data["bm25_verb"]["arms"][a]["playable_30s_pct"] for a in arms]
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        x = range(len(arms))
        ax.bar([i - .2 for i in x], g, width=.4, label="probes on the minute")
        ax.bar([i + .2 for i in x], j, width=.4, label="probes at random seconds")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("cited 0 to 30 s before (%)")
        ax.set_title("Same index and query rule; only the probe times differ (BM25)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("fig/fig2_grid_artifact.png", dpi=150)


if __name__ == "__main__":
    main()
