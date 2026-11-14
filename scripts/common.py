"""Shared pieces for the T5 citation study (v2).

Everything reads the dataset JSON of one kizukidata/youtube-transcripts run
(output "both"). No network, no API keys.
"""

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict

STOP = set("""a an the and or but if of to in on at for with from by as is are was were be been being
it its this that these those i you he she we they me my your our their them his her so do does did
not no yes uh um like just what which who whom how why when where there here than then also very can
will would could should have has had going gonna want know think really right okay ok well get got
one two all some any more most about into out up down over again only own same too s t don ll re ve""".split())

WORD = re.compile(r"[a-z0-9']+")
PARA = re.compile(r"^\[(\d+:)?\d+:\d+\]\(([^)]*?)[?&]t=(\d+)s\)\s*(.*)$")


def words(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2]


def load(path="data.json"):
    rows = json.load(open(path, encoding="utf-8"))
    if isinstance(rows, dict) and "items" in rows:
        rows = rows["items"]
    return [r for r in rows if not r.get("error") and r.get("segments")]


def duration(r):
    return float(r.get("durationSeconds") or r["segments"][-1]["start"])


def make_probes(rows, every=60.0, jitter=False, seed=20261113):
    """Same rule as v1 measure.py: one cue per `every` seconds from 60 s to dur-60 s,
    query = that cue and its two neighbours, kept if it has >= 5 content words.
    jitter=True moves each probe time by a seeded uniform offset in [0, every), so
    probes no longer sit on a 60-second grid that fixed-window chunks share."""
    out = []
    R = random.Random(seed)
    for r in rows:
        segs = r["segments"]
        if len(segs) < 10:
            continue
        dur = duration(r)
        t, k = 60.0, 0
        while t < dur - 60:
            tt = t + (R.uniform(0, every) if jitter else 0.0)
            if tt >= dur - 60:
                break
            while k < len(segs) - 1 and segs[k]["start"] < tt:
                k += 1
            window = " ".join(s["text"] for s in segs[max(0, k - 1):k + 2])
            if len(words(window)) >= 5:
                out.append({"pid": f"{r['videoId']}@{int(t)}{'j' if jitter else ''}", "vid": r["videoId"], "cue": k,
                            "true": float(segs[k]["start"]), "query": window})
            t += every
    return out


class BM25:
    """Okapi BM25 with an inverted index (same k1, b and idf as v1)."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(words(t)) for t in texts]
        self.lens = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t, f in c.items():
                self.post[t].append((i, f))
        n = len(texts)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def scores(self, query, restrict=None):
        acc = defaultdict(float)
        for t in words(query):
            if t not in self.post:
                continue
            idf = self.idf[t]
            for i, f in self.post[t]:
                if restrict is not None and i not in restrict:
                    continue
                acc[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return acc

    def top(self, query, restrict=None):
        acc = self.scores(query, restrict)
        if not acc:
            # v1 returns index 0 when nothing scores; keep that behaviour
            return min(restrict) if restrict else 0
        # ties: lowest index wins, as in v1 (strict > while scanning in order)
        best = max(acc.values())
        return min(i for i, s in acc.items() if s == best)


def cue_offsets(r):
    """Char offset of each cue inside r['text'] (cues joined by a space or a blank line).
    Returns (offsets, len(text), matched) where matched counts cues found in place."""
    segs, text = r["segments"], r.get("text") or ""
    offs, pos, matched = [], 0, 0
    for s in segs:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        ct = s["text"].strip()
        if text.startswith(ct, pos):
            matched += 1
            offs.append(pos)
            pos += len(ct)
        else:
            j = text.find(ct, pos)
            if j != -1 and j - pos < 200:
                matched += 1
                offs.append(j)
                pos = j + len(ct)
            else:
                offs.append(pos)
    return offs, len(text), matched


def actor_paragraphs(r):
    out = []
    for line in (r.get("markdown") or "").split("\n"):
        m = PARA.match(line.strip())
        if m:
            out.append({"start": float(m.group(3)), "text": m.group(4)})
    return out


def summarize(errs_signed, n, dur_list=None):
    """errs_signed: list of (cited - true) for right-video hits, None for wrong video."""
    fin = [e for e in errs_signed if e is not None]
    ab = [abs(e) for e in fin]
    w = lambda s: round(100 * sum(1 for e in ab if e <= s) / n, 1)
    # 'playable': cited at or up to 30 s before the moment (pressing play reaches it)
    play = round(100 * sum(1 for e in fin if -30 <= e <= 1) / n, 1)
    after = round(100 * sum(1 for e in fin if e > 1) / n, 1)
    return {
        "right_video_pct": round(100 * len(fin) / n, 1),
        "within_15s_pct": w(15), "within_30s_pct": w(30), "within_60s_pct": w(60),
        "median_abs_error_s": round(statistics.median(ab), 1) if ab else None,
        "playable_30s_pct": play,
        "lands_after_pct": after,
    }


def rng(seed=20261113):
    return random.Random(seed)
