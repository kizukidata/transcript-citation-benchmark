"""Passage builders ('arms') for the T5 study.

Each arm is a list of passages {vid, start, end, text, c0, c1}:
  start  the second a citation of this passage points at
  end    where the passage stops (next passage's start, or the video end)
  c0,c1  first and last caption cue the passage covers (for small-to-big)

Arms separate two things v1 changed together: passage size and where the second
comes from.
  charN_prop   r['text'] cut every N chars; second = offset / len * duration (v1 'naive500' at N=500)
  charN_true   the same pieces; second = start of the cue the piece begins inside
  cueN         whole cues packed up to N chars; second = first cue's start
  winT         cues grouped into fixed T-second windows (LangChain TranscriptFormat.CHUNKS style);
               second = first cue's start in the window
  para         the Actor's Markdown paragraphs, each with its own timestamp link
  chunk1500    the Actor's 1,500-char chunks with their start second (v1 'chunks1500')
  cue1         one passage per caption cue (ceiling: the finest grain the captions allow)
"""

import bisect

from common import actor_paragraphs, cue_offsets, duration

SIZES = [250, 500, 1000, 1500, 3000]
WINDOWS = [15, 30, 60, 120, 300]


def _cue_index_at_time(starts, t):
    return max(0, bisect.bisect_right(starts, t + 1e-9) - 1)


def build_arms(rows, sizes=SIZES, windows=WINDOWS):
    arms = {}

    def add(name, p):
        arms.setdefault(name, []).append(p)

    for r in rows:
        vid, segs, dur = r["videoId"], r["segments"], duration(r)
        starts = [float(s["start"]) for s in segs]
        text = r.get("text") or ""
        total = max(len(text), 1)
        offs, _, _ = cue_offsets(r)

        for n in sizes:
            pieces = list(range(0, len(text), n))
            true_starts = []
            for i in pieces:
                k = max(0, bisect.bisect_right(offs, i) - 1)
                true_starts.append((k, starts[k]))
            for j, i in enumerate(pieces):
                piece = text[i:i + n]
                k0 = true_starts[j][0]
                k1 = max(k0, bisect.bisect_right(offs, min(i + n, len(text)) - 1) - 1)
                add(f"char{n}_prop", {"vid": vid, "start": i / total * dur,
                                      "end": min(i + n, total) / total * dur, "text": piece, "c0": k0, "c1": k1})
                nxt = true_starts[j + 1][1] if j + 1 < len(pieces) else dur
                add(f"char{n}_true", {"vid": vid, "start": true_starts[j][1], "end": max(nxt, true_starts[j][1]),
                                      "text": piece, "c0": k0, "c1": k1})

        for n in sizes:
            cur, c0, size = [], 0, 0
            groups = []
            for k, s in enumerate(segs):
                t = s["text"].strip()
                if cur and size + len(t) + 1 > n:
                    groups.append((c0, k - 1, " ".join(cur)))
                    cur, size = [], 0
                if not cur:
                    c0 = k
                cur.append(t)
                size += len(t) + 1
            if cur:
                groups.append((c0, len(segs) - 1, " ".join(cur)))
            for j, (a, b, t) in enumerate(groups):
                end = starts[groups[j + 1][0]] if j + 1 < len(groups) else dur
                add(f"cue{n}", {"vid": vid, "start": starts[a], "end": end, "text": t, "c0": a, "c1": b})

        for w in windows:
            buckets = {}
            for k, s in enumerate(segs):
                buckets.setdefault(int(starts[k] // w), []).append(k)
            keys = sorted(buckets)
            for j, key in enumerate(keys):
                ks = buckets[key]
                end = starts[buckets[keys[j + 1]][0]] if j + 1 < len(keys) else dur
                add(f"win{w}", {"vid": vid, "start": starts[ks[0]], "end": end,
                                "text": " ".join(segs[k]["text"].strip() for k in ks), "c0": ks[0], "c1": ks[-1]})

        # lcT: a line-for-line copy of langchain_community 0.4.2
        # YoutubeLoader._get_transcript_chunks (TranscriptFormat.CHUNKS, default 120 s):
        # a cue whose END passes the limit starts a new chunk; the chunk's second is the
        # window boundary (0, T, 2T, ...), not the first cue's start.
        for w in windows:
            if w not in (30, 60, 120):
                continue
            chunks, cur, c_start, limit = [], [], 0, w
            for k, s in enumerate(segs):
                piece_end = float(s["start"]) + float(s.get("duration") or 0)
                if piece_end > limit:
                    if cur:
                        chunks.append((c_start, cur))
                    cur = []
                    c_start = limit
                    limit += w
                cur.append(k)
            if cur:
                chunks.append((c_start, cur))
            for j, (cs, ks) in enumerate(chunks):
                end = chunks[j + 1][0] if j + 1 < len(chunks) else dur
                add(f"lc{w}", {"vid": vid, "start": float(cs), "end": float(max(end, cs)),
                               "text": " ".join(segs[k]["text"].strip(" ") for k in ks), "c0": ks[0], "c1": ks[-1]})

        ps = actor_paragraphs(r)
        for j, p in enumerate(ps):
            end = ps[j + 1]["start"] if j + 1 < len(ps) else dur
            k0 = _cue_index_at_time(starts, p["start"])
            k1 = max(k0, _cue_index_at_time(starts, end - 0.01))
            add("para", {"vid": vid, "start": p["start"], "end": float(end), "text": p["text"], "c0": k0, "c1": k1})

        ch = r.get("chunks") or []
        for j, c in enumerate(ch):
            end = float(ch[j + 1]["start"]) if j + 1 < len(ch) else dur
            k0 = _cue_index_at_time(starts, float(c["start"]))
            k1 = max(k0, _cue_index_at_time(starts, end - 0.01))
            add("chunk1500", {"vid": vid, "start": float(c["start"]), "end": end, "text": c["text"], "c0": k0, "c1": k1})

        for k, s in enumerate(segs):
            end = starts[k + 1] if k + 1 < len(segs) else dur
            add("cue1", {"vid": vid, "start": starts[k], "end": end, "text": s["text"], "c0": k, "c1": k})

    return arms
