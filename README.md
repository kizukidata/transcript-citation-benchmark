# Transcript citation benchmark

When a RAG answer links to a second in a YouTube video, how far from the right moment does that second land, and what decides it?

This repository measures it for 14 public videos (1,322 minutes of English talks and lectures) and many ways of cutting a transcript into timestamped passages: fixed character splits, cues packed to a character budget, fixed time windows, a line-for-line copy of LangChain's `YoutubeLoader` (`TranscriptFormat.CHUNKS`, `langchain_community` 0.4.2), transcript-tool paragraphs and chunks, and single caption cues. Each splitting rule is run with the second taken from the real cue start and, for character splits, with the second estimated from the character offset, so passage length and timestamp source can be told apart.

Write-up: ARTICLE_URL

## Main results

A link counts as good when the cited second is from 30 s before the right moment up to 1 s after it, so pressing play reaches the moment ("playable" in the result files; `within_30s` is the two-sided version).

1. If the cited second is the passage start and retrieval returns the passage that contains the moment, the share of good links is `sum(L * min(1, 30/L)) / sum(L)` over passage lengths L. With verbatim queries (BM25, 1,257 probes at random seconds, right video 98.4% to 100%), 19 of the 20 passage rules that do not estimate the second from a character offset match this within -2.8 to +1.1 points (`out/09_decompose.md`, `bm25_verb`; single-cue passages left out because each query also contains the next cue). LangChain's default 120 s chunks: 25.1% predicted, 24.7% measured. The probes are spread evenly over time by construction, so this checks the passage builders and the retriever, not how real questions are spread.
2. The exception is LangChain 30 s chunks (99.9% predicted, 88.3% measured). LangChain's chunk second is the window boundary, and a cue that runs past a boundary goes into the next chunk, so 10.9% of verbatim probes with 30-second chunks are cited after the moment.
3. With 322 rewritten questions (written by an LLM), most misses happen before the timestamp: the top-1 passage is from the wrong video in about 4 of 10 cases and from the wrong part of the right video in another 3 to 4 (`out/09_decompose.md`). When the right passage comes back, the formula roughly holds.
4. Top k, rewritten questions (`out/10_topk_*.json`): if each passage carries one start second, the best start among the top 5 is a good link for 46.6% (BM25) and 42.9% (dense) of questions with 30-second windows, and 15.2% and 12.1% with LangChain's default. If each passage stores every cue's start second, the ceiling is whether the right passage is handed over at all. Compared at the same seconds of transcript handed to the LLM (`out/11_budget.md`), 120-second LangChain chunks and 30-second windows are level with BM25 at 600 s (+1.6 points, 95% interval -3.4 to +6.3 over videos), and 30-second windows are ahead with the dense retriever (13.7 points, 7.9 to 20.2). Compared at the same k instead, long passages look better only because they hand over more transcript. Picking the cue inside one 120-second window with BM25 gives 13.7% good links against a ceiling of 31.1% (`out/06`, `s2b_win120_to_cue1`).
5. Probes placed exactly on the minute shift clock-aligned passages by -8.2 to +48.6 points (`out/01` vs `out/02`, `fig/fig2_grid_artifact.png`).

Every setting, with video-level bootstrap 95% intervals, the share of probes that retrieved the right video, and the share within 30 s given the right video: [`out/08_tables.md`](out/08_tables.md).

## Files

| Path | What |
|---|---|
| `scripts/common.py` | data loading, probes, BM25, metrics |
| `scripts/arms.py` | the passage builders |
| `scripts/s01_run.py` | runs passages × probes × retriever, writes a summary JSON and per-probe JSONL |
| `scripts/s02_sample.py` | the probe sample shown to the question writer (writes transcript text to `out/paraphrase_sample.jsonl`, which is not in this repository) |
| `scripts/s03_drift.py` | per-video drift of the offset estimate |
| `scripts/s04_tables.py` | `out/08_tables.md` and the two figures |
| `scripts/s05_decompose.py` | splits top-1 results into wrong video / wrong passage / right passage (`out/09`) |
| `scripts/s06_topk.py` | top-k: right passage and best citation among the top 1, 3, 5, 10, 20, 40 (`out/10`, per-probe ranks in `out/10_*.ranks.jsonl`) |
| `scripts/s07_budget.py` | the same comparison at equal seconds of transcript handed over, with video-level bootstrap intervals (`out/11`) |
| `out/queries_paraphrase.jsonl` | the 322 rewritten questions, keyed by probe ID |
| `out/*.json`, `out/*.probes.jsonl` | summaries and per-probe results (cited second, right second, right video; no transcript text) |
| `measure-input.json` | the transcript tool input that produced the dataset |

## Reproduce it

1. Get the transcripts. The dataset was produced by one run of the [`kizukidata/youtube-transcripts`](https://apify.com/kizukidata/youtube-transcripts) Apify Actor with `measure-input.json` (`output: "both"`, so every row has `segments` with cue start times, `markdown` paragraphs and `chunks`). Save the dataset items as `data.json` in the repository root. Any source works if it gives the same fields per video: `videoId`, `durationSeconds`, `captionKind`, `text`, `segments` (`start`, `duration`, `text`), `markdown`, `chunks`.
2. Run the measurements from the repository root. BM25 needs Python 3.10+ and nothing else. The dense runs need `pip install fastembed numpy` and cache embeddings in `emb_cache/`.

```bash
python3 scripts/s01_run.py --out out/01_bm25_verbatim.json --per-probe out/01_bm25_verbatim.probes.jsonl
python3 scripts/s01_run.py --jitter --out out/02_bm25_verbatim_jitter.json --per-probe out/02_bm25_verbatim_jitter.probes.jsonl
python3 scripts/s01_run.py --jitter --caption manual --out out/04_bm25_jitter_manual.json
python3 scripts/s01_run.py --jitter --caption auto --out out/04_bm25_jitter_auto.json
python3 scripts/s01_run.py --jitter --queries out/queries_paraphrase.jsonl --out out/06_bm25_paraphrase.json --per-probe out/06_bm25_paraphrase.probes.jsonl
ARMS=char500_prop,char500_true,char1500_prop,char1500_true,cue250,cue500,cue1000,cue1500,cue3000,win30,win60,win120,lc30,lc60,lc120,para,chunk1500,cue1
python3 scripts/s01_run.py --retriever dense --jitter --arms $ARMS --out out/03_dense_verbatim_jitter.json --per-probe out/03_dense_verbatim_jitter.probes.jsonl
python3 scripts/s01_run.py --retriever dense --jitter --queries out/queries_paraphrase.jsonl --arms $ARMS --out out/07_dense_paraphrase.json --per-probe out/07_dense_paraphrase.probes.jsonl
python3 scripts/s03_drift.py
python3 scripts/s05_decompose.py bm25_verb out/02_bm25_verbatim_jitter.probes.jsonl dense_verb out/03_dense_verbatim_jitter.probes.jsonl bm25_para out/06_bm25_paraphrase.probes.jsonl dense_para out/07_dense_paraphrase.probes.jsonl
TK=cue250,cue500,win30,lc30,para,win60,lc60,chunk1500,win120,lc120,cue1
python3 scripts/s06_topk.py --retriever bm25 --queries out/queries_paraphrase.jsonl --arms $TK --out out/10_topk_bm25_paraphrase.json
python3 scripts/s06_topk.py --retriever dense --queries out/queries_paraphrase.jsonl --arms $TK --out out/10_topk_dense_paraphrase.json
python3 scripts/s07_budget.py
python3 scripts/s04_tables.py
```

The probe times come from a fixed seed, so the same `data.json` gives the same probes and the same numbers. A new run of the transcript tool can differ slightly if a channel edits its captions. The transcripts are not in this repository; they belong to the people who made the videos.

## How it works

- Probes. Starting at 1:00, one probe per 60-second step, stopping a minute before the end. With `--jitter` the probe time moves to a seeded random second inside its step. The probe is the cue at that time; the query is that cue and one cue on each side (at least five content words). The right answer is the probe cue's start second.
- Index. All passages of all 14 videos go into one index per setting, so a query can return the wrong video, which counts as a miss. Top-1 in `s01_run.py`; top 1, 3, 5 and 10 in `s06_topk.py`.
- Metrics. Playable (cited from 30 s before the right second up to 1 s after); within 15 / 30 / 60 s either side; right video; within 30 s given the right video; cited before the moment (up to 30 s early) or after it. 95% intervals resample videos, not probes.
- `lcT` reproduces `langchain_community` 0.4.2 `YoutubeLoader._get_transcript_chunks`: a cue whose end passes the limit starts a new chunk, and the chunk's second is the window boundary (0, T, 2T, ...), not its first cue's start.
- `s2b_X_to_Y` retrieves with passages X, then cites the best passage Y inside the retrieved span.

## What it does not measure

- 14 videos, English only, all technical talks and lectures; 8 are talks by one speaker, and all auto-generated tracks are his long videos, so caption type, length and speaker overlap.
- Top-1 retrieval with BM25 and one small embedding model. No reranker, no LLM reading several passages.
- The rewritten questions come from one LLM, instructed not to copy any three-word run from the target cues; it saw the title and about 30 s of context on each side. They take about 25 probes per video, while the verbatim probes follow video length.
- The wrong-video rate depends on how much these videos overlap in topic (8% to 100% per video for 30-second windows with BM25).
- One annotator, the probe cue, decides the right moment.

## Who made this

I maintain the `youtube-transcripts` Actor used to get the transcripts. The scripts only read its output; they fetch nothing.

## Licence

MIT for the code and the result files. See `LICENSE`.
