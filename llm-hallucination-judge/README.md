# llm-hallucination-judge

Lightweight LLM evaluation framework — hallucination detection without API calls.

Evaluates whether an LLM's answer is faithful to a given context, with **zero API calls** and **zero GPU requirements**. Perfect for large-scale eval jobs where you can't afford per-call pricing — the judge runs entirely locally.

## How it works

1. **Claim extraction** — the answer is split into sentence-level claims.
2. **Faithfulness scoring** per claim against the context:
   - IDF-weighted token overlap (rare/informative words count more than stopwords)
   - Numeric consistency — numbers/dates in the claim must exist in the context
   - Entity consistency — named entities in the claim must exist in the context
   - Negation-mismatch detection
3. **Optional upgrade** — if `transformers` is installed, a local cross-encoder NLI model (`nli-deberta-v3-xsmall`) refines the scores. Still no API calls.
4. **Verdicts** — `supported` / `uncertain` / `hallucinated`, aggregated into a document-level *hallucination rate* and confidence.

## Usage

```bash
# built-in demo: 5 QA pairs, 3 with planted hallucinations
python eval.py

# your own data (each line of the JSONL: {"context": ..., "answer": ...})
python eval.py --data data.jsonl

# write a markdown report instead of printing
python eval.py --data data.jsonl --report report.md
```

Example output line:
```
- ❌ HALLUCINATED (0.12) — "Hemoglobin is the pigment responsible for absorbing sunlight."
      - feature match: 'pigment'
      - entity NOT in context: 'Hemoglobin'
```

The pure-Python mode needs **nothing at all** (stdlib only). With `transformers` installed it automatically uses the local NLI cross-encoder for better precision.

## Why this design

- **No API calls** → free, fast, privacy-safe, reproducible, and usable on huge eval sets.
- Deterministic and fully explainable — every verdict comes with evidence bullets.
- Designed to slot into any eval pipeline: feed it `{context, answer}` pairs produced by any LLM (including fully local ones) and get a hallucination-rate metric back.

## Extending

- Swap the heuristic in `claim_score()` for any local scoring model.
- Add an independent-source check in `_idf_scores` (e.g., TF-IDF over a small knowledge corpus).
- CLI already supports JSONL in / markdown report out, so it pipelines with your existing eval scripts.
