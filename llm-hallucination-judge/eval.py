# llm-hallucination-judge
# Lightweight LLM evaluation framework — hallucination detection WITHOUT API calls.
#
# How it works (pure Python, stdlib only — no torch, no API keys):
#   1. Extract claims: the answer is split into independent sentence-level claims.
#   2. Score each claim against the provided context using lexical/semantic proxies:
#        - IDF-weighted token overlap (rare words matter more than "the", "is" ...)
#        - numeric consistency (numbers/dates in the claim must appear in context)
#        - entity consistency (capitalized names in the claim must appear in context)
#        - negation mismatch detection ("not" flipping the meaning of the context)
#   3. Map the score to a verdict:  supported / uncertain / hallucinated.
#   4. Aggregate into document-level metrics (hallucination rate, confidence).
#
# Optional upgrade: if `transformers` is installed it auto-uses a cross-encoder NLI
# model (local, still 0 API calls) for entailment instead of pure lexical scoring.
#
# Usage:
#   python eval.py                     # runs the built-in demo dataset
#   python eval.py --data data.jsonl   # each line: {"context": ..., "answer": ...}
#   python eval.py --data data.jsonl --report report.md

import argparse
import json
import os
import re
import string
import sys
from collections import Counter

# ----------------------------------------------------------------------------- #
# Config: IDF-weighted lexical entailment + number/entity/negation checks
# ------------------------------------------------------------------------------ #
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "in", "on", "at", "to", "for", "and", "or", "it", "its", "this", "that",
    "these", "those", "as", "by", "with", "from", "which", "who", "has", "have",
    "had", "not", "no", "can", "could", "will", "would", "there", "their",
}
SUPPORTED_MIN = 0.60     # score >= this  -> supported
SUSPICIOUS_MIN = 0.35    # score >= this  -> uncertain, else hallucinated

VERDICT_EMOJI = {"supported": "\u2705 SUPPORTED",
                 "uncertain": "\u26a0\ufe0f  UNCERTAIN",
                 "hallucinated": "\u274c HALLUCINATED"}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*\d?|[A-Za-z]+|\d+(?:[.,]\d+)?%?")
_NUMBER = re.compile(r"\b\d[\d.,%]*\b")
Entity = re.compile(r"\b[A-Z][a-zA-Z0-9-]{1,}\b", re.UNICODE)

try:  # optional local cross-encoder NLI (no API!)
    from transformers import pipeline as _hf_pipeline
    _nli = _hf_pipeline("text-classification",
                        model="cross-encoder/nli-deberta-v3-xsmall",
                        truncation=True)
    print("[eval] using local cross-encoder NLI (no API calls)", file=sys.stderr)
    HAVE_NLI = True
except Exception:
    _nli = None
    HAVE_NLI = False
    print("[eval] using lexical/semantic heuristic judge (no API calls)",
          file=sys.stderr)


# ---------------------------------------------------------------- tokenizing #
def _tokens(text):
    toks = _TOKEN.findall(text.lower())
    return [t.strip(string.punctuation) for t in toks if t.strip(string.punctuation)]


def _content_tokens(text):
    return [t for t in _tokens(text) if t not in STOPWORDS]


def _split_claims(text):
    claims = [c.strip() for c in _SENT_SPLIT.split(text) if c.strip()]
    return claims or [text.strip()]


# ------------------------------------------------------------ scoring engine #
def _idf_scores(context):
    """Content tokens that appear once in the context get more weight."""
    counts = Counter(_content_tokens(context))
    max_c = max(counts.values()) if counts else 1
    return {tok: 1.0 / (1 + (c - 1) / max_c) for tok, c in counts.items()}


def claim_score(claim, context, idf):
    """Return (score in [0,1], reasons list)."""
    reasons, score = [], 0.0
    ctoks = set(_content_tokens(context))
    claim_toks = _content_tokens(claim)
    if not claim_toks:
        return 0.0, ["empty claim"]

    # 1. IDF-weighted lexical overlap
    matched, total = 0.0, 0.0
    for tok in claim_toks:
        w = idf.get(tok, 1.0)
        total += w
        if tok in ctoks:
            matched += w
            if idf.get(tok, 1.0) >= 0.55:
                reasons.append(f"feature match: '{tok}'")
    overlap = matched / total if total else 0.0
    score += 55 * overlap

    # 2. numbers must exist in context
    claim_nums, ctx_nums = _NUMBER.findall(claim), set(_NUMBER.findall(context))
    for n in claim_nums:
        n_clean = n.rstrip(".%,")
        if n_clean in ctx_nums or n in ctx_nums:
            score += 12
            reasons.append(f"number match: {n}")
        else:
            score -= 18
            reasons.append(f"number NOT in context: {n}")

    # 3. named entities must exist in context
    ctx_ents = {e.lower() for e in Entity.findall(context)}
    for e in Entity.findall(claim):
        if e.lower() in ctx_ents:
            score += 8
            reasons.append(f"entity match: {e}")
        elif e.lower() not in STOPWORDS:
            score -= 12
            reasons.append(f"entity NOT in context: {e}")

    # 4. negation consistency
    if ("not" in claim.lower() or "no " in claim.lower()) and \
            "not" not in context.lower():
        score -= 10
        reasons.append("negation mismatch")

    # 5. optional NLI stage
    if HAVE_NLI:
        try:
            r = _nli({"text": context, "text_pair": claim},
                     truncation=True)[0]
            if r["label"].lower() == "entailment":
                score += 8
                reasons.append(f"NLI: entailment ({r['score']:.2f})")
            elif r["label"].lower() == "contradiction":
                score -= 25
                reasons.append(f"NLI: contradiction ({r['score']:.2f})")
        except Exception:
            pass

    return max(0.0, min(1.0, score / 100.0)), reasons


def verdict(score):
    if score >= SUPPORTED_MIN:
        return "supported"
    if score >= SUSPICIOUS_MIN:
        return "uncertain"
    return "hallucinated"


# ------------------------------------------------------------- judge object  #
class HallucinationJudge:
    def __init__(self):
        pass

    def judge(self, context, answer):
        claims = _split_claims(answer)
        idf = _idf_scores(context)
        per_claim, scores = [], []
        for c in claims:
            s, reasons = claim_score(c, context, idf)
            scores.append(s)
            per_claim.append({
                "claim": c,
                "score": round(s, 3),
                "verdict": verdict(s),
                "evidence": reasons[:4],
                "reasons": reasons,
            })
        avg = sum(scores) / len(scores) if scores else 0.0
        n_hall = sum(1 for p in per_claim if p["verdict"] == "hallucinated")

        # distinct entities claimed that never appear in the context
        entity_misses = set()
        for p in per_claim:
            for ev in p["reasons"]:
                if ev.startswith("entity NOT in context: "):
                    entity_misses.add(ev.split(": ", 1)[1].lower())

        is_hallucinated = (
            (n_hall > 0 and avg < SUPPORTED_MIN)        # an unsupported claim
            or avg < SUSPICIOUS_MIN                     # overall too faithful
            or len(entity_misses) >= 2                  # ≥2 invented entities
        )
        return {
            "answer": answer,
            "is_hallucinated": is_hallucinated,
            "confidence": round(avg, 3),
            "hallucinated_claims": n_hall,
            "claims": per_claim,
        }


# ----------------------------------------------------------------- demo data #
DEMO = [
    {"context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
                "Mars in Paris, France. It is named after engineer Gustave Eiffel, "
                "whose company designed and built the tower from 1887 to 1889. It "
                "stands 330 meters tall and was the tallest man-made structure in "
                "the world for 41 years.",
     "answer": "The Eiffel Tower is located in Paris, France. It was designed by "
               "Gustave Eiffel and completed in 1889. It is 330 meters tall."},
    {"context": "Photosynthesis is the process used by plants, algae and some "
                "bacteria to convert light energy into chemical energy that is "
                "later released to fuel the organism's activities. Chlorophyll is "
                "the green pigment that absorbs light in photosynthesis.",
     "answer": "Photosynthesis converts light into chemical energy in plants. "
               "Hemoglobin is the pigment responsible for absorbing sunlight."},
    {"context": "The Large Hadron Collider is the world's largest and highest-"
                "energy particle accelerator. It was built by CERN between 1998 "
                "and 2008 and lies in a 27 kilometer tunnel near Geneva.",
     "answer": "The Large Hadron Collider is operated by NASA and was built in "
               "Russia. It sits inside a 27 kilometer tunnel near Geneva."},
    {"context": "PyTorch is an open-source machine learning library based on the "
                "Torch library, developed by Meta AI. It is widely used for deep "
                "learning research and production.",
     "answer": "PyTorch is an open-source deep learning library. It was developed "
               "by OpenAI and released in 1995."},
    {"context": "Vaccines train the immune system to recognize and fight pathogens "
                "without causing the disease. The first vaccine was developed by "
                "Edward Jenner in 1796 against smallpox.",
     "answer": "Vaccines help the immune system recognize pathogens. Edward Jenner "
               "developed the first vaccine against smallpox in 1796."},
]


# -------------------------------------------------------------------- report #
def evaluate(dataset, judge, report_path=None):
    results, lines = [], []
    n_hall, total = 0, len(dataset)

    lines.append("# LLM Hallucination Judge — evaluation report\n")
    lines.append("`No API calls used` — heuristic / local-NLI judge only.\n")
    lines.append("| # | answer (first 60 chars) | verdict | confidence | hallucinated claims |")
    lines.append("|---|---|---|---|---|")

    for i, ex in enumerate(dataset, 1):
        r = judge.judge(ex["context"], ex["answer"])
        n_hall += int(r["is_hallucinated"])
        ans = ex["answer"][:57].replace("|", "/") + ("..." if len(ex["answer"]) > 60 else "")
        tag = "HALLUCINATED" if r["is_hallucinated"] else "faithful"
        lines.append(f"| {i} | {ans} | {tag} | {r['confidence']} | {r['hallucinated_claims']}/{len(r['claims'])} |")
        results.append(r)

    lines.append("")
    lines.append(f"**Hallucination rate: {n_hall}/{total} = "
                 f"{n_hall / total:.0%}**\n")

    lines.append("## Per-claim detail\n")
    for i, (ex, r) in enumerate(zip(dataset, results), 1):
        lines.append(f"### Example {i}")
        lines.append(f"*Answer:* {r['answer']}\n")
        for c in r["claims"]:
            lines.append(f"- {VERDICT_EMOJI[c['verdict']]}  "
                         f"({c['score']})  —  \"{c['claim']}\"")
            for ev in c["evidence"]:
                lines.append(f"    - {ev}")
        lines.append("")

    report = "\n".join(lines)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nreport written -> {report_path}")
    else:
        print("\n" + "-" * 70 + "\n" + report + "-" * 70 + "\n")
    return results


# ---------------------------------------------------------------------- main #
def main():
    # Windows consoles default to cp1252 and crash on emoji output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(
        description="Lightweight LLM hallucination judge (no API calls)")
    p.add_argument("--data", type=str, default=None,
                   help="JSONL file with {\"context\":..., \"answer\":...} per line")
    p.add_argument("--report", type=str, default=None,
                   help="write a markdown report to this path instead of stdout")
    args = p.parse_args()

    if args.data:
        data = []
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        print(f"[eval] loaded {len(data)} examples from {args.data}")
    else:
        print("[eval] no --data given: running built-in demo dataset "
              "(5 QA pairs, 2 with planted hallucinations)\n")
        data = DEMO

    judge = HallucinationJudge()
    results = evaluate(data, judge, report_path=args.report)

    n_hall = sum(r["is_hallucinated"] for r in results)
    print(f"[eval] hallucination rate: {n_hall}/{len(results)}")


if __name__ == "__main__":
    main()
