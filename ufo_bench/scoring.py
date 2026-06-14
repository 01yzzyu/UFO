"""Answer scoring.

MCQ  : robust option-letter extraction, then exact match to the gold letter.
Open : an LLM judge decides factual correctness (binary), matching the paper.

Each scored instance gets, per protocol/model:
    score_<protocol>_<model_tag>  in {0.0, 1.0}
"""

import re

from .prompts import OPEN_ANSWER_JUDGE


def extract_option_letter(text):
    """Pull an A/B/C/D choice out of a free-form model answer."""
    if not text:
        return None
    t = text.strip()
    # 1) leading standalone letter: "A", "A.", "A)", "(A)"
    m = re.match(r"^\s*\(?([A-D])\)?[\.\):]?\s*$", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2) "answer is B", "Option: C"
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([A-D])\)?\b",
                  t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 3) first standalone letter token (e.g. "B" or lowercase "b"),
    #    but NOT a letter embedded in a word like "blah".
    m = re.search(r"\b([A-Da-d])\b", t)
    if m:
        return m.group(1).upper()
    return None


def score_mcq(pred_text, gold_letter):
    pred = extract_option_letter(pred_text)
    if pred is None or not gold_letter:
        return 0.0
    return 1.0 if pred == str(gold_letter).strip().upper() else 0.0


def score_open_with_judge(judge, question, gold, pred_text):
    """Binary correctness via LLM judge provider. Returns (score, raw)."""
    if not pred_text:
        return 0.0, ""
    prompt = OPEN_ANSWER_JUDGE.format(question=question or "N/A",
                                      gt=gold or "N/A", pred=pred_text)
    out, _ = judge.complete(prompt, system="You are a strict evaluator.",
                            max_tokens=8)
    raw = out or ""
    m = re.search(r"[01]", raw)
    return (float(m.group(0)) if m else 0.0), raw


def score_item(item, model_tag, protocols, judge_client=None):
    """Score all protocol predictions for one instance, in place."""
    item = dict(item)
    qtype = item.get("question_type")
    gold = item.get("answer")
    question = item.get("question", "")

    for protocol in protocols:
        pred_key = f"pred_{protocol}_{model_tag}"
        score_key = f"score_{protocol}_{model_tag}"
        if score_key in item:
            continue
        pred = item.get(pred_key)
        if pred is None:
            continue
        if qtype == "mcq":
            item[score_key] = score_mcq(pred, gold)
        else:
            if judge_client is None:
                raise ValueError("Open-ended scoring requires a judge_client.")
            sc, raw = score_open_with_judge(judge_client, question, gold, pred)
            item[score_key] = sc
            item[f"judge_raw_{protocol}_{model_tag}"] = raw
    return item
