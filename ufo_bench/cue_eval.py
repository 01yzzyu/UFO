"""Intermediate cue-quality evaluation.

Judges whether a *generated* cue matches the *ground-truth* cue:
  - text cue : LLM judge compares generated text cue vs GT text cue (binary)
  - visual cue: VLM judge compares generated image cue vs GT image cue (binary)

This supports the paper's "process evaluation" (evidential coupling): a correct
answer should be backed by a state-consistent cue. Optionally computes agreement
(accuracy + Cohen's kappa) against human labels if present in the data.
"""

import os
import re

from .prompts import TEXT_CUE_JUDGE, VISUAL_CUE_JUDGE


def _parse_binary(text):
    if not text:
        return 0.0
    m = re.search(r"[01]", text)
    return float(m.group(0)) if m else 0.0


def judge_text_cue(judge, question, gt_cue, gen_cue):
    if not gen_cue:
        return 0.0, ""
    prompt = TEXT_CUE_JUDGE.format(question=question or "N/A",
                                   gt=gt_cue or "N/A", pred=gen_cue)
    out, _ = judge.complete(prompt, system="You are a strict evaluator.",
                            max_tokens=8)
    return _parse_binary(out), (out or "")


def judge_visual_cue(judge, question, gt_img, gen_img):
    """Two images go to the judge in order: GT cue (image 1), candidate (image 2)."""
    if not (gt_img and gen_img and os.path.exists(gt_img) and os.path.exists(gen_img)):
        return 0.0, "missing_image"
    prompt = (
        "Image 1 is the Ground Truth Cue. Image 2 is the Candidate Cue.\n\n"
        + VISUAL_CUE_JUDGE.format(question=question or "N/A")
    )
    out, _ = judge.complete(prompt, image_paths=[gt_img, gen_img],
                            system="Follow instructions strictly.", max_tokens=8)
    return _parse_binary(out), (out or "")


def evaluate_cues(item, tag, judge, targets=("text", "visual")):
    """Score generated cues against ground-truth cues for one instance."""
    item = dict(item)
    q = item.get("question", "")

    if "text" in targets:
        gen = item.get(f"text_cue_generated_{tag}")
        if gen:
            sc, raw = judge_text_cue(judge, q, item.get("text_cue"), gen)
            item[f"cue_text_score_{tag}"] = sc
            item[f"cue_text_raw_{tag}"] = raw

    if "visual" in targets:
        gen = item.get(f"image_cue_generated_{tag}")
        gt = item.get("image_cue")
        if gen and gt:
            sc, raw = judge_visual_cue(judge, q, gt, gen)
            item[f"cue_visual_score_{tag}"] = sc
            item[f"cue_visual_raw_{tag}"] = raw
    return item


def cohen_kappa(preds, golds):
    """Cohen's kappa for binary labels (no sklearn dependency)."""
    if not preds or len(preds) != len(golds):
        return 0.0
    n = len(preds)
    p = [1 if x else 0 for x in preds]
    g = [1 if x else 0 for x in golds]
    po = sum(1 for a, b in zip(p, g) if a == b) / n
    pp, gp = sum(p) / n, sum(g) / n
    pe = pp * gp + (1 - pp) * (1 - gp)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)
