"""Loading UFO instances from local JSONL files or the Hugging Face Hub.

A normalized instance is a dict with these keys:
    id, category, task, question_type,
    images       : list[str]  absolute image paths
    question     : str
    choices      : dict|None  {"A":..,"B":..,"C":..,"D":..} for mcq, None for open
    answer       : str        option letter (mcq) or reference text (open)
    text_cue     : str|None   ground-truth textual cue
    image_cue    : str|None   ground-truth visual cue (absolute path)
"""

import json
import os

from .config import QUESTION_TYPES


def _abs(path, root):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))


def _normalize(rec, root):
    images = rec.get("images") or []
    if isinstance(images, str):
        images = [images]
    images = [_abs(p, root) for p in images if p]

    cue = rec.get("image_cue")
    if isinstance(cue, list):
        cue = cue[0] if cue else None

    choices = rec.get("choices")
    if rec.get("question_type") == "open" or not isinstance(choices, dict):
        choices = None

    return {
        "id": rec.get("id"),
        "category": rec.get("category"),
        "task": rec.get("task"),
        "question_type": rec.get("question_type"),
        "images": images,
        "question": (rec.get("question") or "").strip(),
        "choices": choices,
        "answer": rec.get("answer"),
        "text_cue": rec.get("text_cue"),
        "image_cue": _abs(cue, root) if cue else None,
    }


def load_jsonl(path, image_root=None):
    """Load a UFO JSONL file. Image paths are resolved relative to image_root
    (defaults to the directory containing the JSONL file)."""
    root = image_root or os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(_normalize(json.loads(line), root))
    return out


def load_hf(repo_id="yzzyu/UFO", split="mcq", cache_dir=None):
    """Load a split from the Hugging Face Hub. Returns normalized instances
    with images materialized to a local cache directory so paths are usable.

    Requires `datasets`. The parquet build embeds images, so we write them to
    disk under cache_dir/images/<id>/.
    """
    from datasets import load_dataset

    cache_dir = cache_dir or os.path.join(os.getcwd(), "data_cache", split)
    img_dir = os.path.join(cache_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    ds = load_dataset(repo_id, split=split)
    out = []
    for rec in ds:
        sid = rec["id"]
        sample_dir = os.path.join(img_dir, sid)
        os.makedirs(sample_dir, exist_ok=True)

        img_paths = []
        for i, im in enumerate(rec.get("input_images") or []):
            p = os.path.join(sample_dir, f"input_{i + 1}.png")
            if not os.path.exists(p):
                im.save(p)
            img_paths.append(p)

        cue_path = None
        if rec.get("cue_image") is not None:
            cue_path = os.path.join(sample_dir, "cue.png")
            if not os.path.exists(cue_path):
                rec["cue_image"].save(cue_path)

        choices = None
        if rec["question_type"] == "mcq":
            choices = {k: rec.get(f"choice_{k.lower()}", "") for k in ("A", "B", "C", "D")}

        out.append({
            "id": sid,
            "category": rec["category"],
            "task": rec["task"],
            "question_type": rec["question_type"],
            "images": img_paths,
            "question": rec["question"],
            "choices": choices,
            "answer": rec["answer"],
            "text_cue": rec.get("text_cue") or None,
            "image_cue": cue_path,
        })
    return out


def filter_instances(instances, tasks=None, categories=None, question_type=None, limit=0):
    """Subset instances by task / category / question type, optionally limited."""
    out = []
    for it in instances:
        if question_type and it.get("question_type") != question_type:
            continue
        if tasks and it.get("task") not in tasks:
            continue
        if categories and it.get("category") not in categories:
            continue
        out.append(it)
    if limit and limit > 0:
        out = out[:limit]
    return out


def load_any(source, image_root=None, split=None):
    """Convenience loader.

    - If `source` is a local .jsonl file -> load_jsonl.
    - Otherwise treat `source` as an HF repo id -> load_hf (split required).
    """
    if source.endswith(".jsonl") and os.path.exists(source):
        return load_jsonl(source, image_root=image_root)
    if not split or split not in QUESTION_TYPES:
        raise ValueError(f"For HF source '{source}', pass split in {QUESTION_TYPES}.")
    return load_hf(source, split=split)
