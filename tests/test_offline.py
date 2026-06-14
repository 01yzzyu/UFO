"""Offline unit tests — no GPU, no API keys, no network required.

Run with:  pytest    (from the repo root, after `pip install -e ".[test]"`)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from ufo_bench.scoring import extract_option_letter, score_mcq
from ufo_bench.imutil import concat_images
from ufo_bench.config import ALL_TASKS, PROTOCOLS, TASK_TO_CATEGORY
from ufo_bench import aggregate
from ufo_bench.cli import detect_model_tag
from ufo_bench.registry import load_registry, resolve_models


# ---- scoring -------------------------------------------------------------- #
def test_extract_option_letter():
    cases = {
        "A": "A", "The answer is B.": "B", "(C)": "C", "D) microwave": "D",
        "Answer: C": "C", "b": "B", "blah": None, "": None,
    }
    for text, expected in cases.items():
        assert extract_option_letter(text) == expected, text


def test_score_mcq():
    assert score_mcq("The answer is B", "B") == 1.0
    assert score_mcq("A", "C") == 0.0
    assert score_mcq("", "C") == 0.0


# ---- imutil --------------------------------------------------------------- #
def test_concat_images():
    a = Image.new("RGB", (100, 80), (255, 0, 0))
    b = Image.new("RGB", (60, 120), (0, 255, 0))
    assert concat_images([]) is None
    assert concat_images([a]).size == (100, 80)
    merged = concat_images([a, b])  # resized to common height 80
    assert merged.height == 80 and merged.width > 100


# ---- config --------------------------------------------------------------- #
def test_taxonomy():
    assert len(ALL_TASKS) == 10
    assert set(PROTOCOLS) == {"direct", "textual", "visual", "joint"}
    assert all(t in TASK_TO_CATEGORY for t in ALL_TASKS)


# ---- aggregate + detect_model_tag ---------------------------------------- #
def _scored_fixture(tag="M"):
    return [
        {"task": "Chemical", "category": "state_determination",
         f"score_direct_{tag}": 1.0, f"score_joint_{tag}": 0.0},
        {"task": "Chemical", "category": "state_determination",
         f"score_direct_{tag}": 0.0, f"score_joint_{tag}": 1.0},
        {"task": "Jigsaw", "category": "state_reconstruction",
         f"score_direct_{tag}": 1.0, f"score_joint_{tag}": 1.0},
    ]


def test_detect_model_tag():
    assert detect_model_tag(_scored_fixture("GPT-5.1")) == "GPT-5.1"
    assert detect_model_tag([{"foo": 1}]) is None


def test_aggregate():
    data = _scored_fixture("M")
    bt = aggregate.accuracy_by_task(data, "M", ["direct", "joint"])
    assert bt["direct"]["Chemical"] == (0.5, 2)
    assert bt["joint"]["Jigsaw"] == (1.0, 1)
    bc = aggregate.accuracy_by_category(data, "M", ["direct", "joint"])
    assert bc["direct"]["state_determination"] == (0.5, 2)
    md = aggregate.summary_markdown(
        {"M": aggregate.overall_accuracy(data, "M", ["direct", "joint"])},
        ["M"], ["direct", "joint"])
    assert "Model" in md and "M" in md


# ---- registry resolve_models --------------------------------------------- #
def test_resolve_models():
    cfg = load_registry()
    names = {m["name"] for m in cfg["models"]}
    assert "GPT-5.1" in names

    one = resolve_models(cfg, ["GPT-5.1"])
    assert len(one) == 1 and one[0]["name"] == "GPT-5.1"

    everything = resolve_models(cfg, ["all"])
    assert len(everything) == len(cfg["models"])

    unified = resolve_models(cfg, ["group:unified"])
    assert unified and all(m["group"] == "unified" for m in unified)

    adhoc = resolve_models(cfg, ["some/random-model"])
    assert adhoc[0]["provider"] == "openai"
