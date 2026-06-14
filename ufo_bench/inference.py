"""Inference engine: runs the four UFO reasoning protocols for one model.

Platform-agnostic: it talks to one or two `Provider` backends
(see ufo_bench.providers):

    answer_provider : answers questions and generates the *textual* cue
    image_provider  : generates the *visual* cue (defaults to answer_provider)

This lets a unified model (e.g. Bagel) answer via its chat/understanding
endpoint while synthesizing visual cues via fal.ai, or a VLM answer via
OpenRouter while a Gemini/fal image model produces the visual cue.

Protocols:
    direct  : images only
    textual : images + generated text cue
    visual  : images + generated visual cue image
    joint   : images + text cue + visual cue image
"""

import os

from .config import (
    PROTOCOLS,
    PROTOCOL_NEEDS_IMAGE_CUE,
    PROTOCOL_NEEDS_TEXT_CUE,
)
from .prompts import (
    ANSWER_SYSTEM,
    IMAGE_CUE_GEN,
    TEXT_CUE_GEN,
    build_answer_prompt,
)


def model_tag(model_id):
    return model_id.replace("/", "_").replace(":", "_").replace("\\", "_")


# Backwards-compatible alias used by scripts and other modules.
_tag = model_tag


def _safe(x):
    return str(x).replace("/", "_").replace(" ", "_")


class Reasoner:
    def __init__(self, answer_provider, image_provider=None, protocols=None,
                 model_name=None, output_img_dir="outputs/generated_cues",
                 answer_max_tokens=512, cue_max_tokens=400):
        self.ap = answer_provider
        self.ip = image_provider or answer_provider
        self.protocols = protocols or list(PROTOCOLS)
        # tag used in result keys; stable per model under test
        self.tag = model_tag(model_name or answer_provider.model_id)
        self.output_img_dir = output_img_dir
        self.answer_max_tokens = answer_max_tokens
        self.cue_max_tokens = cue_max_tokens
        if any(PROTOCOL_NEEDS_IMAGE_CUE[p] for p in self.protocols):
            os.makedirs(output_img_dir, exist_ok=True)

    # -- cue generation ----------------------------------------------------
    def _gen_text_cue(self, item):
        key = f"text_cue_generated_{self.tag}"
        if item.get(key):
            return item[key]
        out, _ = self.ap.complete(
            TEXT_CUE_GEN.format(question=item["question"]),
            image_paths=item["images"], max_tokens=self.cue_max_tokens,
        )
        item[key] = out or ""
        return item[key]

    def _gen_image_cue(self, item, index):
        key = f"image_cue_generated_{self.tag}"
        if item.get(key) and os.path.exists(item[key]):
            return item[key]
        if not self.ip.supports_image_gen:
            item[key] = ""
            return ""
        save = os.path.join(
            self.output_img_dir, f"{_safe(item.get('id', index))}_{self.tag}.png")
        if os.path.exists(save):
            item[key] = save
            return save
        prompt = IMAGE_CUE_GEN.format(question=item["question"])
        path, err = self.ip.generate_image(prompt, image_paths=item["images"],
                                           save_path=save)
        item[key] = path or ""
        if err:
            item[f"image_cue_err_{self.tag}"] = err
        return item[key]

    # -- answering ---------------------------------------------------------
    def _answer(self, item, protocol, text_cue, image_cue_path):
        use_text = text_cue if PROTOCOL_NEEDS_TEXT_CUE[protocol] else None
        use_img = bool(image_cue_path) and PROTOCOL_NEEDS_IMAGE_CUE[protocol]
        imgs = list(item["images"])
        if use_img:
            imgs = imgs + [image_cue_path]
        prompt = build_answer_prompt(
            item["question"], item.get("choices"),
            text_cue=use_text, has_image_cue=use_img,
        )
        return self.ap.complete(prompt, image_paths=imgs, system=ANSWER_SYSTEM,
                                max_tokens=self.answer_max_tokens)

    # -- main --------------------------------------------------------------
    def process(self, item, index=0):
        item = dict(item)
        need_text = any(PROTOCOL_NEEDS_TEXT_CUE[p] for p in self.protocols)
        need_img = any(PROTOCOL_NEEDS_IMAGE_CUE[p] for p in self.protocols)

        text_cue = self._gen_text_cue(item) if need_text else None
        image_cue = self._gen_image_cue(item, index) if need_img else None

        for protocol in self.protocols:
            out_key = f"pred_{protocol}_{self.tag}"
            if item.get(out_key):
                continue
            ans, err = self._answer(item, protocol, text_cue, image_cue)
            item[out_key] = ans or ""
            if err:
                item[f"err_{protocol}_{self.tag}"] = err
        return item
