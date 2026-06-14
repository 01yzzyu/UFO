"""OpenAI-compatible provider (OpenRouter / OpenAI / DashScope / SiliconFlow ...).

Used for GPT, Qwen-VL, Gemma and most chat VLMs. Text generation via
chat.completions; image generation via images.generate (only if the endpoint
and model support it, e.g. gpt-image-1 / dall-e on OpenAI).
"""

import os
import time

from openai import OpenAI

from .base import Provider
from ..imutil import encode_data_url, save_b64_image


def _resolve_key(explicit, env_names):
    for name in env_names:
        v = explicit if explicit else os.getenv(name)
        if v:
            return v
    return None


class OpenAIProvider(Provider):
    supports_image_gen = True  # depends on the model; generate_image guards it

    def __init__(self, model_id, base_url="https://openrouter.ai/api/v1",
                 api_key=None, retries=5, timeout=120,
                 enable_reasoning=False, image_gen_model=None, **kwargs):
        super().__init__(model_id, retries=retries, timeout=timeout)
        key = _resolve_key(api_key, ["OPENROUTER_API_KEY", "OPENAI_API_KEY"])
        if not key:
            raise RuntimeError(
                "No API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
            )
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.enable_reasoning = enable_reasoning
        self.image_gen_model = image_gen_model

    def complete(self, text, image_paths=None,
                 system="You are a helpful assistant.",
                 temperature=0.0, max_tokens=2048):
        if image_paths:
            content = [{"type": "text", "text": text}]
            for p in image_paths:
                url, err = encode_data_url(p)
                if url:
                    content.append({"type": "image_url",
                                    "image_url": {"url": url}})
            user_msg = {"role": "user", "content": content}
        else:
            user_msg = {"role": "user", "content": text}
        messages = [{"role": "system", "content": system}, user_msg]

        extra = None if self.enable_reasoning else {"reasoning": {"enabled": False}}
        last_err = ""
        for attempt in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                    extra_body=extra, timeout=self.timeout,
                )
                out = (resp.choices[0].message.content or "").strip()
                if out:
                    return out, None
                last_err = "empty_response"
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < self.retries - 1:
                time.sleep(min(2 ** attempt, 8))
        return None, last_err

    def generate_image(self, prompt, image_paths=None, save_path=None):
        model = self.image_gen_model or self.model_id
        try:
            resp = self.client.images.generate(
                model=model, prompt=prompt, n=1, size="1024x1024",
                response_format="b64_json",
            )
            b64 = resp.data[0].b64_json
            if save_path and save_b64_image(b64, save_path):
                return save_path, None
            return None, "save_failed"
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:200]}"
