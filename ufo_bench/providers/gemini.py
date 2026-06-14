"""Gemini-style provider.

Calls a Gemini `generateContent` endpoint (official Google host by default, or a
proxy host) that can return text AND an inline generated image in one response.
This is the path for models like `gemini-*-image` that natively synthesize
images (no separate image endpoint needed).

API key: GEMINI_API_KEY (or GOOGLE_API_KEY).
"""

import http.client
import json
import os
import time

from .base import Provider
from ..imutil import encode_b64, save_b64_image


def _resolve_key(explicit):
    return (explicit or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY"))


class GeminiProvider(Provider):
    supports_image_gen = True

    def __init__(self, model_id, host="generativelanguage.googleapis.com",
                 api_key=None, retries=5, timeout=120, **kwargs):
        super().__init__(model_id, retries=retries, timeout=timeout)
        self.host = host
        self.api_key = _resolve_key(api_key)
        if not self.api_key:
            raise RuntimeError("No API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

    def _image_parts(self, image_paths):
        parts = []
        for p in image_paths or []:
            b64, mime, err = encode_b64(p)
            if b64:
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        return parts

    def _call(self, parts, want_image=False):
        """Return (text, image_b64, error)."""
        conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)
        gen_cfg = {"temperature": 0.0, "maxOutputTokens": 8192}
        if want_image:
            gen_cfg["responseModalities"] = ["TEXT", "IMAGE"]
        payload = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": gen_cfg,
            "safetySettings": [
                {"category": c, "threshold": "BLOCK_NONE"} for c in (
                    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT")
            ],
        })
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        try:
            conn.request("POST",
                         f"/v1beta/models/{self.model_id}:generateContent",
                         payload, headers)
            data = conn.getresponse().read()
            obj = json.loads(data.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return None, None, f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            conn.close()

        if "error" in obj:
            return None, None, str(obj["error"].get("message", "api_error"))[:200]
        cands = obj.get("candidates") or []
        if not cands:
            return None, None, "no_candidates"
        text_out, img_out = None, None
        for part in cands[0].get("content", {}).get("parts", []):
            if "text" in part and text_out is None:
                text_out = part["text"].strip()
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and img_out is None:
                img_out = inline.get("data")
        return text_out, img_out, None

    def complete(self, text, image_paths=None,
                 system="You are a helpful assistant.",
                 temperature=0.0, max_tokens=2048):
        # Gemini has no separate system role here; prepend it to the prompt.
        prompt = f"{system}\n\n{text}" if system else text
        parts = self._image_parts(image_paths) + [{"text": prompt}]
        last_err = ""
        for attempt in range(self.retries):
            txt, _img, err = self._call(parts, want_image=False)
            if txt:
                return txt, None
            last_err = err or "empty_response"
            if attempt < self.retries - 1:
                time.sleep(min(2 ** attempt, 8))
        return None, last_err

    def generate_image(self, prompt, image_paths=None, save_path=None):
        parts = self._image_parts(image_paths) + [{"text": prompt}]
        last_err = ""
        for attempt in range(self.retries):
            _txt, img_b64, err = self._call(parts, want_image=True)
            if img_b64 and save_path and save_b64_image(img_b64, save_path):
                return save_path, None
            last_err = err or "no_image_returned"
            if attempt < self.retries - 1:
                time.sleep(min(2 ** attempt, 8))
        return None, last_err
