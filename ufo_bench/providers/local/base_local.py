"""Base class for local (GPU-deployed) UFM adapters.

A local adapter loads a model once (lazily, on first use) and exposes:
    complete(text, image_paths, ...)        -> (answer_text, error)
    generate_image(prompt, image_paths, save_path) -> (path, error)

Subclasses implement `_load()` (build the underlying model/inferencer) and
`_understand(...)` / `_generate(...)` using the model's official API.
"""

import os

from PIL import Image

from ..base import Provider


class LocalProvider(Provider):
    supports_image_gen = True

    def __init__(self, model_path, device="cuda", dtype="bfloat16",
                 retries=1, timeout=0, **kwargs):
        super().__init__(model_id=model_path, retries=retries, timeout=timeout)
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.extra = kwargs
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    def _ensure_loaded(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self):
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _open_images(image_paths):
        imgs = []
        for p in image_paths or []:
            if p and os.path.exists(p):
                imgs.append(Image.open(p).convert("RGB"))
        return imgs

    @staticmethod
    def _save(pil_image, save_path):
        if pil_image is None or not save_path:
            return None
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        pil_image.save(save_path)
        return save_path

    # -- Provider interface ------------------------------------------------
    def complete(self, text, image_paths=None,
                 system="You are a helpful assistant.",
                 temperature=0.0, max_tokens=2048):
        try:
            self._ensure_loaded()
            imgs = self._open_images(image_paths)
            prompt = f"{system}\n\n{text}" if system else text
            out = self._understand(prompt, imgs, max_tokens=max_tokens,
                                   temperature=temperature)
            return (out or "").strip(), None
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:200]}"

    def generate_image(self, prompt, image_paths=None, save_path=None):
        try:
            self._ensure_loaded()
            imgs = self._open_images(image_paths)
            pil = self._generate(prompt, imgs)
            saved = self._save(pil, save_path)
            return (saved, None) if saved else (None, "no_image_generated")
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:200]}"

    # -- to implement in subclasses ---------------------------------------
    def _understand(self, prompt, pil_images, max_tokens=2048, temperature=0.0):
        """Return answer text given a prompt and PIL images."""
        raise NotImplementedError

    def _generate(self, prompt, pil_images):
        """Return a PIL.Image visual cue (or None if unsupported)."""
        raise NotImplementedError
