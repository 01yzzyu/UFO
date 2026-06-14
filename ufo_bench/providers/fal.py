"""fal.ai provider.

fal.ai hosts unified / image-generation models that are NOT on OpenRouter, such
as Bagel and OmniGen2. These are the "UFM" models in the paper's main table.

This provider primarily supports `generate_image` (visual cue synthesis): it
uploads the input images, calls the fal model, and downloads the result image.

Some fal models also return text; `complete` tries to extract a text field from
the response and otherwise reports that the model is image-only (in which case
answering should be routed to a chat provider via models.yaml).

Requirements:
  pip install fal-client
  export FAL_KEY=...            # fal reads this automatically

Argument schemas differ across fal models, so the input-image key and the
prompt key are configurable via models.yaml (fal_image_arg / fal_prompt_arg).
"""

import os

from .base import Provider
from ..imutil import download_to


class FalProvider(Provider):
    supports_image_gen = True

    def __init__(self, model_id, retries=5, timeout=300,
                 fal_prompt_arg="prompt", fal_image_arg="image_url",
                 fal_multi_image=False, api_key=None, **kwargs):
        super().__init__(model_id, retries=retries, timeout=timeout)
        # fal_client reads FAL_KEY from the environment; allow explicit override.
        if api_key:
            os.environ["FAL_KEY"] = api_key
        if not os.getenv("FAL_KEY"):
            raise RuntimeError("No fal API key. Set FAL_KEY in your environment.")
        try:
            import fal_client  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "fal-client is not installed. Run: pip install fal-client"
            ) from e
        self.prompt_arg = fal_prompt_arg
        self.image_arg = fal_image_arg
        self.multi_image = fal_multi_image

    def _upload(self, image_paths):
        import fal_client
        urls = []
        for p in image_paths or []:
            if p and os.path.exists(p):
                urls.append(fal_client.upload_file(p))
        return urls

    def _build_args(self, prompt, image_urls):
        args = {self.prompt_arg: prompt}
        if image_urls:
            if self.multi_image:
                args[self.image_arg] = image_urls
            else:
                args[self.image_arg] = image_urls[0]
        return args

    def _run(self, prompt, image_paths):
        """Call the fal model; return (result_dict, error)."""
        import fal_client
        try:
            urls = self._upload(image_paths)
            args = self._build_args(prompt, urls)
            result = fal_client.subscribe(self.model_id, arguments=args,
                                          with_logs=False)
            return result, None
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:200]}"

    @staticmethod
    def _first_image_url(result):
        imgs = result.get("images") or result.get("image") or []
        if isinstance(imgs, dict):
            return imgs.get("url")
        if isinstance(imgs, list) and imgs:
            first = imgs[0]
            return first.get("url") if isinstance(first, dict) else first
        return None

    @staticmethod
    def _text(result):
        for k in ("text", "output", "answer", "description", "response"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def complete(self, text, image_paths=None,
                 system="You are a helpful assistant.",
                 temperature=0.0, max_tokens=2048):
        result, err = self._run(text, image_paths)
        if err:
            return None, err
        txt = self._text(result)
        if txt:
            return txt, None
        # Image-only fal model: route answering to a chat provider in models.yaml.
        return None, "fal_model_returned_no_text:image_only_model"

    def generate_image(self, prompt, image_paths=None, save_path=None):
        result, err = self._run(prompt, image_paths)
        if err:
            return None, err
        url = self._first_image_url(result)
        if url and save_path and download_to(url, save_path, timeout=self.timeout):
            return save_path, None
        return None, "no_image_in_fal_result"
