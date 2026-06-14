"""Provider backends for calling models on different platforms.

Different models live on different platforms:
  - openai      : OpenAI-compatible endpoints (OpenRouter, DashScope, etc.)
                  used for GPT / Qwen / Gemma and most chat-VLMs.
  - gemini      : Google Gemini-style endpoint that returns text AND can return
                  an inline generated image in one call.
  - fal         : fal.ai, which hosts unified / image-generation models such as
                  Bagel, OmniGen2, etc. that are NOT on OpenRouter.

Every provider implements the same interface (see base.Provider) so the
inference engine is platform-agnostic.
"""

from .base import Provider

_PROVIDERS = {"openai", "openrouter", "gemini", "fal", "local"}


def build_provider(provider_name, model_id, **kwargs):
    """Factory: instantiate a provider by name.

    Concrete providers are imported lazily so that optional dependencies
    (e.g. fal-client, torch, the UFM repos) are only required when that
    provider is actually used.

    For provider='local', pass `local_adapter` (e.g. 'bagel') and
    `model_path`; `model_id` is treated as the model_path if model_path is
    not given.
    """
    key = (provider_name or "openai").lower()
    if key in ("openai", "openrouter"):
        from .openai_compatible import OpenAIProvider
        return OpenAIProvider(model_id, **kwargs)
    if key == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(model_id, **kwargs)
    if key == "fal":
        from .fal import FalProvider
        return FalProvider(model_id, **kwargs)
    if key == "local":
        from .local import build_local_provider
        adapter = kwargs.pop("local_adapter", None)
        model_path = kwargs.pop("model_path", None) or model_id
        if not adapter:
            raise ValueError("provider='local' requires 'local_adapter'.")
        return build_local_provider(adapter, model_path, **kwargs)
    raise ValueError(
        f"Unknown provider '{provider_name}'. Available: {sorted(_PROVIDERS)}"
    )


__all__ = ["Provider", "build_provider"]
