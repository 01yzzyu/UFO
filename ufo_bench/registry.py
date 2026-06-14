"""Load the model registry (configs/models.yaml) and build providers."""

import os

import yaml

from .providers import build_provider


def load_registry(path=None):
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "configs", "models.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("base_url", "https://openrouter.ai/api/v1")
    cfg.setdefault("judge_model", "openai/gpt-5.1")
    cfg.setdefault("judge_provider", "openai")
    cfg.setdefault("models", [])
    return cfg


def resolve_models(cfg, names=None):
    """Return model dicts selected by `names`.

    `names` accepts a mix of:
      - a display name or id from configs/models.yaml (e.g. "GPT-5.1")
      - "all"                 -> every model in the registry
      - "group:<g>"           -> every model whose group == <g>
                                 (proprietary | open_source | unified)
      - an unknown string     -> treated as an ad-hoc OpenAI-compatible model id
    If `names` is empty, all registry models are returned.
    """
    models = cfg["models"]
    if not names:
        return models

    selected, seen = [], set()

    def _add(m):
        key = m.get("name") or m.get("id")
        if key not in seen:
            seen.add(key)
            selected.append(m)

    known = {m["name"] for m in models} | {m.get("id") for m in models}
    for n in names:
        if n == "all":
            for m in models:
                _add(m)
        elif n.startswith("group:"):
            g = n.split(":", 1)[1]
            for m in models:
                if m.get("group") == g:
                    _add(m)
        elif n in known:
            for m in models:
                if m["name"] == n or m.get("id") == n:
                    _add(m)
        else:  # ad-hoc model id
            _add({"name": n, "id": n, "group": "custom", "provider": "openai"})
    return selected


def _provider_kwargs(entry, cfg, api_key):
    """Collect provider-construction kwargs from a model entry."""
    prov = (entry.get("provider") or "openai").lower()
    kw = {"api_key": api_key}
    if prov in ("openai", "openrouter"):
        kw["base_url"] = entry.get("base_url", cfg.get("base_url"))
    if prov == "gemini" and entry.get("host"):
        kw["host"] = entry["host"]
    if prov == "fal":
        for k in ("fal_prompt_arg", "fal_image_arg", "fal_multi_image"):
            if k in entry:
                kw[k] = entry[k]
    if prov == "local":
        kw.pop("api_key", None)  # local models need no API key
        # Forward every adapter-specific field from the model entry (e.g.
        # config_path, checkpoint, image_size, und_mode, gen_mode, lmm_path,
        # gen_model_path, vq_path, flux_path, siglip_path, device, dtype).
        reserved = {"name", "id", "group", "provider", "image_provider", "image_id"}
        for k, v in entry.items():
            if k not in reserved:
                kw[k] = v
    return prov, kw


def build_model_providers(entry, cfg, api_key=None):
    """Return (answer_provider, image_provider) for a model entry.

    image_provider defaults to the answer provider unless the entry specifies
    a separate `image_provider`/`image_id` (e.g. answer on OpenRouter, generate
    the visual cue on fal/gemini).
    """
    prov, kw = _provider_kwargs(entry, cfg, api_key)
    model_ref = entry.get("id") or entry.get("model_path") or entry["name"]
    answer = build_provider(prov, model_ref, **kw)

    if entry.get("image_provider") or entry.get("image_id"):
        img_entry = dict(entry)
        img_entry["provider"] = entry.get("image_provider", prov)
        iprov, ikw = _provider_kwargs(img_entry, cfg, api_key)
        image = build_provider(iprov, entry.get("image_id", model_ref), **ikw)
    else:
        image = answer
    return answer, image


def build_judge(cfg, judge_model=None, api_key=None):
    """Build the judge provider (OpenAI-compatible by default)."""
    model = judge_model or cfg.get("judge_model", "openai/gpt-5.1")
    prov = cfg.get("judge_provider", "openai")
    kw = {"api_key": api_key}
    if prov in ("openai", "openrouter"):
        kw["base_url"] = cfg.get("base_url")
    return build_provider(prov, model, **kw)
