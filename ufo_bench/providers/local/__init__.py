"""Local (GPU) adapters for Unified Foundation Models (UFMs).

These models (Bagel, Emu3, Janus-Pro, OmniGen2, ...) are NOT served via a chat
API. They must be deployed locally on a GPU from their official repositories.
Each adapter wraps a single model's official inference API behind the common
`Provider` interface (complete / generate_image).

Usage prerequisites (per model):
  1. Clone the model's official repo and `pip install` its requirements.
  2. Download the model weights (HuggingFace).
  3. Make the repo importable (add to PYTHONPATH) and point the adapter at the
     weights via `model_path` in configs/models.yaml.

Adapters are imported lazily so that heavy deps (torch, the repos) are only
required for the model actually being run.
"""

# adapter name -> "module:ClassName" (lazy import target)
LOCAL_ADAPTERS = {
    # Faithful drafts based on official repos (untested without GPU + weights).
    "bagel": "bagel:BagelAdapter",
    "janus_pro": "janus_pro:JanusProAdapter",
    "emu3": "emu3:Emu3Adapter",
    "omnigen2": "omnigen2:OmniGen2Adapter",
    "ovis_u1": "ovis_u1:OvisU1Adapter",
    "unipic2": "unipic2:UniPic2Adapter",
    # Scaffolded stubs to complete from each official repo.
    "uniworld_v1": "uniworld_v1:UniWorldV1Adapter",
    "unicot": "unicot:UniCoTAdapter",
    "omni_r1": "omni_r1:OmniR1Adapter",
    "unipic1": "unipic1:UniPic1Adapter",
}


def build_local_provider(adapter, model_path, **kwargs):
    """Instantiate a local UFM adapter by name."""
    import importlib

    if adapter not in LOCAL_ADAPTERS:
        raise ValueError(
            f"Unknown local adapter '{adapter}'. "
            f"Available: {sorted(LOCAL_ADAPTERS)}"
        )
    mod_name, cls_name = LOCAL_ADAPTERS[adapter].split(":")
    module = importlib.import_module(f".{mod_name}", __package__)
    cls = getattr(module, cls_name)
    return cls(model_path=model_path, **kwargs)
