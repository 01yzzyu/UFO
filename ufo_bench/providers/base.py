"""Provider interface shared by all platform backends.

A Provider abstracts a single model on a single platform and exposes two
capabilities used by the UFO inference engine:

    complete(text, image_paths, system, ...) -> (text, error)
        Multimodal text generation. Used for answering questions and for
        generating *textual* cues.

    generate_image(prompt, image_paths, save_path) -> (path, error)
        Generate a *visual* cue image. Returns the saved file path, or
        (None, error) if the model/platform does not support image generation.

Concrete providers override these. `supports_image_gen` advertises capability
so the engine can skip visual/joint protocols gracefully.
"""

import abc


class Provider(abc.ABC):
    #: whether this provider can synthesize images (visual cues)
    supports_image_gen = False

    def __init__(self, model_id, retries=5, timeout=120, **kwargs):
        self.model_id = model_id
        self.retries = retries
        self.timeout = timeout

    @abc.abstractmethod
    def complete(self, text, image_paths=None,
                 system="You are a helpful assistant.",
                 temperature=0.0, max_tokens=2048):
        """Return (text, error). error is None on success."""
        raise NotImplementedError

    def generate_image(self, prompt, image_paths=None, save_path=None):
        """Return (saved_path, error). Default: not supported."""
        return None, "image_generation_not_supported"
