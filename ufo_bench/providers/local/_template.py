"""Template for adding a new local UFM adapter.

Copy this file to `<model>.py`, implement the three methods from the model's
official repo, then register it in `local/__init__.py`:

    LOCAL_ADAPTERS["<name>"] = "<module>:<ClassName>"

and add a models.yaml entry:

    - {name: <Name>, group: unified, provider: local,
       local_adapter: <name>, model_path: <hf_or_local_path>}

The remaining paper UFMs that still need an adapter (with their official repos):
    Ovis-U1        : https://github.com/AIDC-AI/Ovis-U1
    UniPic1        : https://github.com/SkyworkAI/UniPic
    UniPic2-Metaquery : https://github.com/SkyworkAI/UniPic   (UniPic2)
    Omni-R1        : https://github.com/MiniMax-AI / official Omni-R1 repo
    UniCoT         : https://github.com/Fr0zenCrane/UniCoT
    UniWorld-V1    : https://github.com/PKU-YuanGroup/UniWorld-V1

Each exposes understanding (VQA) and image-generation entry points; wire those
into `_understand` and `_generate` below.
"""

from .base_local import LocalProvider


class TemplateAdapter(LocalProvider):
    def _load(self):
        # Load the model + processor from the official repo here, e.g.:
        #   import torch
        #   from <repo> import <Model>, <Processor>
        #   self.model = <Model>.from_pretrained(self.model_path, ...).cuda().eval()
        #   self.processor = <Processor>.from_pretrained(self.model_path)
        raise NotImplementedError("Implement _load() from the official repo.")

    def _understand(self, prompt, pil_images, max_tokens=512, temperature=0.0):
        # Run the model's VQA / understanding path and return answer text.
        raise NotImplementedError("Implement _understand() from the official repo.")

    def _generate(self, prompt, pil_images):
        # Run the model's image-generation path and return a PIL.Image.
        # If the model cannot generate images, set supports_image_gen = False
        # at the class level and remove this method.
        raise NotImplementedError("Implement _generate() from the official repo.")
