"""UniPic1 (Skywork/Skywork-UniPic-1.5B) local adapter.

Faithful to the official scripts:
  understanding : https://github.com/SkyworkAI/UniPic/blob/main/UniPic-1/scripts/image2text.py
  text-to-image : https://github.com/SkyworkAI/UniPic/blob/main/UniPic-1/scripts/text2image.py

UniPic-1 is an mmengine-config model: it is built via `src.builder.BUILDER` from a
config file, then a checkpoint .bin is loaded. Provide both in models.yaml:
    config_path : configs/models/qwen2_5_1_5b_kl16_mar_h.py  (in the repo)
    checkpoint  : checkpoint/pytorch_model.bin               (downloaded weights)

Prerequisites:
    git clone https://github.com/SkyworkAI/UniPic && cd UniPic/UniPic-1
    conda create -n unipic python==3.10.14 && conda activate unipic
    pip install -r requirements.txt
    export PYTHONPATH=./:$PYTHONPATH            # so `src` and `configs` import
    huggingface-cli download Skywork/Skywork-UniPic-1.5B --local-dir checkpoint

models.yaml entry:
    - {name: UniPic1, group: unified, provider: local, local_adapter: unipic1,
       model_path: Skywork/Skywork-UniPic-1.5B,
       config_path: configs/models/qwen2_5_1_5b_kl16_mar_h.py,
       checkpoint: checkpoint/pytorch_model.bin, image_size: 1024}
"""

import numpy as np
from PIL import Image

from .base_local import LocalProvider


def _expand2square(pil_img, bg=(127, 127, 127)):
    w, h = pil_img.size
    if w == h:
        return pil_img
    side = max(w, h)
    result = Image.new(pil_img.mode, (side, side), bg)
    result.paste(pil_img, (0, (side - h) // 2) if w > h else ((side - w) // 2, 0))
    return result


class UniPic1Adapter(LocalProvider):
    def _load(self):
        import torch
        from einops import rearrange  # noqa: F401  (used in generation)
        from mmengine.config import Config
        from src.builder import BUILDER

        self.torch = torch
        self.image_size = int(self.extra.get("image_size", 1024))
        config_path = self.extra.get("config_path")
        checkpoint = self.extra.get("checkpoint")
        if not config_path or not checkpoint:
            raise ValueError(
                "UniPic1 needs config_path and checkpoint in models.yaml.")

        config = Config.fromfile(config_path)
        self.model = BUILDER.build(config.model).eval().cuda()
        self.model = self.model.to(self.model.dtype)
        state = torch.load(checkpoint)
        self.model.load_state_dict(state, strict=False)

        # register the <image> special token (per image2text.py)
        self.model.tokenizer.add_special_tokens(
            {'additional_special_tokens': ["<image>"]})
        self.image_token_idx = self.model.tokenizer.encode(
            "<image>", add_special_tokens=False)[-1]

    def _understand(self, prompt, pil_images, max_tokens=1024, temperature=0.0):
        torch = self.torch
        from einops import rearrange
        from ...imutil import concat_images
        model = self.model
        # UniPic-1's official image2text takes a single image; merge if multiple.
        image = concat_images(pil_images)
        image = _expand2square(image.convert("RGB"), (127, 127, 127))
        image = image.resize((self.image_size, self.image_size))
        image = torch.from_numpy(np.array(image)).to(dtype=model.dtype, device=model.device)
        image = rearrange(image, 'h w c -> c h w')[None]
        image = 2 * (image / 255) - 1

        text = model.prompt_template['INSTRUCTION'].format(input="<image>\n" + prompt)
        image_length = (self.image_size // 16) ** 2 + 64
        text = text.replace('<image>', '<image>' * image_length)
        input_ids = model.tokenizer.encode(
            text, add_special_tokens=True, return_tensors='pt').cuda()
        with torch.no_grad():
            _, z_enc = model.extract_visual_feature(model.encode(image))
            inputs_embeds = z_enc.new_zeros(*input_ids.shape, model.llm.config.hidden_size)
            inputs_embeds[input_ids == self.image_token_idx] = z_enc.flatten(0, 1)
            inputs_embeds[input_ids != self.image_token_idx] = \
                model.llm.get_input_embeddings()(input_ids[input_ids != self.image_token_idx])
            out = model.llm.generate(
                inputs_embeds=inputs_embeds, use_cache=True, do_sample=False,
                max_new_tokens=max_tokens, eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id)
        return model.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _generate(self, prompt, pil_images, cfg=3.0, cfg_prompt="Generate an image.",
                  temperature=1.0, cfg_schedule="constant", num_iter=32):
        torch = self.torch
        from einops import rearrange
        model = self.model
        full_prompt = f"Generate an image: {prompt}"
        class_info = model.prepare_text_conditions(full_prompt, cfg_prompt)
        input_ids = class_info['input_ids']
        attention_mask = class_info['attention_mask']
        # input_ids has 2 rows: conditional + unconditional
        if cfg == 1.0:
            input_ids = input_ids[:1]
            attention_mask = attention_mask[:1]
        m = n = self.image_size // 16
        with torch.no_grad():
            samples = model.sample(
                input_ids=input_ids, attention_mask=attention_mask,
                num_iter=num_iter, cfg=cfg, cfg_schedule=cfg_schedule,
                temperature=temperature, progress=False, image_shape=(m, n))
        samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=1, n=1)
        arr = torch.clamp(127.5 * samples + 128.0, 0, 255).to(
            "cpu", dtype=torch.uint8).numpy()
        return Image.fromarray(arr)
