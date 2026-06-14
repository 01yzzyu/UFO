"""Bagel (ByteDance-Seed/BAGEL-7B-MoT) local adapter.

Wraps the official `InterleaveInferencer` API. Based on the official repo:
    https://github.com/ByteDance-Seed/BAGEL  (app.py, inferencer.py)

Prerequisites:
    git clone https://github.com/ByteDance-Seed/BAGEL && cd BAGEL
    pip install -r requirements.txt
    pip install flash_attn==2.5.8 --no-build-isolation
    # download weights to e.g. models/BAGEL-7B-MoT
    export PYTHONPATH=/path/to/BAGEL:$PYTHONPATH   # so `modeling`, `inferencer` import

models.yaml entry:
    - {name: Bagel, group: unified, provider: local, local_adapter: bagel,
       model_path: /path/to/models/BAGEL-7B-MoT}

Understanding (answering) uses interleave_inference(..., understanding_output=True).
Visual-cue generation uses the default (image) output.
"""

import os

from .base_local import LocalProvider


class BagelAdapter(LocalProvider):
    def _load(self):
        import torch
        from accelerate import (
            infer_auto_device_map, load_checkpoint_and_dispatch,
            init_empty_weights,
        )
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer

        mp = self.model_path
        llm_config = Qwen2Config.from_json_file(os.path.join(mp, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(mp, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vae_model, vae_config = load_ae(local_path=os.path.join(mp, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True, visual_und=True, llm_config=llm_config,
            vit_config=vit_config, vae_config=vae_config,
            vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
            latent_patch_size=2, max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(mp)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        device_map = infer_auto_device_map(
            model,
            max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        same_device = [
            "language_model.model.embed_tokens", "time_embedder",
            "latent_pos_embed", "vae2llm", "llm2vae", "connector",
            "vit_pos_embed",
        ]
        first = device_map.get(same_device[0], "cuda:0")
        for k in same_device:
            device_map[k] = device_map.get(k, first) if torch.cuda.device_count() > 1 else first

        model = load_checkpoint_and_dispatch(
            model, checkpoint=os.path.join(mp, "ema.safetensors"),
            device_map=device_map, offload_buffers=True, offload_folder="offload",
            dtype=torch.bfloat16, force_hooks=True,
        ).eval()

        self.inferencer = InterleaveInferencer(
            model=model, vae_model=vae_model, tokenizer=tokenizer,
            vae_transform=vae_transform, vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

    def _understand(self, prompt, pil_images, max_tokens=2048, temperature=0.0):
        inputs = list(pil_images) + [prompt]
        out = self.inferencer.interleave_inference(
            inputs, understanding_output=True,
            max_think_token_n=max_tokens, do_sample=temperature > 0,
            text_temperature=max(temperature, 0.01),
        )
        for x in out:
            if isinstance(x, str):
                return x
        return ""

    def _generate(self, prompt, pil_images):
        from PIL import Image
        inputs = list(pil_images) + [prompt]
        out = self.inferencer.interleave_inference(
            inputs, understanding_output=False,
        )
        for x in out:
            if isinstance(x, Image.Image):
                return x
        return None
