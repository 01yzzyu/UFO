"""Emu3 (BAAI/Emu3-Chat + BAAI/Emu3-Gen) local adapter.

Based on the official model cards / repo: https://github.com/baaivision/Emu3
Understanding uses Emu3-Chat (mode='U'); image generation uses Emu3-Gen (mode='G')
with classifier-free guidance, per the official quickstart.

Prerequisites:
    git clone https://github.com/baaivision/Emu3   # provides emu3.mllm.processing_emu3
    export PYTHONPATH=/path/to/Emu3:$PYTHONPATH
    # weights: BAAI/Emu3-Chat (understanding), BAAI/Emu3-Gen (generation),
    #          BAAI/Emu3-VisionTokenizer (shared VQ tokenizer)

models.yaml entry:
    - {name: EMU3, group: unified, provider: local, local_adapter: emu3,
       model_path: BAAI/Emu3-Chat, gen_model_path: BAAI/Emu3-Gen,
       vq_path: BAAI/Emu3-VisionTokenizer}
"""

from .base_local import LocalProvider


class Emu3Adapter(LocalProvider):
    def _load(self):
        import torch
        from transformers import (
            AutoModel, AutoModelForCausalLM, AutoImageProcessor, AutoTokenizer,
        )
        from emu3.mllm.processing_emu3 import Emu3Processor

        self.torch = torch
        vq_path = self.extra.get("vq_path", "BAAI/Emu3-VisionTokenizer")
        self.gen_path = self.extra.get("gen_model_path", "BAAI/Emu3-Gen")

        # understanding model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map="cuda:0", torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="left")
        image_processor = AutoImageProcessor.from_pretrained(
            vq_path, trust_remote_code=True)
        image_tokenizer = AutoModel.from_pretrained(
            vq_path, device_map="cuda:0", trust_remote_code=True).eval()
        self.processor = Emu3Processor(image_processor, image_tokenizer,
                                       self.tokenizer)
        self._gen_model = None  # lazily loaded on first image generation

    def _understand(self, prompt, pil_images, max_tokens=1024, temperature=0.0):
        from transformers.generation.configuration_utils import GenerationConfig
        from ...imutil import concat_images

        # Emu3-Chat's official 'U' mode takes a single image; merge if multiple
        # (needed for multi-image tasks and visual/joint protocols).
        image = concat_images(pil_images) if pil_images else None
        inputs = self.processor(text=prompt, image=image, mode="U",
                                return_tensors="pt", padding="longest")
        cfg = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
        )
        out = self.model.generate(
            inputs.input_ids.to("cuda:0"), cfg,
            attention_mask=inputs.attention_mask.to("cuda:0"),
        )
        out = out[:, inputs.input_ids.shape[-1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def _ensure_gen_model(self):
        if self._gen_model is None:
            from transformers import AutoModelForCausalLM
            self._gen_model = AutoModelForCausalLM.from_pretrained(
                self.gen_path, device_map="cuda:0",
                torch_dtype=self.torch.bfloat16,
                attn_implementation="flash_attention_2", trust_remote_code=True,
            ).eval()
        return self._gen_model

    def _generate(self, prompt, pil_images):
        # Mirrors https://github.com/baaivision/Emu3/blob/main/image_generation.py
        from PIL import Image
        from transformers.generation.configuration_utils import GenerationConfig
        from transformers.generation import (
            LogitsProcessorList, PrefixConstrainedLogitsProcessor,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
        )

        model = self._ensure_gen_model()
        positive = " masterpiece, film grained, best quality."
        negative = ("lowres, bad anatomy, bad hands, text, error, missing fingers, "
                    "extra digit, fewer digits, cropped, worst quality, low quality, "
                    "normal quality, jpeg artifacts, signature, watermark, username, blurry.")
        classifier_free_guidance = 3.0

        kwargs = dict(mode="G", ratio=["1:1"], image_area=model.config.image_area,
                      return_tensors="pt", padding="longest")
        pos_inputs = self.processor(text=[prompt + positive], **kwargs)
        neg_inputs = self.processor(text=[negative], **kwargs)

        cfg = GenerationConfig(
            use_cache=True, eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id, max_new_tokens=40960,
            do_sample=True, top_k=2048,
        )
        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                classifier_free_guidance, model,
                unconditional_ids=neg_inputs.input_ids.to("cuda:0")),
            PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
        ])
        outputs = model.generate(
            pos_inputs.input_ids.to("cuda:0"), cfg,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to("cuda:0"))
        for out in outputs:
            for im in self.processor.decode(out):
                if isinstance(im, Image.Image):
                    return im
        return None
