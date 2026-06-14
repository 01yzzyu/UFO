"""UniWorld-V1 (LanguageBind/UniWorld-V1) local adapter.

Faithful to the official CLI:
    https://github.com/PKU-YuanGroup/UniWorld/blob/main/UniWorld-V1/univa/serve/cli.py

UniWorld-V1 is a composite system:
  - understanding : UnivaQwen2p5VL backbone (chat -> model.generate)
  - generation    : the backbone produces `denoise_embeds`, combined with T5
                    text embeds, fed into a FLUX pipeline; SigLIP encodes
                    reference images for high-res semantic control.

Prerequisites:
    git clone https://github.com/PKU-YuanGroup/UniWorld && cd UniWorld/UniWorld-V1
    conda create -n univa python=3.10 -y && conda activate univa
    pip install -r requirements.txt
    export PYTHONPATH=/path/to/UniWorld/UniWorld-V1:$PYTHONPATH   # provides `univa`
    # weights: LanguageBind/UniWorld-V1 (model_path, contains task_head_final.pt),
    #          a FLUX repo (flux_path), and a SigLIP repo (siglip_path)

models.yaml entry:
    - {name: UniWorld-V1, group: unified, provider: local, local_adapter: uniworld_v1,
       model_path: LanguageBind/UniWorld-V1,
       flux_path: /path/to/flux, siglip_path: /path/to/siglip,
       height: 1024, width: 1024, num_inference_steps: 28, guidance_scale: 3.5}
"""

from .base_local import LocalProvider


class UniWorldV1Adapter(LocalProvider):
    def _load(self):
        import torch
        from transformers import AutoProcessor, SiglipImageProcessor, SiglipVisionModel
        from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import (
            UnivaQwen2p5VLForConditionalGeneration)
        from univa.utils.flux_pipeline import FluxPipeline

        self.torch = torch
        device = "cuda"
        self.device = device
        flux_path = self.extra.get("flux_path")
        siglip_path = self.extra.get("siglip_path")
        if not flux_path:
            raise ValueError("UniWorld-V1 needs flux_path in models.yaml.")

        self.model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").to(device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, min_pixels=448 * 448, max_pixels=448 * 448)

        self.pipe = FluxPipeline.from_pretrained(
            flux_path, transformer=self.model.denoise_tower.denoiser,
            torch_dtype=torch.bfloat16).to(device)
        self.tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]
        self.text_encoders = [self.pipe.text_encoder, self.pipe.text_encoder_2]

        self.siglip_processor = self.siglip_model = None
        if siglip_path:
            self.siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
            self.siglip_model = SiglipVisionModel.from_pretrained(
                siglip_path, torch_dtype=torch.bfloat16).to(device)

        self.height = int(self.extra.get("height", 1024))
        self.width = int(self.extra.get("width", 1024))
        self.num_inference_steps = int(self.extra.get("num_inference_steps", 28))
        self.guidance_scale = float(self.extra.get("guidance_scale", 3.5))

    def _build_inputs(self, prompt, pil_images):
        from qwen_vl_utils import process_vision_info
        content = [{"type": "text", "text": prompt}]
        for im in pil_images:
            content.append({"type": "image", "image": im,
                            "min_pixels": 448 * 448, "max_pixels": 448 * 448})
        conversation = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])  # drop system
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(text=[chat_text], images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.device)
        return inputs

    def _understand(self, prompt, pil_images, max_tokens=128, temperature=0.0):
        torch = self.torch
        inputs = self._build_inputs(prompt, list(pil_images))
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        trimmed = [out[len(inp):] for inp, out in
                   zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()

    def _siglip_hidden(self, pil_images):
        torch = self.torch
        if self.siglip_model is None or not pil_images:
            return None
        vals = []
        for im in pil_images:
            v = self.siglip_processor.preprocess(
                images=im.convert("RGB"), do_resize=True, return_tensors="pt",
                do_convert_rgb=True).pixel_values
            vals.append(v)
        vals = torch.concat(vals).to(self.siglip_model.device)
        return self.siglip_model(vals).last_hidden_state

    def _generate(self, prompt, pil_images):
        torch = self.torch
        from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
        inputs = self._build_inputs(prompt, list(pil_images))
        siglip_hidden = self._siglip_hidden(list(pil_images))
        with torch.no_grad():
            lvlm_embeds = self.model(
                inputs.input_ids,
                pixel_values=getattr(inputs, "pixel_values", None),
                attention_mask=inputs.attention_mask,
                image_grid_thw=getattr(inputs, "image_grid_thw", None),
                siglip_hidden_states=siglip_hidden,
                output_type="denoise_embeds")
            t5_embeds, pooled = encode_prompt(
                self.text_encoders, self.tokenizers, prompt, 256, self.device, 1)
            input_embeds = torch.concat([t5_embeds, lvlm_embeds], dim=1)
            image = self.pipe(
                prompt_embeds=input_embeds, pooled_prompt_embeds=pooled,
                height=self.height, width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).images[0]
        return image
