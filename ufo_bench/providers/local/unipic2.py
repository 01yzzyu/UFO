"""UniPic2-Metaquery (Skywork/UniPic2-Metaquery-9B) local adapter.

Based on the official model card / repo: https://github.com/SkyworkAI/UniPic (UniPic-2).
Built on Qwen2.5-VL (understanding/conditioning) + SD3.5 Kontext (generation).
Understanding uses the Qwen2.5-VL backbone; generation/editing use the custom
StableDiffusion3KontextPipeline with a meta-query conditioner. Requires ~40GB VRAM.

Prerequisites:
    git clone https://github.com/SkyworkAI/UniPic && cd UniPic-2
    pip install -r requirements.txt
    export PYTHONPATH=/path/to/UniPic-2:$PYTHONPATH   # provides `unipicv2`
    # weights: Skywork/UniPic2-Metaquery-9B  (+ Qwen/Qwen2.5-VL-7B-Instruct)

models.yaml entry:
    - {name: UniPic2-Metaquery, group: unified, provider: local,
       local_adapter: unipic2, model_path: Skywork/UniPic2-Metaquery-9B,
       lmm_path: Qwen/Qwen2.5-VL-7B-Instruct}
"""

from .base_local import LocalProvider

NEG_PROMPT = (
    "blurry, low quality, low resolution, distorted, deformed, broken content, "
    "missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, "
    "compression artifacts, bad composition, wrong proportion, incomplete editing, "
    "unfinished, unedited areas."
)


class UniPic2Adapter(LocalProvider):
    def _load(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
        from unipicv2.pipeline_stable_diffusion_3_kontext import (
            StableDiffusion3KontextPipeline)
        from unipicv2.transformer_sd3_kontext import SD3Transformer2DKontextModel
        from unipicv2.stable_diffusion_3_conditioner import (
            StableDiffusion3Conditioner)

        self.torch = torch
        mp = self.model_path
        lmm_path = self.extra.get("lmm_path", "Qwen/Qwen2.5-VL-7B-Instruct")

        transformer = SD3Transformer2DKontextModel.from_pretrained(
            mp, subfolder="transformer", torch_dtype=torch.bfloat16).cuda()
        vae = AutoencoderKL.from_pretrained(
            mp, subfolder="vae", torch_dtype=torch.bfloat16).cuda()
        self.lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            lmm_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2").cuda()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(lmm_path)
        self.conditioner = StableDiffusion3Conditioner.from_pretrained(
            mp, subfolder="conditioner", torch_dtype=torch.bfloat16).cuda()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            mp, subfolder="scheduler")
        self.pipeline = StableDiffusion3KontextPipeline(
            transformer=transformer, vae=vae, text_encoder=None, tokenizer=None,
            text_encoder_2=None, tokenizer_2=None, text_encoder_3=None,
            tokenizer_3=None, scheduler=scheduler)

    def _understand(self, prompt, pil_images, max_tokens=1024, temperature=0.0):
        content = [{"type": "image", "image": im} for im in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=list(pil_images) or None,
                                videos=None, padding=True,
                                return_tensors="pt").to("cuda")
        out = self.lmm.generate(**inputs, max_new_tokens=max_tokens,
                                do_sample=temperature > 0)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True)[0].strip()

    def _embed_prompts(self, prompt, image=None):
        torch = self.torch
        if image is None:
            messages = [[{"role": "user", "content": [
                {"type": "text", "text": f"Generate an image: {txt}"}]}]
                for txt in [prompt, NEG_PROMPT]]
            texts = [self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True) for m in messages]
            inputs = self.processor(text=texts, images=None, videos=None,
                                    padding=True, return_tensors="pt").to("cuda")
            input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
            nq = self.conditioner.config.num_queries
            input_ids = torch.cat([input_ids, input_ids.new_zeros(2, nq)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(2, nq)], dim=1)
            inputs_embeds = self.lmm.get_input_embeddings()(input_ids)
            inputs_embeds[:, -nq:] = self.conditioner.meta_queries[None].expand(2, -1, -1)
            outputs = self.lmm.model(inputs_embeds=inputs_embeds,
                                     attention_mask=attention_mask, use_cache=False)
        else:
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": txt}]}]
                for txt in [prompt, NEG_PROMPT]]
            texts = [self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True) for m in messages]
            mp_ = int(image.height * 28 / 32 * image.width * 28 / 32)
            inputs = self.processor(text=texts, images=[image] * 2,
                                    min_pixels=mp_, max_pixels=mp_, videos=None,
                                    padding=True, return_tensors="pt").to("cuda")
            input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
            nq = self.conditioner.config.num_queries
            input_ids = torch.cat([input_ids, input_ids.new_zeros(2, nq)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(2, nq)], dim=1)
            inputs_embeds = self.lmm.get_input_embeddings()(input_ids)
            inputs_embeds[:, -nq:] = self.conditioner.meta_queries[None].expand(2, -1, -1)
            image_embeds = self.lmm.visual(inputs.pixel_values,
                                           grid_thw=inputs.image_grid_thw)
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|image_pad|>")
            inputs_embeds[input_ids == image_token_id] = image_embeds
            self.lmm.model.rope_deltas = None
            outputs = self.lmm.model(inputs_embeds=inputs_embeds,
                                     attention_mask=attention_mask,
                                     image_grid_thw=inputs.image_grid_thw,
                                     use_cache=False)
        nq = self.conditioner.config.num_queries
        hidden = outputs.last_hidden_state[:, -nq:]
        return self.conditioner(hidden)

    def _generate(self, prompt, pil_images):
        torch = self.torch
        image = pil_images[0] if pil_images else None
        prompt_embeds, pooled = self._embed_prompts(prompt, image=image)
        h = image.height if image else 512
        w = image.width if image else 512
        kwargs = dict(
            prompt_embeds=prompt_embeds[:1], pooled_prompt_embeds=pooled[:1],
            negative_prompt_embeds=prompt_embeds[1:],
            negative_pooled_prompt_embeds=pooled[1:],
            height=h, width=w, num_inference_steps=50, guidance_scale=3.5,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )
        if image is not None:
            kwargs["image"] = image
        return self.pipeline(**kwargs).images[0]
