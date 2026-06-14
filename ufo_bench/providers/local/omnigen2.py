"""OmniGen2 (VectorSpaceLab/OmniGen2) local adapter.

Based on the official repo: https://github.com/VectorSpaceLab/OmniGen2
  - generation/editing : OmniGen2Pipeline
  - understanding       : Qwen2.5-VL backbone (chat), see app_chat.py

Prerequisites:
    git clone https://github.com/VectorSpaceLab/OmniGen2 && cd OmniGen2
    pip install -r requirements.txt
    # weights: OmniGen2/OmniGen2
    export PYTHONPATH=/path/to/OmniGen2:$PYTHONPATH

models.yaml entry:
    - {name: OmniGen2, group: unified, provider: local, local_adapter: omnigen2,
       model_path: OmniGen2/OmniGen2}
"""

from .base_local import LocalProvider


class OmniGen2Adapter(LocalProvider):
    def _load(self):
        import torch
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

        self.torch = torch
        self.pipe = OmniGen2Pipeline.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
        )
        self.pipe = self.pipe.to("cuda")
        # The pipeline exposes the MLLM (Qwen2.5-VL) used for understanding.
        self.mllm = getattr(self.pipe, "mllm", None)
        self.processor = getattr(self.pipe, "processor", None)

    def _understand(self, prompt, pil_images, max_tokens=512, temperature=0.0):
        if self.mllm is None or self.processor is None:
            raise RuntimeError(
                "OmniGen2 understanding backbone not exposed by this build; "
                "answer this model via its Qwen2.5-VL chat endpoint instead."
            )
        content = [{"type": "image", "image": im} for im in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=list(pil_images) or None,
                                return_tensors="pt").to(self.mllm.device)
        out = self.mllm.generate(**inputs, max_new_tokens=max_tokens,
                                 do_sample=temperature > 0)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True)[0].strip()

    def _generate(self, prompt, pil_images, num_inference_steps=50, width=1024,
                  height=1024, text_guidance_scale=5.0, image_guidance_scale=2.0,
                  seed=0):
        # Mirrors https://github.com/VectorSpaceLab/OmniGen2/blob/main/inference.py
        negative_prompt = ("(((deformed))), blurry, over saturation, bad anatomy, "
                           "disfigured, poorly drawn face, mutation, mutated, "
                           "(extra_limb), (ugly), (poorly drawn hands), fused fingers, "
                           "messy drawing, broken legs censor, censored, censor_bar")
        generator = self.torch.Generator(device="cuda").manual_seed(seed)
        kwargs = dict(
            prompt=prompt, input_images=list(pil_images) if pil_images else None,
            width=width, height=height, num_inference_steps=num_inference_steps,
            max_sequence_length=1024, text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale, cfg_range=(0.0, 1.0),
            negative_prompt=negative_prompt, num_images_per_prompt=1,
            generator=generator, output_type="pil",
        )
        result = self.pipe(**kwargs)
        imgs = getattr(result, "images", None)
        return imgs[0] if imgs else None
