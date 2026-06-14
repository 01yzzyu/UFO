"""Omni-R1 (ModalityDance/Omni-R1) local adapter.

Architecture: Chameleon (ChameleonProcessor + ChameleonForConditionalGeneration).
Faithful to the official inference script:
    https://github.com/ModalityDance/Omni-R1/blob/main/src/Inference/inference.py
(model.generate with multimodal_generation_mode; interleaved text/image token
decoding via model.decode_image_tokens).

Prerequisites:
    git clone https://github.com/ModalityDance/Omni-R1 && cd Omni-R1
    pip install -r requirements.txt   # transformers with Chameleon support
    # weights: ModalityDance/Omni-R1  (or ModalityDance/Omni-R1-Zero)

models.yaml entry:
    - {name: Omni-R1, group: unified, provider: local, local_adapter: omni_r1,
       model_path: ModalityDance/Omni-R1,
       gen_mode: interleaved-text-image, und_mode: text}
"""

from PIL import Image

from .base_local import LocalProvider


class OmniR1Adapter(LocalProvider):
    def _load(self):
        import torch
        from transformers import (
            ChameleonProcessor, ChameleonForConditionalGeneration)

        self.torch = torch
        self.processor = ChameleonProcessor.from_pretrained(self.model_path)
        self.model = ChameleonForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
        cfg = self.model.config
        self.boi = int(getattr(cfg, "boi_token_id"))
        self.eoi = int(getattr(cfg, "eoi_token_id"))
        self.eos = int(getattr(cfg, "eos_token_id"))
        # transformers Chameleon multimodal_generation_mode valid values:
        # "text-only" | "image-only" | "interleaved-text-image" | "unrestricted"
        self.und_mode = self.extra.get("und_mode", "text-only")
        self.gen_mode = self.extra.get("gen_mode", "interleaved-text-image")

    # -- helpers (from official inference.py) ------------------------------
    def _build_tokens(self, prompt, pil_images):
        if pil_images:
            inputs = self.processor(
                prompt, images=pil_images, padding=False,
                return_for_text_completion=True, return_tensors="pt")
        else:
            inputs = self.processor(
                prompt, padding=False, return_for_text_completion=True,
                return_tensors="pt")
        inputs = inputs.to(next(self.model.parameters()).device)
        return inputs["input_ids"].to(dtype=self.torch.long)

    def _split_interleaved(self, tokens):
        segments, current, in_image = [], [], False
        for t in tokens:
            if t == self.boi:
                if current:
                    segments.append(("text", current)); current = []
                in_image = True
                continue
            if t == self.eoi and in_image:
                segments.append(("image", current)); current = []
                in_image = False
                continue
            current.append(t)
        if current:
            segments.append(("image" if in_image else "text", current))
        return segments

    def _pixels_to_pil(self, pixels):
        ip = (getattr(self.processor, "image_processor", None)
              or getattr(self.processor, "feature_extractor", None))
        px = ip.postprocess(pixels.float(), do_rescale=True, do_unnormalize=True)
        arr = px[0].permute(1, 2, 0).detach().cpu().numpy()
        return Image.fromarray(arr)

    def _generate_tokens(self, input_ids, mode, max_tokens):
        from transformers.generation.stopping_criteria import (
            StoppingCriteria, StoppingCriteriaList)
        eos = self.eos

        class _StopOnEos(StoppingCriteria):
            def __call__(self, ids, scores, **kw):
                return bool((ids[0, -1] == eos).item())

        out = self.model.generate(
            input_ids=input_ids, max_new_tokens=max_tokens,
            do_sample=False, pad_token_id=1, multimodal_generation_mode=mode,
            stopping_criteria=StoppingCriteriaList([_StopOnEos()]))
        return out[0].tolist()

    # -- understanding -----------------------------------------------------
    def _understand(self, prompt, pil_images, max_tokens=512, temperature=0.0):
        placeholders = "".join(["<image>"] * len(pil_images))
        input_ids = self._build_tokens(f"{placeholders}{prompt}", list(pil_images))
        full = self._generate_tokens(input_ids, self.und_mode, max_tokens)
        gen = full[input_ids.shape[1]:]
        text = ""
        for kind, seg in self._split_interleaved(gen):
            if kind == "text":
                seg_t = self.torch.tensor([seg], device=self.model.device,
                                          dtype=self.torch.long)
                text += self.processor.batch_decode(
                    seg_t, skip_special_tokens=True)[0]
        return text.strip()

    # -- image generation --------------------------------------------------
    def _generate(self, prompt, pil_images, max_tokens=2048):
        placeholders = "".join(["<image>"] * len(pil_images))
        input_ids = self._build_tokens(
            f"{placeholders}{prompt}" if pil_images else prompt, list(pil_images))
        full = self._generate_tokens(input_ids, self.gen_mode, max_tokens)
        gen = full[input_ids.shape[1]:]
        for kind, seg in self._split_interleaved(gen):
            if kind == "image" and seg:
                seg_t = self.torch.tensor([seg], device=self.model.device,
                                          dtype=self.torch.long)
                pixels = self.model.decode_image_tokens(seg_t)
                return self._pixels_to_pil(pixels)
        return None
