"""Ovis-U1 (AIDC-AI/Ovis-U1-3B) local adapter.

Faithful to the official inference scripts:
  understanding : https://github.com/AIDC-AI/Ovis-U1/blob/main/test_img_to_txt.py
  text-to-image : https://github.com/AIDC-AI/Ovis-U1/blob/main/test_txt_to_img.py
  image editing : https://github.com/AIDC-AI/Ovis-U1/blob/main/test_img_edit.py

The real API uses model.preprocess_inputs / model.generate (understanding) and
model.generate_condition + model.generate_img (generation/editing). The HF card's
`model.chat()` is NOT the real API.

Prerequisites:
    git clone https://github.com/AIDC-AI/Ovis-U1 && cd Ovis-U1
    pip install -r requirements.txt && pip install -e .
    # weights: AIDC-AI/Ovis-U1-3B

models.yaml entry:
    - {name: Ovis-U1, group: unified, provider: local, local_adapter: ovis_u1,
       model_path: AIDC-AI/Ovis-U1-3B}
"""

from PIL import Image

from .base_local import LocalProvider


class OvisU1Adapter(LocalProvider):
    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).eval().to("cuda").to(torch.bfloat16)

    # -- shared input builder (from the official test scripts) -------------
    def _build_inputs(self, prompt, pil_image, target_w=None, target_h=None,
                      generation_preface=''):
        torch = self.torch
        model = self.model
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        vae_pixel_values = None
        if pil_image is not None and target_w is not None:
            target_size = (int(target_w), int(target_h))
            pil_image, vae_pixel_values, cond_img_ids = \
                model.visual_generator.process_image_aspectratio(pil_image, target_size)
            cond_img_ids[..., 0] = 1.0
            vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)
            w, h = pil_image.width, pil_image.height
            rh, rw = visual_tokenizer.smart_resize(
                h, w, max_pixels=visual_tokenizer.image_processor.min_pixels)
            pil_image = pil_image.resize((rw, rh))

        prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            prompt, [pil_image], generation_preface=generation_preface,
            return_labels=False, propagate_exception=False,
            multimodal_type='single_image', fix_sample_overall_length_navit=False)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        if pixel_values is not None:
            pixel_values = torch.cat(
                [pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16)], dim=0)
        if grid_thws is not None:
            grid_thws = torch.cat([grid_thws.to(device=visual_tokenizer.device)], dim=0)
        return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values

    # -- understanding (single- and multi-image, official APIs) ------------
    def _understand(self, prompt, pil_images, max_tokens=4096, temperature=0.0):
        torch = self.torch
        model = self.model
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        gen_kwargs = dict(
            max_new_tokens=max_tokens, do_sample=False, top_p=None, top_k=None,
            temperature=None, repetition_penalty=None,
            eos_token_id=text_tokenizer.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id, use_cache=True,
        )
        n = len(pil_images)
        if n <= 1:
            # test_img_to_txt.py
            full_prompt = " \n" + prompt
            mm_type = "single_image"
            images = pil_images[0] if n == 1 else None
            images_list = [images]
        else:
            # test_multi_img_to_txt.py
            full_prompt = "".join([f" Image {i + 1}: \n" for i in range(n)]) + prompt
            mm_type = "multiple_image"
            images_list = list(pil_images)

        _p, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            full_prompt, images_list, generation_preface='', return_labels=False,
            propagate_exception=False, multimodal_type=mm_type,
            fix_sample_overall_length_navit=False)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        if pixel_values is not None:
            pixel_values = torch.cat(
                [pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16)], dim=0)
        if grid_thws is not None:
            grid_thws = torch.cat([grid_thws.to(device=visual_tokenizer.device)], dim=0)
        with torch.inference_mode():
            out = model.generate(
                input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                grid_thws=grid_thws, **gen_kwargs)[0]
        return text_tokenizer.decode(out, skip_special_tokens=True)

    # -- generation / editing ----------------------------------------------
    def _generate(self, prompt, pil_images, height=1024, width=1024, steps=50,
                  txt_cfg=5.0, img_cfg=1.5, seed=42):
        torch = self.torch
        model = self.model
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        edit = bool(pil_images)

        if edit:
            input_img = pil_images[0]
            w, h = input_img.size
            height, width = visual_tokenizer.smart_resize(h, w, factor=32)

        gen_kwargs = dict(
            max_new_tokens=1024, do_sample=False, top_p=None, top_k=None,
            temperature=None, repetition_penalty=None,
            eos_token_id=text_tokenizer.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id, use_cache=True,
            height=height, width=width, num_steps=steps, seed=seed,
            img_cfg=img_cfg if edit else 0, txt_cfg=txt_cfg,
        )
        uncond_image = Image.new("RGB", (width, height), (255, 255, 255)).convert("RGB")
        uncond_prompt = " \nGenerate an image."

        ids, pv, am, gt, _ = self._build_inputs(uncond_prompt, uncond_image, width, height)
        with torch.inference_mode():
            no_both_cond = model.generate_condition(
                ids, pixel_values=pv, attention_mask=am, grid_thws=gt, **gen_kwargs)

        no_txt_cond = None
        if edit:
            input_img = input_img.resize((width, height))
            ids, pv, am, gt, _ = self._build_inputs(uncond_prompt, input_img, width, height)
            with torch.inference_mode():
                no_txt_cond = model.generate_condition(
                    ids, pixel_values=pv, attention_mask=am, grid_thws=gt, **gen_kwargs)
            cond_prompt = " \n" + prompt.strip()
            cond_image = input_img
        else:
            cond_prompt = (" \nDescribe the image by detailing the color, shape, "
                           "size, texture, quantity, text, and spatial relationships "
                           "of the objects:" + prompt)
            cond_image = uncond_image

        ids, pv, am, gt, vae_pv = self._build_inputs(cond_prompt, cond_image, width, height)
        with torch.inference_mode():
            cond = model.generate_condition(
                ids, pixel_values=pv, attention_mask=am, grid_thws=gt, **gen_kwargs)
        cond["vae_pixel_values"] = vae_pv
        images = model.generate_img(
            cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
        return images[0]
