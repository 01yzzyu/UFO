"""Janus-Pro (deepseek-ai/Janus-Pro-7B / -1B) local adapter.

Based on the official repo: https://github.com/deepseek-ai/Janus
(understanding + text-to-image generation via VLChatProcessor + MultiModalityCausalLM).

Prerequisites:
    git clone https://github.com/deepseek-ai/Janus && cd Janus && pip install -e .
    # weights: deepseek-ai/Janus-Pro-7B
    export PYTHONPATH=/path/to/Janus:$PYTHONPATH

models.yaml entry:
    - {name: Janus-Pro, group: unified, provider: local, local_adapter: janus_pro,
       model_path: deepseek-ai/Janus-Pro-7B}
"""

import numpy as np
import PIL.Image

from .base_local import LocalProvider


class JanusProAdapter(LocalProvider):
    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM
        from janus.models import VLChatProcessor

        self.torch = torch
        self.processor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = (
            AutoModelForCausalLM.from_pretrained(self.model_path,
                                                 trust_remote_code=True)
            .to(torch.bfloat16).cuda().eval()
        )

    def _understand(self, prompt, pil_images, max_tokens=512, temperature=0.0):
        placeholders = "".join(["<image_placeholder>\n"] * len(pil_images))
        conversation = [
            {"role": "<|User|>", "content": f"{placeholders}{prompt}",
             "images": [""] * len(pil_images)},
            {"role": "<|Assistant|>", "content": ""},
        ]
        inputs = self.processor(
            conversations=conversation, images=list(pil_images),
            force_batchify=True,
        ).to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01) if temperature > 0 else None,
            use_cache=True,
        )
        return self.tokenizer.decode(outputs[0].cpu().tolist(),
                                     skip_special_tokens=True).strip()

    def _generate(self, prompt, pil_images, image_size=384, cfg_weight=5.0,
                  temperature=1.0, parallel_size=1):
        """Text-conditioned image generation (Janus-Pro generates from text).

        Janus-Pro's official generation is text-to-image; input images are not
        used as a generation condition. We render one image from the prompt.
        """
        torch = self.torch
        vl = self.model
        proc = self.processor

        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft = proc.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=proc.sft_format,
            system_prompt="",
        )
        prompt_text = sft + proc.image_start_tag

        input_ids = proc.tokenizer.encode(prompt_text)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size * 2, len(input_ids)),
                             dtype=torch.int).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = proc.pad_id
        inputs_embeds = vl.language_model.get_input_embeddings()(tokens)

        image_token_num = 576
        generated_tokens = torch.zeros((parallel_size, image_token_num),
                                       dtype=torch.int).cuda()
        outputs = None
        for k in range(image_token_num):
            out = vl.language_model.model(
                inputs_embeds=inputs_embeds, use_cache=True,
                past_key_values=outputs.past_key_values if k else None,
            )
            outputs = out
            hidden = out.last_hidden_state[:, -1, :]
            logits = vl.gen_head(hidden)
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, k] = next_token.squeeze(dim=-1)
            next_token = torch.cat(
                [next_token.unsqueeze(1)] * 2, dim=1).view(-1)
            img_embeds = vl.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = vl.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, image_size // 16, image_size // 16],
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return PIL.Image.fromarray(dec[0])
