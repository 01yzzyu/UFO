# UFM Official Inference Code (for writing local adapters)

This file collects the **official, verbatim inference code** for each Unified
Foundation Model (UFM) in the UFO paper, pulled directly from each project's
official GitHub repo / HuggingFace model card, with the exact source URL above
each block. Use these to write the local adapters in
`ufo_bench/providers/local/` yourself.

For UFO we need, per model, two capabilities:
- **Understanding** (answer a question given image(s)) → used by `complete()`
- **Image generation / editing** (produce the visual cue) → used by `generate_image()`

> All of these run **locally on GPU**. Clone each repo, install its
> requirements, download weights, and add the repo to `PYTHONPATH`. None of this
> was runnable in the environment where this file was compiled, so the code below
> is the upstream original — verify versions against the repo you clone.

Paper UFMs, their official sources, and the adapter that implements each
(`ufo_bench/providers/local/<adapter>.py`):

| # | Model | Adapter | Official repo | Weights |
| --- | --- | --- | --- | --- |
| [1](#1-bagel) | Bagel | `bagel` | https://github.com/ByteDance-Seed/BAGEL | ByteDance-Seed/BAGEL-7B-MoT |
| [2](#2-emu3) | Emu3 | `emu3` | https://github.com/baaivision/Emu3 | BAAI/Emu3-Chat, BAAI/Emu3-Gen, BAAI/Emu3-VisionTokenizer |
| [3](#3-janus-pro) | Janus-Pro | `janus_pro` | https://github.com/deepseek-ai/Janus | deepseek-ai/Janus-Pro-7B |
| [4](#4-ovis-u1) | Ovis-U1 | `ovis_u1` | https://github.com/AIDC-AI/Ovis-U1 | AIDC-AI/Ovis-U1-3B |
| [5](#5-omnigen2) | OmniGen2 | `omnigen2` | https://github.com/VectorSpaceLab/OmniGen2 | OmniGen2/OmniGen2 |
| [6](#6-unipic2-metaquery) | UniPic2-Metaquery | `unipic2` | https://github.com/SkyworkAI/UniPic | Skywork/UniPic2-Metaquery-9B |
| [7](#7-unipic1) | UniPic1 | `unipic1` | https://github.com/SkyworkAI/UniPic | Skywork/Skywork-UniPic-1.5B |
| [8](#8-omni-r1) | Omni-R1 | `omni_r1` | https://github.com/ModalityDance/Omni-R1 | ModalityDance/Omni-R1 |
| [9](#9-unicot-uni-cot) | UniCoT | `unicot` | https://github.com/Fr0zenCrane/UniCoT | Fr0zencr4nE/UniCoT-7B-MoT |
| [10](#10-uniworld-v1) | UniWorld-V1 | `uniworld_v1` | https://github.com/PKU-YuanGroup/UniWorld | LanguageBind/UniWorld-V1 |

Each section below follows the same layout: **Repo / Weights → Install →
Understanding code → Generation code** (verbatim from the source, with the URL).

---

## 1. Bagel

- Repo: https://github.com/ByteDance-Seed/BAGEL
- Weights: https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
- Inference API: `InterleaveInferencer` (understanding via `understanding_output=True`;
  image generation via the default image output).

### Install (official README)
```bash
git clone https://github.com/bytedance-seed/BAGEL.git
cd BAGEL
conda create -n bagel python=3.10 -y
conda activate bagel
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

### Model loading — source: https://github.com/ByteDance-Seed/Bagel/blob/main/app.py
```python
import os
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer

model_path = "models/BAGEL-7B-MoT"   # download from HF to here

llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

config = BagelConfig(
    visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
    vae_config=vae_config, vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh', latent_patch_size=2, max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

device_map = infer_auto_device_map(
    model, max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
same_device_modules = [
    'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
    'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed',
]
if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        device_map[k] = first_device if k not in device_map else device_map[k]
        device_map.setdefault(k, "cuda:0")
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

model = load_checkpoint_and_dispatch(
    model, checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map, offload_buffers=True, offload_folder="offload",
    dtype=torch.bfloat16, force_hooks=True,
).eval()

inferencer = InterleaveInferencer(
    model=model, vae_model=vae_model, tokenizer=tokenizer,
    vae_transform=vae_transform, vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)
```

### Inference API — source: https://github.com/ByteDance-Seed/Bagel/blob/main/inferencer.py
```python
# Understanding (VQA): pass images then text, set understanding_output=True
output_list = inferencer.interleave_inference(
    [pil_image, "your question"], understanding_output=True,
    max_think_token_n=1000, do_sample=False, text_temperature=0.3,
)
answer = next(x for x in output_list if isinstance(x, str))

# Image generation (visual cue): default output is an image
output_list = inferencer.interleave_inference(
    [pil_image, "generation prompt"], understanding_output=False,
    cfg_text_scale=4.0, cfg_img_scale=1.5, cfg_interval=[0.4, 1.0],
    timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=0.0, cfg_renorm_type="global",
)
image = next(x for x in output_list if hasattr(x, "save"))

# Convenience __call__:
out = inferencer(image=pil_image, text="...", understanding_output=True)  # -> {"image":..., "text":...}
```

---

## 2. Emu3

- Repo: https://github.com/baaivision/Emu3
- Weights: BAAI/Emu3-Chat (understanding), BAAI/Emu3-Gen (generation),
  BAAI/Emu3-VisionTokenizer (shared VQ tokenizer)
- NOTE: the processor import is `from emu3.mllm.processing_emu3 import Emu3Processor`.

### Understanding — source: https://huggingface.co/BAAI/Emu3-Chat (model card Quickstart)
```python
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
import torch

from emu3.mllm.processing_emu3 import Emu3Processor   # repo module

EMU_HUB = "BAAI/Emu3-Chat"
VQ_HUB = "BAAI/Emu3-VisionTokenier"

model = AutoModelForCausalLM.from_pretrained(
    EMU_HUB, device_map="cuda:0", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

text = "Please describe the image"
image = Image.open("assets/demo.png")
inputs = processor(text=text, image=image, mode='U', return_tensors="pt", padding="longest")

GENERATION_CONFIG = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id, max_new_tokens=1024,
)
outputs = model.generate(
    inputs.input_ids.to("cuda:0"), GENERATION_CONFIG,
    attention_mask=inputs.attention_mask.to("cuda:0"),
)
outputs = outputs[:, inputs.input_ids.shape[-1]:]
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
```

### Image generation — source: https://github.com/baaivision/Emu3/blob/main/image_generation.py
```python
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch
from emu3.mllm.processing_emu3 import Emu3Processor

EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

model = AutoModelForCausalLM.from_pretrained(
    EMU_HUB, device_map="cuda:0", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", trust_remote_code=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

classifier_free_guidance = 3.0
prompt = ["a portrait of young girl.", "a shiba inu"]
prompt = [p + POSITIVE_PROMPT for p in prompt]

kwargs = dict(mode='G', ratio=["1:1", "16:9"], image_area=model.config.image_area,
              return_tensors="pt", padding="longest")
pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=[NEGATIVE_PROMPT] * len(prompt), **kwargs)

GENERATION_CONFIG = GenerationConfig(
    use_cache=True, eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id, max_new_tokens=40960,
    do_sample=True, top_k=2048,
)
h = pos_inputs.image_size[:, 0]
w = pos_inputs.image_size[:, 1]
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList([
    UnbatchedClassifierFreeGuidanceLogitsProcessor(
        classifier_free_guidance, model,
        unconditional_ids=neg_inputs.input_ids.to("cuda:0")),
    PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
])
outputs = model.generate(
    pos_inputs.input_ids.to("cuda:0"), GENERATION_CONFIG,
    logits_processor=logits_processor,
    attention_mask=pos_inputs.attention_mask.to("cuda:0"),
)
for idx_i, out in enumerate(outputs):
    mm_list = processor.decode(out)
    for idx_j, im in enumerate(mm_list):
        if isinstance(im, Image.Image):
            im.save(f"result_{idx_i}_{idx_j}.png")
```

---

## 3. Janus-Pro

- Repo: https://github.com/deepseek-ai/Janus
- Weights: deepseek-ai/Janus-Pro-7B (use this in place of the 1.3B path below)
- Install: `pip install -e .` inside the cloned repo (provides the `janus` package).

### Understanding — source: https://github.com/deepseek-ai/Janus/blob/main/inference.py
```python
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {"role": "User",
     "content": "<image_placeholder>\nConvert the formula into latex code.",
     "images": ["images/equation.png"]},
    {"role": "Assistant", "content": ""},
]
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds, attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id, max_new_tokens=512,
    do_sample=False, use_cache=True,
)
answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(answer)
```

### Image generation (text-to-image) — source: https://github.com/deepseek-ai/Janus/blob/main/generation_inference.py
```python
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image

model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {"role": "User", "content": "A close-up high-contrast photo of Sydney Opera House ..."},
    {"role": "Assistant", "content": ""},
]
sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation, sft_format=vl_chat_processor.sft_format, system_prompt="")
prompt = sft_format + vl_chat_processor.image_start_tag

@torch.inference_mode()
def generate(mmgpt, vl_chat_processor, prompt, temperature=1, parallel_size=16,
             cfg_weight=5, image_token_num_per_image=576, img_size=384, patch_size=16):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        PIL.Image.fromarray(visual_img[i]).save(os.path.join('generated_samples', f"img_{i}.jpg"))

generate(vl_gpt, vl_chat_processor, prompt)
```
> Note: Janus-Pro generation is **text-to-image** (does not condition on an input image).

---

## 4. Ovis-U1

- Repo: https://github.com/AIDC-AI/Ovis-U1
- Weights: AIDC-AI/Ovis-U1-3B
- Install: `pip install -r requirements.txt && pip install -e .`
- The real API uses `model.preprocess_inputs`, `model.generate` (understanding),
  and `model.generate_condition` + `model.generate_img` (generation/editing).
  (The HF card's `model.chat()` is NOT the real API.)

### Understanding — source: https://github.com/AIDC-AI/Ovis-U1/blob/main/test_img_to_txt.py
```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

model, loading_info = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis-U1-3B", torch_dtype=torch.bfloat16,
    output_loading_info=True, trust_remote_code=True)
model = model.eval().to("cuda").to(torch.bfloat16)

def build_inputs(model, text_tokenizer, visual_tokenizer, prompt, pil_image):
    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt, [pil_image], generation_preface='', return_labels=False,
        propagate_exception=False, multimodal_type='single_image',
        fix_sample_overall_length_navit=False)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = torch.cat([pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16)], dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([grid_thws.to(device=visual_tokenizer.device)], dim=0)
    return input_ids, pixel_values, attention_mask, grid_thws

def pipe_txt_gen(model, pil_image, prompt):
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    gen_kwargs = dict(max_new_tokens=4096, do_sample=False, top_p=None, top_k=None,
                      temperature=None, repetition_penalty=None,
                      eos_token_id=text_tokenizer.eos_token_id,
                      pad_token_id=text_tokenizer.pad_token_id, use_cache=True)
    prompt = " \n" + prompt
    input_ids, pixel_values, attention_mask, grid_thws = build_inputs(
        model, text_tokenizer, visual_tokenizer, prompt, pil_image)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, pixel_values=pixel_values,
                                    attention_mask=attention_mask, grid_thws=grid_thws,
                                    **gen_kwargs)[0]
    return text_tokenizer.decode(output_ids, skip_special_tokens=True)

print(pipe_txt_gen(model, Image.open("docs/imgs/cat.png").convert("RGB"), "What is it?"))
```

### Text-to-image — source: https://github.com/AIDC-AI/Ovis-U1/blob/main/test_txt_to_img.py
(Full `build_inputs` + `pipe_t2i`; uses `model.visual_generator.process_image_aspectratio`,
`model.generate_condition`, `model.generate_img`. cfg via `txt_cfg`/`img_cfg`.)

### Image editing — source: https://github.com/AIDC-AI/Ovis-U1/blob/main/test_img_edit.py
(`pipe_img_edit`: build unconditional, no-text, and full conditions, then `generate_img`.)

> See those two files in the repo for the exact `build_inputs`/`pipe_t2i`/`pipe_img_edit`
> bodies (they are long; copy them verbatim into your adapter).

---

## 5. OmniGen2

- Repo: https://github.com/VectorSpaceLab/OmniGen2
- Weights: OmniGen2/OmniGen2
- Generation pipeline: `OmniGen2Pipeline`. Understanding: chat backbone via `app_chat.py`.

### Generation/editing — source: https://github.com/VectorSpaceLab/OmniGen2/blob/main/inference.py
```python
import torch
from PIL import Image, ImageOps
from accelerate import Accelerator
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

accelerator = Accelerator(mixed_precision="bf16")
weight_dtype = torch.bfloat16

pipeline = OmniGen2Pipeline.from_pretrained(
    "OmniGen2/OmniGen2", torch_dtype=weight_dtype, trust_remote_code=True)
pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
    "OmniGen2/OmniGen2", subfolder="transformer", torch_dtype=weight_dtype)
pipeline = pipeline.to(accelerator.device)

# input_images: list[PIL.Image] or None for pure text-to-image
input_images = [Image.open("ref.png").convert("RGB")]   # optional
generator = torch.Generator(device=accelerator.device).manual_seed(0)
results = pipeline(
    prompt="A dog running in the park",
    input_images=input_images,
    width=1024, height=1024, num_inference_steps=50, max_sequence_length=1024,
    text_guidance_scale=5.0, image_guidance_scale=2.0, cfg_range=(0.0, 1.0),
    negative_prompt="(((deformed))), blurry, bad anatomy, ...",
    num_images_per_prompt=1, generator=generator, output_type="pil",
)
results.images[0].save("output.png")
```
> Understanding: run `app_chat.py` in the repo (Qwen2.5-VL backbone). Inspect that
> file for the exact chat call to wire into `complete()`.

---

## 6. UniPic2-Metaquery

- Repo: https://github.com/SkyworkAI/UniPic  (UniPic-2)
- Weights: Skywork/UniPic2-Metaquery-9B (+ Qwen/Qwen2.5-VL-7B-Instruct)
- Source: https://huggingface.co/Skywork/UniPic2-Metaquery-9B (model card)

### Install
```bash
git clone https://github.com/SkyworkAI/UniPic
cd UniPic-2
conda create -n unipic python=3.10 && conda activate unipic
pip install -r requirements.txt
```

### Text-to-image (verbatim from card)
```python
import torch
from PIL import Image
from unipicv2.pipeline_stable_diffusion_3_kontext import StableDiffusion3KontextPipeline
from unipicv2.transformer_sd3_kontext import SD3Transformer2DKontextModel
from unipicv2.stable_diffusion_3_conditioner import StableDiffusion3Conditioner
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL

pretrained = "Skywork/UniPic2-Metaquery-9B"
transformer = SD3Transformer2DKontextModel.from_pretrained(pretrained, subfolder="transformer", torch_dtype=torch.bfloat16).cuda()
vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", torch_dtype=torch.bfloat16).cuda()
lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
processor.chat_template = processor.chat_template.replace(
    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}", "")
conditioner = StableDiffusion3Conditioner.from_pretrained(pretrained, subfolder="conditioner", torch_dtype=torch.bfloat16).cuda()
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
pipeline = StableDiffusion3KontextPipeline(
    transformer=transformer, vae=vae, text_encoder=None, tokenizer=None,
    text_encoder_2=None, tokenizer_2=None, text_encoder_3=None, tokenizer_3=None, scheduler=scheduler)

prompt = 'a pig with wings and a top hat flying over a happy futuristic scifi city'
negative_prompt = 'blurry, low quality, low resolution, distorted, deformed, ...'
messages = [[{"role": "user", "content": [{"type": "text", "text": f'Generate an image: {txt}'}]}]
            for txt in [prompt, negative_prompt]]
texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
inputs = processor(text=texts, images=None, videos=None, padding=True, return_tensors="pt").to("cuda")
input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
input_ids = torch.cat([input_ids, input_ids.new_zeros(2, conditioner.config.num_queries)], dim=1)
attention_mask = torch.cat([attention_mask, attention_mask.new_ones(2, conditioner.config.num_queries)], dim=1)
inputs_embeds = lmm.get_input_embeddings()(input_ids)
inputs_embeds[:, -conditioner.config.num_queries:] = conditioner.meta_queries[None].expand(2, -1, -1)
outputs = lmm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
hidden_states = outputs.last_hidden_state[:, -conditioner.config.num_queries:]
prompt_embeds, pooled_prompt_embeds = conditioner(hidden_states)
image = pipeline(
    prompt_embeds=prompt_embeds[:1], pooled_prompt_embeds=pooled_prompt_embeds[:1],
    negative_prompt_embeds=prompt_embeds[1:], negative_pooled_prompt_embeds=pooled_prompt_embeds[1:],
    height=512, width=384, num_inference_steps=50, guidance_scale=3.5,
    generator=torch.Generator(device=transformer.device).manual_seed(42)).images[0]
image.save("text2image.png")
```
Image editing (also on the card): build messages with `{"type":"image","image":image}`,
run the Qwen2.5-VL vision path (`lmm.visual(pixel_values, grid_thw=...)`), then call
`pipeline(image=image, prompt_embeds=..., ...)`. See the card for the full editing block.
> Understanding: the LMM is Qwen2.5-VL — answer with a standard Qwen2.5-VL `generate` call.

---

## 7. UniPic1

- Repo: https://github.com/SkyworkAI/UniPic  (folder: UniPic-1)
- Weights: Skywork/Skywork-UniPic-1.5B
- mmengine-config model built via `src.builder.BUILDER`; load a `.bin` checkpoint.
- Install: `export PYTHONPATH=./:$PYTHONPATH` inside UniPic-1 (so `src`, `configs` import).

### Understanding — source: https://github.com/SkyworkAI/UniPic/blob/main/UniPic-1/scripts/image2text.py
```python
import numpy as np, torch
from PIL import Image
from mmengine.config import Config
from src.builder import BUILDER
from einops import rearrange

config = Config.fromfile("configs/models/qwen2_5_1_5b_kl16_mar_h.py")
model = BUILDER.build(config.model).eval().cuda().to(  # dtype set below
    BUILDER.build(config.model).dtype) if False else BUILDER.build(config.model).eval().cuda()
model = model.to(model.dtype)
model.load_state_dict(torch.load("checkpoint/pytorch_model.bin"), strict=False)

model.tokenizer.add_special_tokens({'additional_special_tokens': ["<image>"]})
image_token_idx = model.tokenizer.encode("<image>", add_special_tokens=False)[-1]

image_size = 1024
image = Image.open("data/sample.png").convert("RGB")
# expand2square to (127,127,127), then resize to image_size, then normalize to [-1,1]
image = torch.from_numpy(np.array(image)).to(dtype=model.dtype, device=model.device)
image = rearrange(image, 'h w c -> c h w')[None]
image = 2 * (image / 255) - 1

prompt = model.prompt_template['INSTRUCTION'].format(input="<image>\n" + "Describe the image in detail.")
image_length = (image_size // 16) ** 2 + 64
prompt = prompt.replace('<image>', '<image>' * image_length)
input_ids = model.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt').cuda()
with torch.no_grad():
    _, z_enc = model.extract_visual_feature(model.encode(image))
    inputs_embeds = z_enc.new_zeros(*input_ids.shape, model.llm.config.hidden_size)
    inputs_embeds[input_ids == image_token_idx] = z_enc.flatten(0, 1)
    inputs_embeds[input_ids != image_token_idx] = \
        model.llm.get_input_embeddings()(input_ids[input_ids != image_token_idx])
    output = model.llm.generate(
        inputs_embeds=inputs_embeds, use_cache=True, do_sample=False,
        max_new_tokens=1024, eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id)
print(model.tokenizer.decode(output[0]))
```

### Text-to-image — source: https://github.com/SkyworkAI/UniPic/blob/main/UniPic-1/scripts/text2image.py
```python
from einops import rearrange
prompt = "Generate an image: A glossy-coated golden retriever ..."
class_info = model.prepare_text_conditions(prompt, "Generate an image.")  # cfg_prompt
input_ids, attention_mask = class_info['input_ids'], class_info['attention_mask']
# input_ids has 2 rows (conditional + unconditional); drop row 2 if cfg == 1.0
m = n = 1024 // 16
samples = model.sample(input_ids=input_ids, attention_mask=attention_mask,
                       num_iter=32, cfg=3.0, cfg_schedule="constant",
                       temperature=1.0, progress=True, image_shape=(m, n))
samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=1, n=1)
arr = torch.clamp(127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
Image.fromarray(arr).save("output.jpg")
```

---

## 8. Omni-R1

- Repo: https://github.com/ModalityDance/Omni-R1
- Weights: ModalityDance/Omni-R1 (and Omni-R1-Zero)
- Architecture: **Chameleon** (`ChameleonProcessor` + `ChameleonForConditionalGeneration`).

### Load — source: https://huggingface.co/ModalityDance/Omni-R1 (model card)
```python
import torch
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

model_id = "ModalityDance/Omni-R1"   # or "ModalityDance/Omni-R1-Zero"
processor = ChameleonProcessor.from_pretrained(model_id)
model = ChameleonForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
```

### Core inference — source: https://github.com/ModalityDance/Omni-R1/blob/main/src/Inference/inference.py
```python
# Build prompt tokens (image(s) via Chameleon processor; <image> placeholder per image)
inputs = processor(prompt, images=pil_images, padding=False,
                   return_for_text_completion=True, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"].to(dtype=torch.long)

# Generate (multimodal_generation_mode: 'text-only' for understanding,
#           'interleaved-text-image' (or 'unrestricted') for image generation)
out = model.generate(
    input_ids=input_ids, max_new_tokens=2048, do_sample=False, pad_token_id=1,
    multimodal_generation_mode="interleaved-text-image",
    stopping_criteria=StoppingCriteriaList([StopOnToken(model.config.eos_token_id)]),
)
gen = out[0].tolist()[input_ids.shape[1]:]

# Split into text/image segments by boi/eoi token ids, then decode:
boi = model.config.boi_token_id; eoi = model.config.eoi_token_id
# text segment -> processor.batch_decode(seg_tensor, skip_special_tokens=True)
# image segment -> pixels = model.decode_image_tokens(seg_tensor)
#                  pil = processor.image_processor.postprocess(
#                            pixels.float(), do_rescale=True, do_unnormalize=True)[0]
```
(See the file for `split_interleaved_tokens`, `decode_interleaved_sample`,
`pixels_to_pil_via_processor`, and `StopOnToken` — all reproduced in the adapter
`ufo_bench/providers/local/omni_r1.py`.)

---

## 9. UniCoT (Uni-CoT)

- Repo: https://github.com/Fr0zenCrane/UniCoT
- Weights: Fr0zencr4nE/UniCoT-7B-MoT (and v0.2)
- **Extends Bagel-7B-MoT** (so the model internals mirror Bagel's `InterleaveInferencer`),
  adding a self-reflection reasoning mechanism for text-to-image.

### Install
```bash
git clone https://github.com/Fr0zenCrane/UniCoT.git
cd UniCoT
conda create -n unicot python=3.10 -y && conda activate unicot
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

### Inference — source: https://github.com/Fr0zenCrane/UniCoT (README, scripts/run_user_self_reflection.sh)
```bash
gpu_num=8
for i in $(seq 0 $((gpu_num-1))); do
    CUDA_VISIBLE_DEVICES=$i python inference_mdp_self_reflection.py \
        --group_id $i --group_num $gpu_num \
        --model_path "Fr0zencr4nE/UniCoT-7B-MoT" \
        --data_path "./test_prompts.txt" \
        --outdir "./results" \
        --cfg_text_scale 4 &
done
wait
```
> ACTION: open `inference_mdp_self_reflection.py` for the exact understanding +
> generation calls. Since UniCoT extends Bagel, the Bagel `InterleaveInferencer`
> code in §1 is the closest reference for wiring `complete()` / `generate_image()`.

---

## 10. UniWorld-V1

- Repo: https://github.com/PKU-YuanGroup/UniWorld-V1
- Weights: LanguageBind/UniWorld-V1 (library tag: `univa`, backbone `univa_qwen2p5vl`)
- Needs FLUX + SigLIP paths in addition to the model path.

### Core inference — source: https://github.com/PKU-YuanGroup/UniWorld/blob/main/UniWorld-V1/univa/serve/cli.py
```python
import torch
from transformers import AutoProcessor, SiglipImageProcessor, SiglipVisionModel
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from univa.utils.flux_pipeline import FluxPipeline
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from qwen_vl_utils import process_vision_info

device = "cuda"
model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
processor = AutoProcessor.from_pretrained(model_path, min_pixels=448*448, max_pixels=448*448)
pipe = FluxPipeline.from_pretrained(flux_path, transformer=model.denoise_tower.denoiser,
                                    torch_dtype=torch.bfloat16).to(device)
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
siglip_model = SiglipVisionModel.from_pretrained(siglip_path, torch_dtype=torch.bfloat16).to(device)

# Build chat inputs (drop the system turn), then:
# UNDERSTANDING:
generated_ids = model.generate(**inputs, max_new_tokens=128)
reply = processor.batch_decode([g[len(i):] for i,g in zip(inputs.input_ids, generated_ids)],
                               skip_special_tokens=True)[0]

# GENERATION:
siglip_hidden = siglip_model(siglip_pixel_values).last_hidden_state     # from ref images
lvlm_embeds = model(inputs.input_ids, pixel_values=inputs.pixel_values,
                    attention_mask=inputs.attention_mask,
                    image_grid_thw=inputs.image_grid_thw,
                    siglip_hidden_states=siglip_hidden, output_type="denoise_embeds")
t5_embeds, pooled = encode_prompt(text_encoders, tokenizers, prompt, 256, device, 1)
input_embeds = torch.concat([t5_embeds, lvlm_embeds], dim=1)
image = pipe(prompt_embeds=input_embeds, pooled_prompt_embeds=pooled,
             height=1024, width=1024, num_inference_steps=28, guidance_scale=3.5,
             generator=torch.Generator(device="cuda").manual_seed(42)).images[0]
```
(The official CLI uses a small `task_head` to auto-route understand vs generate;
the adapter forces the path per call, so the task_head is not needed.)

---

## How these map to UFO adapters

Each adapter in `ufo_bench/providers/local/<model>.py` subclasses `LocalProvider`
and implements:
- `_load(self)` — the model-loading block above
- `_understand(self, prompt, pil_images, max_tokens, temperature)` — the
  understanding block, returning the answer string
- `_generate(self, prompt, pil_images)` — the generation/editing block, returning
  a `PIL.Image`

Current adapter state in `ufo_bench/providers/local/` (10/10 implemented from
official code, understanding + generation):
`bagel`, `janus_pro`, `emu3`, `omnigen2`, `ovis_u1`, `unipic2`,
`unicot` (subclasses `bagel`), `omni_r1` (Chameleon interleaved decode),
`unipic1` (mmengine BUILDER), `uniworld_v1` (UnivaQwen2.5-VL + FLUX + SigLIP).

Multi-image: Bagel, UniCoT, Janus-Pro, OmniGen2, Ovis-U1 (multiple_image mode),
UniPic2, Omni-R1, UniWorld-V1 take multiple images natively. Emu3 and UniPic1 are
single-image-only officially, so multiple images are merged into one via
`imutil.concat_images` (needed for multi-image tasks and the visual/joint cue).

Corrections applied vs the first drafts: Emu3 import is
`emu3.mllm.processing_emu3`; Ovis-U1 uses `preprocess_inputs` /
`generate_condition` / `generate_img` (not `model.chat`) and `multiple_image` mode
for >1 image; Omni-R1 uses the interleaved Chameleon decode with valid
`multimodal_generation_mode` values (`text-only` / `interleaved-text-image`);
UniPic1 uses `BUILDER.build` + `model.sample`; UniWorld-V1 uses the univa backbone
+ FLUX pipeline; the registry now forwards all adapter config fields.
