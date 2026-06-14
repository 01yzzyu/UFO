# UFO Benchmark — 评测工具包

[English](README.md) | **中文**

[![Website](https://img.shields.io/badge/🌐%20Project-Page-2b6cff)](https://01yzzyu.github.io/UFO/)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-yzzyu%2FUFO-yellow)](https://huggingface.co/datasets/yzzyu/UFO)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

一个干净、可复现的评测流水线，用于在 **UFO** 基准上评测多模态模型的「组合式多模态推理」能力，与论文对齐：
*"Do Vision and Text Cues Exhibit Evidential Coupling? UFO: A Benchmark for
Compositional Multimodal Reasoning in Unified Models."*

**数据集：** https://huggingface.co/datasets/yzzyu/UFO

UFO 是一个**两步式**任务：模型先生成描述「未来状态」的中间**文本线索 (textual cue)** 和**视觉线索 (visual cue)**，再基于这些线索回答问题。本工具包在四种协议下跑完整个流程，并输出与论文表格对齐的准确率。

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="UFO 任务：3 大类 × 10 个 task">
</p>
<p align="center"><em>UFO 涵盖 3 种状态转移类别 × 10 个 task。</em></p>

---

## 快速开始（30 秒）

```bash
pip install -e .                      # 安装 `ufo-eval` 命令
cp .env.example .env                  # 在 .env 里填入 OPENROUTER_API_KEY
ufo-eval run --models GPT-5.1 --split mcq --limit 30 --out outputs/demo
```

这一条命令会跑完 **推理 → 打分 → 出表**，并打印准确率摘要。结果在 `outputs/demo/`
（`tables/results.csv`、`tables/summary.md`、`tables/main_table.tex`）。

> 不想安装？用 `python -m ufo_bench run ...` 或 `python scripts/run_eval.py ...`。

---

## 流程

<p align="center">
  <img src="assets/framework.png" width="85%" alt="UFO 推理：文本线索、视觉线索、联合线索">
</p>

给定输入图像 + 问题，模型生成中间的**文本线索**和/或**视觉线索**，然后回答——
要么直接回答（`direct`），要么基于线索回答（`textual` / `visual` / `joint`）。
工具包随后对答案打分并生成结果表。

```
            ┌─ 生成文本线索 ─┐
 图像 ──────┤                ├─► 回答 (direct/textual/visual/joint) ─► 打分 ─► 出表
 问题       └─ 生成视觉线索 ─┘
```

### 四种协议

| 协议 | 回答步骤的输入 |
| --- | --- |
| `direct`  | 仅输入图像（无中间线索） |
| `textual` | 输入图像 + 生成的**文本**线索 |
| `visual`  | 输入图像 + 生成的**视觉**线索（图像） |
| `joint`   | 输入图像 + 文本线索 + 视觉线索 |

真正的跨模态协同表现为 **joint > 单模态**。

### 任务体系（3 大类 × 10 个 task）

- **State Determination（状态确定）**：Hybridisation、Chemical、Multi-table、Multi-view
- **State Reconstruction（状态重建）**：Inpainting、Exo-to-Ego、Jigsaw
- **State Augmentation（状态增强）**：Geometric、Logical、Physics

---

## 安装与配置

```bash
pip install -e .                 # 核心
pip install -e ".[fal,dotenv]"   # 可选：fal 托管模型 + .env 自动加载
```

API key 只从环境变量读取（绝不硬编码）。放进 `.env`（装了 `python-dotenv` 会自动加载）
或用 `export` 设置：

| 环境变量 | 用途 |
| --- | --- |
| `OPENROUTER_API_KEY` / `OPENAI_API_KEY` | OpenAI 兼容模型（GPT、Qwen、Gemma） |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `provider: gemini` |
| `FAL_KEY` | `provider: fal` |
| *(无需)* | `provider: local` 的 UFM 在你的 GPU 上运行 |

---

## 使用方法

### 一条命令（推荐）

```bash
# 用配置文件（改一次 configs/run.yaml，可复现）
ufo-eval run --run-config configs/run.yaml

# 或用命令行参数（参数会覆盖配置文件）
ufo-eval run --models GPT-5.1 Qwen3-VL-8B --split mcq --limit 30 --out outputs/mcq
```

按名字、按分组、或全选模型：

```bash
ufo-eval run --models all
ufo-eval run --models group:proprietary
ufo-eval run --models group:unified           # 本地 UFM 模型
```

不调用任何 API、仅检查配置是否正确：

```bash
ufo-eval infer --source yzzyu/UFO --split mcq --models GPT-5.1 --dry-run
```

### 分步执行（如果你想）

```bash
ufo-eval infer  --source yzzyu/UFO --split mcq --models GPT-5.1 --limit 30 --out outputs/mcq
ufo-eval score  --pred outputs/mcq/GPT-5.1.json          # --model 会自动识别
ufo-eval tables --scored "outputs/mcq/*_scored.json" --out outputs/mcq/tables
```

### 你会得到什么

`outputs/.../tables/summary.md`（也会打印到控制台）：

```
| Model       | Direct | Textual | Visual | Joint |
| ---         |   ---: |    ---: |   ---: |  ---: |
| GPT-5.1     |  46.50 |   48.46 |  49.12 | 52.12 |
| Qwen3-VL-8B |  ...   |   ...   |  ...   | ...   |
```

外加 `results.csv`（每个 task 的准确率）和 `main_table.tex`（论文风格表格）。

---

## 数据集

UFO 基准托管在 Hugging Face Hub：
**https://huggingface.co/datasets/yzzyu/UFO** —— 2 个 split（`mcq`、`open`），
图像已内嵌，约 3.4k 道题，覆盖 10 个 task。

### 获取方式（任选其一）

**1. 无需任何操作 —— 自动下载（默认）。** 任何带 `--source yzzyu/UFO` 的命令都会通过
🤗 `datasets` 拉取数据，并把图像缓存到 `data_cache/`：

```bash
ufo-eval run --source yzzyu/UFO --split mcq --models GPT-5.1 --limit 30 --out outputs/mcq
```

**2. 预下载**（离线 / 集群 / 首次更快）：

```bash
ufo-eval download                       # 两个 split -> data_cache/
ufo-eval download --splits mcq --out data_cache
```

**3. 用 HF CLI 下载原始文件**（自己浏览 parquet/jsonl/图片）：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download yzzyu/UFO --repo-type dataset --local-dir ./UFO_data
```

**4. 在 Python 里加载：**

```python
from datasets import load_dataset
ds = load_dataset("yzzyu/UFO", split="mcq")   # 或 "open"
ex = ds[0]
ex["input_images"]   # list[PIL.Image]
ex["cue_image"]      # PIL.Image | None（标准答案的视觉线索）
ex["question"], ex["choice_a"], ex["answer"]
```

> 数据集是 gated/私有，或遇到限流？先 `huggingface-cli login`，或在 `.env` 里设置
> `HF_TOKEN`。公开的 `yzzyu/UFO` 不需要 token。

### 字段说明（每个 split）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id`、`category`、`task`、`question_type` | str | `question_type` ∈ {mcq, open} |
| `input_images` | list[Image] | 当前状态的图像 |
| `question` | str | 问题 |
| `choice_a`–`choice_d` | str | 选择题选项（open 题为空） |
| `answer` | str | 选项字母（mcq）或参考答案文本（open） |
| `text_cue` | str | 标准答案文本线索 |
| `cue_image` | Image \| null | 标准答案视觉线索 |
| `solution` | str | 推理过程（如有） |

### 使用本地副本

`--source /path/to/ufo_mcq.jsonl`（图片路径相对该文件解析，或用 `--image_root` 指定）。
适合你已经把数据集导出到本地的情况。

---

## 模型与平台

模型定义在 [`configs/models.yaml`](configs/models.yaml)，每个模型通过一个 `Provider` 后端访问：

| Provider | 模型 | Key |
| --- | --- | --- |
| `openai` | OpenRouter / OpenAI / DashScope：GPT、Qwen-VL、Gemma | `OPENROUTER_API_KEY` |
| `gemini` | Gemini 原生（文本 + 内联出图） | `GEMINI_API_KEY` |
| `fal` | fal.ai 托管模型 | `FAL_KEY` |
| `local` | **本地 GPU 上的统一基础模型（UFM）**（见下） | — |

### 统一基础模型（本地 GPU）

论文里的 10 个 UFM 都在 `ufo_bench/providers/local/` 有适配器，每个都按模型**官方推理代码**编写
（来源 + 逐字代码见 [`docs/UFM_OFFICIAL_INFERENCE.md`](docs/UFM_OFFICIAL_INFERENCE.md)）：
`bagel`、`janus_pro`、`emu3`、`omnigen2`、`ovis_u1`、`unipic2`、`unicot`、
`omni_r1`、`unipic1`、`uniworld_v1`。

运行某个 UFM：clone 它的官方 repo、装依赖、下载权重、`export PYTHONPATH=/path/to/repo`、
在 `configs/models.yaml` 里设好 `model_path`，然后 `ufo-eval run --models Bagel ...`。
10 个都实现了「理解」和「视觉线索生成」，所以每个模型都支持全部 4 种协议（需在你的 GPU 上验证——
这些无法在纯 CPU 环境运行）。要新增模型，复制 `ufo_bench/providers/local/_template.py`。

---

## 仓库结构

```
ufo_benchmark/
├── configs/
│   ├── models.yaml          # 模型注册表（provider、id、judge 模型）
│   └── run.yaml             # 配置驱动的 `ufo-eval run`
├── ufo_bench/
│   ├── cli.py               # `ufo-eval` 命令（所有子命令）
│   ├── config.py            # 任务体系与协议定义（与论文对齐）
│   ├── data.py              # 从本地 JSONL 或 HF 加载 UFO
│   ├── prompts.py           # 线索生成 / 回答 / 评判提示词
│   ├── inference.py         # 4 协议推理引擎
│   ├── scoring.py           # MCQ 字母匹配 + open 题 LLM 评判
│   ├── cue_eval.py          # 生成线索 vs 标准线索 的质量评测
│   ├── aggregate.py         # CSV / LaTeX / Markdown 结果表
│   ├── registry.py          # models.yaml 加载 + provider 构建
│   ├── runner.py            # 可断点续跑的并发 + JSON IO
│   ├── imutil.py            # 图像编码 / 保存 / 合并工具
│   └── providers/           # 平台后端（openai / gemini / fal / local/*）
├── scripts/                 # CLI 的薄包装（向后兼容）
├── tests/                   # 离线单测（pytest，无需 GPU/API）
└── docs/UFM_OFFICIAL_INFERENCE.md   # 每个 UFM 的官方推理代码
```

---

## 过程评测（线索质量）

衡量生成的线索是否与标准线索「状态一致」（论文的 evidential-coupling 检验）：

```bash
ufo-eval cue-eval --pred outputs/mcq/GPT-5.1.json --targets text visual
```

## 结果字段（JSON 文件中）

模型 tag = 模型的显示名。每条样本：

- `text_cue_generated_<tag>`、`image_cue_generated_<tag>` —— 生成的线索
- `pred_<protocol>_<tag>` —— 原始答案
- `score_<protocol>_<tag>` —— 0/1 正确性
- `cue_text_score_<tag>`、`cue_visual_score_<tag>` —— 线索质量分

## 测试

```bash
pip install -e ".[test]" && pytest        # 离线；无需 GPU/API/网络
```

## 说明

- 所有 API 调用都带退避重试；运行**可断点续跑**（重跑会跳过已完成的样本）。
- MCQ 打分是离线的（选项字母匹配）；只有 open 题打分需要 judge 模型。
- `assets/` 里的图（`teaser.png`、`framework.png`）来自 UFO 论文。

## 引用

```bibtex
@inproceedings{ufo2026,
  title     = {Do Vision and Text Cues Exhibit Evidential Coupling?
               UFO: A Benchmark for Compositional Multimodal Reasoning in Unified Models},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
