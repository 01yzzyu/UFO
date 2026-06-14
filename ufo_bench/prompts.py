"""Prompt templates, aligned with the UFO paper appendix.

Three families:
  1. Cue generation (textual / visual) — produce the intermediate future-state cue.
  2. Answering           — answer the question, optionally conditioned on cues.
  3. Judging             — score open-ended answers and intermediate cue quality.
"""

# ---------------------------------------------------------------------------
# 1. Cue generation (matches "Prompts for Response Generation" in the paper)
# ---------------------------------------------------------------------------
TEXT_CUE_GEN = (
    "My specific question is:\n{question}\n\n"
    "Your task is to generate a key textual cue based on the provided reference "
    "information.\n"
    "This textual cue should:\n"
    "Identify key entities, attributes, and relationships pertinent to the query. "
    "Focus on critical visual features (e.g., structure, spatial layout, states) "
    "that drive reasoning. Ensure the response is concise and informative, "
    "limited to 1-2 sentences.\n\n"
    "The generated text cue should serve as an abstract, language-only substitute "
    "for a visual cue, helping to answer the question above."
)

# For models that generate images directly, this is the visual-cue instruction.
IMAGE_CUE_GEN = (
    "My specific question is: {question}\n\n"
    "Your task is to generate an image based on the provided reference images. "
    "This generated image should serve as a visual cue to help answer the "
    "question above."
)

# For text-only / VLM models that cannot synthesize images, we ask for a
# detailed description that a downstream image generator can render.
IMAGE_CUE_DESCRIPTION = (
    "My specific question is: {question}\n\n"
    "Your task is to describe an image that should be generated to help answer "
    "the question based on the provided reference images. Provide a detailed "
    "visual description of this hypothetical image so that an image generation "
    "model can create it. The description should serve as a visual cue."
)

# ---------------------------------------------------------------------------
# 2. Answering
# ---------------------------------------------------------------------------
ANSWER_SYSTEM = (
    "You are a precise assistant. Provide a short, clear answer to the question. "
    "Do NOT output your reasoning process, chain of thought, or explanation. "
    "Just give the final answer."
)

MCQ_INSTRUCTION = (
    "\nAnswer ONLY with the option letter (e.g., A, B, C, or D). "
    "Do not explain or provide reasoning."
)

OPEN_INSTRUCTION = (
    "\nPlease only provide a short, clear, and direct answer. "
    "Do not explain your reasoning."
)


def format_choices(choices):
    """Render an MCQ choices dict into a prompt block."""
    if not choices:
        return ""
    lines = ["\nChoices:"]
    for k in ("A", "B", "C", "D"):
        if k in choices:
            lines.append(f"{k}: {choices[k]}")
    return "\n".join(lines) + "\n"


def build_question_block(question, choices):
    """The shared question text used by every protocol."""
    instr = MCQ_INSTRUCTION if choices else OPEN_INSTRUCTION
    return f"Question: {question}{format_choices(choices)}{instr}"


def build_answer_prompt(question, choices, text_cue=None, has_image_cue=False):
    """Assemble the final answering prompt for a given protocol.

    The presence of text_cue / has_image_cue selects the protocol:
      none           -> direct
      text only      -> textual
      image only     -> visual
      text + image   -> joint
    """
    prefix = ""
    if text_cue:
        prefix += f"Text Cue: {text_cue}\n"
    if has_image_cue:
        prefix += "Reference the provided visual cue image (the last image).\n"
    if prefix:
        prefix += "\n"
    return prefix + build_question_block(question, choices)


# ---------------------------------------------------------------------------
# 3. Judging
# ---------------------------------------------------------------------------
# Open-ended answer correctness (binary, factual).
OPEN_ANSWER_JUDGE = (
    "You are a strict evaluator judging whether a model's answer to a question "
    "is correct, given the reference answer.\n\n"
    "Question: {question}\n"
    "Reference Answer: {gt}\n"
    "Model Answer: {pred}\n\n"
    "Rules:\n"
    "- Output 1 if the Model Answer is factually equivalent to the Reference "
    "Answer (paraphrasing, formatting, units, or extra correct detail are fine).\n"
    "- Output 0 if it is wrong, contradictory, missing the key fact, or empty.\n\n"
    "Output ONLY one character: 1 or 0."
)

# Text-cue correctness (binary), used by the cue-quality evaluator.
TEXT_CUE_JUDGE = (
    "You are a strict evaluator assessing an intermediate reasoning cue (text).\n"
    "Decide if the Candidate Cue provides the correct information required by the "
    "Question, consistent with the Ground Truth Cue.\n\n"
    "Question: {question}\n"
    "Ground Truth Cue: {gt}\n"
    "Candidate Cue: {pred}\n\n"
    "Output 1 if the Candidate is factually consistent and covers the essential "
    "information; output 0 if it contradicts, hallucinates, or misses key facts.\n"
    "Output ONLY one character: 1 or 0."
)

# Visual-cue correctness (binary, semantic), two images provided to the judge.
VISUAL_CUE_JUDGE = (
    "You are a strict visual evaluator. Verify whether the Candidate Cue image "
    "(Image 2) correctly visualizes the information needed for the Question, "
    "using the Ground Truth Cue image (Image 1) as reference.\n\n"
    "Question: {question}\n"
    "Image 1: Ground Truth Cue (reference)\n"
    "Image 2: Candidate Cue (generated)\n\n"
    "Ignore style/color/pixel differences. Focus on semantic content: same target "
    "object, correct location/layout. Output 1 if Image 2 captures the correct "
    "target; output 0 if it shows the wrong object, wrong location, or is empty.\n"
    "Output ONLY one character: 1 or 0."
)
