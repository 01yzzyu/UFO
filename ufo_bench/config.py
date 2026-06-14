"""Static configuration: the UFO taxonomy and the four reasoning protocols.

These mirror the paper exactly so that result tables line up with the
manuscript. Do not rename these without also updating the paper tables.
"""

# ---------------------------------------------------------------------------
# Taxonomy: 3 state-transition categories x 10 tasks
# ---------------------------------------------------------------------------
CATEGORIES = [
    "state_determination",
    "state_reconstruction",
    "state_augmentation",
]

CATEGORY_DISPLAY = {
    "state_determination": "State Determination",
    "state_reconstruction": "State Reconstruction",
    "state_augmentation": "State Augmentation",
}

# Ordered task list per category (paper order).
TASKS_BY_CATEGORY = {
    "state_determination": ["Hybridisation", "Chemical", "Multi-table", "Multi-view"],
    "state_reconstruction": ["Inpainting", "Exo-to-Ego", "Jigsaw"],
    "state_augmentation": ["Geometric", "Logical", "Physics"],
}

# Flat list of all 10 tasks.
ALL_TASKS = [t for c in CATEGORIES for t in TASKS_BY_CATEGORY[c]]

# Reverse lookup: task -> category.
TASK_TO_CATEGORY = {
    t: c for c, ts in TASKS_BY_CATEGORY.items() for t in ts
}

# ---------------------------------------------------------------------------
# Reasoning protocols (the four inference schedules in the paper)
# ---------------------------------------------------------------------------
#   direct  : answer from the input images only (no intermediate cue)
#   textual : answer conditioned on a generated *textual* cue
#   visual  : answer conditioned on a generated *visual* cue (image)
#   joint   : answer conditioned on both textual and visual cues
PROTOCOLS = ["direct", "textual", "visual", "joint"]

PROTOCOL_NEEDS_TEXT_CUE = {"direct": False, "textual": True, "visual": False, "joint": True}
PROTOCOL_NEEDS_IMAGE_CUE = {"direct": False, "textual": False, "visual": True, "joint": True}

QUESTION_TYPES = ["mcq", "open"]
