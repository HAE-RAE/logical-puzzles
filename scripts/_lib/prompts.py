"""Reusable prompt constants for distillation/training scripts."""

# Strict <think>...</think> + Final answer format used by distill batch
# preparation and cost-probe scripts. The student model is also trained on
# this format so the reasoning trace is consumable by the eval pipeline.
THINK_FORMAT_INSTRUCTION = (
    "\n\n### Response format (STRICTLY MANDATORY — your response is invalid otherwise)\n"
    "You MUST produce an extensive, thorough chain-of-thought:\n"
    "  - Identify EVERY row from EVERY table that is relevant to the question.\n"
    "  - Show EVERY arithmetic step explicitly (do not skip intermediate calculations).\n"
    "  - State each filtering or grouping criterion before applying it.\n"
    "  - Aim for a detailed, fully verifiable trace (typically 500-1500 reasoning tokens). "
    "A short answer is INSUFFICIENT — show your work.\n"
    "Even if a reference outline is provided in the user message, "
    "you MUST STILL produce the full step-by-step <think> trace. "
    "Responses without proper <think>...</think> reasoning are REJECTED.\n"
    "\n"
    "Output structure (exactly):\n"
    "<think>\n"
    "  ...detailed multi-step reasoning here, listing each row, each calculation...\n"
    "</think>\n"
    "Final answer: <the answer>\n"
    "Do not write anything after this line."
)
