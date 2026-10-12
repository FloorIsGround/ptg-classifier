from __future__ import annotations

import os
from typing import Any

import dspy


def configure_dspy_lm(api_base: str | None, model: str | None) -> None:
    if not api_base or not model:
        raise RuntimeError("LM_STUDIOS_HOST / LM_MODEL not configured")
    lm = dspy.LM(model=model, api_base=api_base, api_key="")
    dspy.configure(lm=lm)


class ClassificationSig(dspy.Signature):
    allowed_code_ids: list[str] = dspy.InputField(description="Allowed code IDs")
    retrieved_examples: list[str] = dspy.InputField(description="Top retrieved examples with codes")
    line_to_analyze: str = dspy.InputField()
    prev_lines: list[str] = dspy.InputField()

    # IMPORTANT: enforce JSON list of IDs in post-processing.
    code_ids_json: str = dspy.OutputField(
        description="Return ONLY a JSON list of code IDs (strings) chosen from allowed_code_ids."
    )


def predict_code_ids(
    *,
    allowed_code_ids: list[str],
    retrieved_examples: list[str],
    line_to_analyze: str,
    prev_lines: list[str],
) -> str:
    predictor = dspy.Predict(ClassificationSig)
    result: Any = predictor(
        allowed_code_ids=allowed_code_ids,
        retrieved_examples=retrieved_examples,
        line_to_analyze=line_to_analyze,
        prev_lines=prev_lines,
    )
    return str(result.code_ids_json)
