from __future__ import annotations

import dspy


class SemanticClassificationSig(dspy.Signature):
    semantic_search: list[str] = dspy.InputField(
        description="Top semantically associated transcripts with associated codes."
    )
    line_to_analyze: str = dspy.InputField()
    prev_lines: list[str] = dspy.InputField()
    category_classification: str = dspy.OutputField(
        description=(
            "What categories would you classify line_to_analyze under. "
            "Be critical of whether semantic_search applies."
        )
    )


def classify_from_examples(
    *, semantic_search: list[str], line_to_analyze: str, prev_lines: list[str]
) -> str:
    predictor = dspy.Predict(SemanticClassificationSig)
    result = predictor(
        semantic_search=semantic_search,
        line_to_analyze=line_to_analyze,
        prev_lines=prev_lines,
    )
    print(f"""Model Classification: {len(str(result.category_classification))}""")
    return str(result.category_classification)
