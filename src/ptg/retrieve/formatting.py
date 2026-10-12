from __future__ import annotations


def format_retrieved_example(meta: dict) -> str:
    """Legacy formatter matching the old dspy_rag.py output style."""
    sheet = meta.get("sheet", meta.get("sheet_name", ""))
    speaker = meta.get("speaker", "")
    transcript = meta.get("transcript", "")
    prev_context = meta.get("prev_context", "")
    codes_key = meta.get("codes_key", "")
    score = meta.get("score")
    score_s = "" if score is None else f" cos={float(score):.4f}"

    text = ""
    text += f"{sheet} Speaker: {speaker}{score_s}\n\n"
    text += "Current Line: " + str(transcript) + "\n\n"
    text += "Previous Lines:\n" + str(prev_context) + "\n\n"
    text += "Associated codes: " + str(codes_key) + "\n"
    return text
