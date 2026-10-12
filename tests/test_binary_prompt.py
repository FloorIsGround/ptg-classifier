from ptg.classify.binary_classifier import BinaryClassificationJob, build_binary_prompt
from ptg.classify.codebook import load_default_codebook


def test_binary_prompt_includes_codebook_context_and_dataset_examples():
    codebook = load_default_codebook()
    entry = codebook.entry_by_label("Emotional awareness")
    job = BinaryClassificationJob(
        row_index=7,
        entry=entry,
        line_to_analyze="I feel really nervous about presenting this.",
        prev_lines=["Ari: We can split the slides.", "Sam: I can go first."],
        retrieved_examples=["Transcript: The fictional exercise leaves me feeling apprehensive.\nCodes: Emotional awareness"],
    )

    system_prompt, user_prompt = build_binary_prompt(job)

    assert system_prompt == "You are a careful assistant that labels transcript lines with existing codes."
    assert "Target label: Emotional awareness" in user_prompt
    assert "Category: Self-Awareness" in user_prompt
    assert "Definition: Understanding of one's own attitudes or feelings." in user_prompt
    assert "Codebook examples where this IS the label:" in user_prompt
    assert "- The fictional exercise leaves me feeling apprehensive." in user_prompt
    assert "Codebook boundary examples where this is NOT the label:" in user_prompt
    assert "- The fictional teammate appears apprehensive. (reading another's emotion; belongs to Empathy)" in user_prompt
    assert "Retrieved previously-coded transcript examples for this label:" in user_prompt
    assert "Transcript: The fictional exercise leaves me feeling apprehensive." in user_prompt
    assert "Previous lines:" in user_prompt
    assert "- Ari: We can split the slides." in user_prompt
    assert "Current line to analyze:" in user_prompt
    assert "I feel really nervous about presenting this." in user_prompt
    assert 'Does the current line exhibit "Emotional awareness"?' in user_prompt
    assert 'Answer ONLY with the single word "yes" or "no". No explanation.' in user_prompt


def test_binary_prompt_uses_none_when_optional_context_is_empty():
    codebook = load_default_codebook()
    entry = codebook.entry_by_label("Logistics")
    job = BinaryClassificationJob(
        row_index=3,
        entry=entry,
        line_to_analyze="How many pages does this need to be?",
        prev_lines=[],
        retrieved_examples=[],
    )

    _, user_prompt = build_binary_prompt(job)

    assert "Previous lines:\n(none)" in user_prompt
    assert "Retrieved previously-coded transcript examples for this label:\n(none found)" in user_prompt
    assert "Definition: Speaker discusses expectations and information associated with the logistics" in user_prompt


def test_binary_prompt_does_not_include_unrelated_label_examples():
    codebook = load_default_codebook()
    entry = codebook.entry_by_label("Directive")
    job = BinaryClassificationJob(
        row_index=5,
        entry=entry,
        line_to_analyze="Record the fictional workshop agenda.",
        prev_lines=[],
        retrieved_examples=[],
    )

    _, user_prompt = build_binary_prompt(job)

    assert "Target label: Directive" in user_prompt
    assert "Record the fictional workshop agenda" in user_prompt
    assert "The fictional exercise leaves me feeling apprehensive." not in user_prompt
    assert "Understanding of one's own attitudes or feelings." not in user_prompt
