from ptg.classify.codebook import load_default_codebook


EXPECTED_LABELS = [
    "Emotional awareness",
    "Accurate self-assessment",
    "Lack of self-awareness",
    "Self-confidence",
    "Lack of self-confidence",
    "Adaptability",
    "Lack of adaptability",
    "Negative emotional outburst",
    "Initiative",
    "Optimism",
    "Empathy",
    "Accurate team assessment",
    "Developing others",
    "Conflict management",
    "Building bonds",
    "Breaking bonds",
    "Questions to group",
    "Facilitate",
    "Suggestion / Change catalyst",
    "Directive",
    "Logistics",
]


def test_load_default_codebook_returns_clean_codebook_labels_in_order():
    codebook = load_default_codebook()

    assert codebook.version == "shareable-synthetic-v1"
    assert codebook.allowed_labels == EXPECTED_LABELS
    assert len(codebook.entries) == 21


def test_codebook_entry_contains_definition_and_examples():
    codebook = load_default_codebook()
    entry = codebook.entry_by_label("Emotional awareness")

    assert entry.id == "emotional_awareness"
    assert entry.category == "Self-Awareness"
    assert entry.definition == "Understanding of one's own attitudes or feelings."
    assert "The fictional exercise leaves me feeling apprehensive." in entry.positive_examples
    assert any(example.text == "The fictional teammate appears apprehensive." for example in entry.negative_examples)


def test_legacy_or_provisional_labels_are_not_added_as_codes():
    codebook = load_default_codebook()

    assert "Service orientation" not in codebook.allowed_labels
    assert "Leveraging Diversity" not in codebook.allowed_labels
    assert "Communication" not in codebook.allowed_labels


def test_alias_canonicalization_maps_observed_f25_variants():
    codebook = load_default_codebook()

    assert codebook.canonicalize_label("Question to Group") == "Questions to group"
    assert codebook.canonicalize_label("Suggestion") == "Suggestion / Change catalyst"
    assert codebook.canonicalize_label("Accurate Team-assessment") == "Accurate team assessment"
    assert codebook.canonicalize_label("logistics") == "Logistics"
    assert codebook.canonicalize_label("Definitely Not A Code") is None
