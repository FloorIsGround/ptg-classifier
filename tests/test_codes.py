from ptg.classify.codebook import load_default_codebook
from ptg.util.codes import canonicalize_codes, parse_codes


def test_parse_codes_dedup_and_strip():
    assert parse_codes(" Optimism, optimism , Building bonds ") == ["Optimism", "Building bonds"]


def test_parse_codes_empty():
    assert parse_codes("") == []


def test_canonicalize_codes_maps_aliases_and_preserves_order():
    codebook = load_default_codebook()

    assert canonicalize_codes(
        ["Question to Group", "Suggestion", "Accurate Team-assessment", "Logistics"],
        codebook,
    ) == [
        "Questions to group",
        "Suggestion / Change catalyst",
        "Accurate team assessment",
        "Logistics",
    ]


def test_canonicalize_codes_deduplicates_after_alias_mapping():
    codebook = load_default_codebook()

    assert canonicalize_codes(["Suggestion", "Suggestion / Change catalyst"], codebook) == [
        "Suggestion / Change catalyst"
    ]


def test_canonicalize_codes_raises_for_unknown_labels():
    codebook = load_default_codebook()

    try:
        canonicalize_codes(["Not A Real Code"], codebook)
    except ValueError as exc:
        assert "Unknown code label" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown code label")


def test_canonicalize_codes_can_drop_unknown_legacy_labels_for_index_migration():
    codebook = load_default_codebook()

    assert canonicalize_codes(
        ["Service Orientation", "Logistics", "Not A Real Code"],
        codebook,
        on_unknown="drop",
    ) == ["Logistics"]
