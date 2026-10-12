"""Configurable codebook + study-arm settings: shared index, isolated results."""

from __future__ import annotations

from pathlib import Path

import pytest

from ptg.classify.codebook import DEFAULT_CODEBOOK_PATH, load_codebook, load_default_codebook
from ptg.config import load_settings
from ptg.storage.files import artifact_paths

REPO = Path(__file__).resolve().parents[1]
EXTENDED_CODEBOOK = REPO / "config/codebooks/f25_extended_DRAFT.yaml"
STUDY_21 = REPO / "config/study-21.yaml"
STUDY_23 = REPO / "config/study-23.yaml"

CLEAN_21_LABELS = (
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
)
DRAFT_LABELS = ("Communication", "Service orientation")
EXTENDED_23_LABELS = CLEAN_21_LABELS + DRAFT_LABELS

# Exact source text from docs/Codebook for Content Analysis.pdf (Social Awareness row).
SERVICE_ORIENTATION_SOURCE_TEXT = "Offering assistance to the team as a whole"


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "study.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_default_settings_fall_back_to_default_codebook_and_full_label_order(tmp_path):
    config = _write_config(tmp_path, "io:\n  artifacts_dir: ./artifacts\n")
    settings = load_settings(config)

    assert settings.codebook_path == DEFAULT_CODEBOOK_PATH
    assert settings.active_labels == tuple(load_default_codebook().allowed_labels)
    assert len(settings.active_labels) == 21
    # No io.index_artifacts_dir -> result namespace doubles as index namespace.
    assert settings.index_artifacts_dir == settings.artifacts_dir


def test_extended_codebook_keeps_the_21_core_entries_unchanged():
    clean = load_default_codebook()
    extended = load_codebook(EXTENDED_CODEBOOK)

    assert extended.entries[:21] == clean.entries
    assert tuple(extended.allowed_labels) == EXTENDED_23_LABELS
    assert len(extended.entries) == 23


def test_extended_codebook_adds_only_the_two_draft_labels():
    extended = load_codebook(EXTENDED_CODEBOOK)
    added = extended.entries[21:]

    assert tuple(entry.label for entry in added) == DRAFT_LABELS
    assert added[0].id == "communication"
    assert added[1].id == "service_orientation"


def test_communication_has_no_invented_definition_or_examples():
    entry = load_codebook(EXTENDED_CODEBOOK).entry_by_label("Communication")

    assert entry.definition == ""
    assert entry.positive_examples == ()
    assert entry.negative_examples == ()


def test_service_orientation_uses_exact_pdf_source_text():
    entry = load_codebook(EXTENDED_CODEBOOK).entry_by_label("Service orientation")

    assert entry.definition == SERVICE_ORIENTATION_SOURCE_TEXT
    assert entry.positive_examples == ()


def test_extended_codebook_notes_cite_the_draft_and_source():
    import yaml

    raw = yaml.safe_load(EXTENDED_CODEBOOK.read_text(encoding="utf-8"))
    notes = "\n".join(raw["source_notes"])

    assert "DRAFT" in notes
    assert "Codebook for Content Analysis.pdf" in notes


def test_study_23_arm_activates_all_23_labels(tmp_path):
    settings = load_settings(STUDY_23)

    assert Path(settings.codebook_path).resolve() == EXTENDED_CODEBOOK.resolve()
    assert settings.active_labels == EXTENDED_23_LABELS
    assert len(settings.active_labels) == 23


def test_study_21_arm_excludes_retired_labels_from_retrieval():
    settings = load_settings(STUDY_21)

    assert Path(settings.codebook_path).resolve() == DEFAULT_CODEBOOK_PATH.resolve()
    assert settings.active_labels == CLEAN_21_LABELS
    assert "Communication" not in settings.active_labels
    assert "Service orientation" not in settings.active_labels
    assert settings.excluded_source_labels == ("Communication", "Service Orientation")
    assert len(load_codebook(settings.codebook_path).entries) == 21


def test_retired_and_draft_arms_do_not_share_retrieval_index():
    arm21 = load_settings(STUDY_21)
    arm23 = load_settings(STUDY_23)

    assert arm21.index_artifacts_dir != arm23.index_artifacts_dir
    assert arm21.run_prefix != arm23.run_prefix
    assert arm21.artifacts_dir != arm23.artifacts_dir

    paths21 = artifact_paths(arm21.index_artifacts_dir, arm21.run_prefix)
    paths23 = artifact_paths(arm23.index_artifacts_dir, arm23.run_prefix)
    assert paths21.index_path != paths23.index_path
    assert paths21.metas_path != paths23.metas_path


def test_excluded_source_labels_cannot_overlap_approved_codebook(tmp_path):
    config = _write_config(tmp_path, "codebook:\n  excluded_source_labels: [Initiative]\n")
    with pytest.raises(ValueError, match="overlaps"):
        load_settings(config)


def test_study_arms_render_explicit_years_without_implicit_2025():
    for config in (STUDY_21, STUDY_23):
        settings = load_settings(config)
        assert "S25" not in settings.sheet_template
        assert settings.sheet_template.startswith("Team {team} Time {meet}")
        rendered = [
            settings.sheet_template.format(team=t, meet=m, semester="F25")
            for t in (1, 2)
            for m in (1, 2)
        ]
        assert rendered == [
            "Team 1 Time 1 (F25)",
            "Team 1 Time 2 (F25)",
            "Team 2 Time 1 (F25)",
            "Team 2 Time 2 (F25)",
        ]
        assert settings.sheet_template.format(team=1, meet=1, semester="S26") == (
            "Team 1 Time 1 (S26)"
        )
        assert settings.sheet_template.format(team=2, meet=1, semester="FA26") == (
            "Team 2 Time 1 (FA26)"
        )


def test_active_labels_must_be_unique(tmp_path):
    config = _write_config(
        tmp_path,
        f'codebook:\n  path: "{EXTENDED_CODEBOOK}"\n'
        "  active_labels:\n    - Initiative\n    - initiative\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_settings(config)


def test_active_labels_must_exist_in_codebook(tmp_path):
    config = _write_config(
        tmp_path,
        f'codebook:\n  path: "{EXTENDED_CODEBOOK}"\n'
        "  active_labels:\n    - Initiative\n    - Definitely not a label\n",
    )
    with pytest.raises(ValueError, match="Unknown active label"):
        load_settings(config)


def test_active_labels_reject_an_empty_list(tmp_path):
    config = _write_config(
        tmp_path,
        f'codebook:\n  path: "{EXTENDED_CODEBOOK}"\n  active_labels: []\n',
    )
    with pytest.raises(ValueError, match="non-empty"):
        load_settings(config)


def test_active_labels_default_to_configured_codebook_order(tmp_path):
    config = _write_config(tmp_path, f'codebook:\n  path: "{EXTENDED_CODEBOOK}"\n')
    settings = load_settings(config)

    assert settings.active_labels == EXTENDED_23_LABELS


def test_active_labels_resolve_aliases_to_canonical_names(tmp_path):
    config = _write_config(
        tmp_path,
        f'codebook:\n  path: "{EXTENDED_CODEBOOK}"\n'
        "  active_labels:\n    - Question to Group\n    - logistics\n",
    )
    settings = load_settings(config)

    assert settings.active_labels == ("Questions to group", "Logistics")


def test_index_artifacts_dir_defaults_to_artifacts_dir(tmp_path):
    config = _write_config(tmp_path, "io:\n  artifacts_dir: ./out\n  index_artifacts_dir:\n")
    settings = load_settings(config)

    assert settings.index_artifacts_dir == Path("./out")
    assert settings.index_artifacts_dir == settings.artifacts_dir


def test_index_artifacts_dir_read_from_io(tmp_path):
    config = _write_config(
        tmp_path,
        "io:\n  artifacts_dir: ./out\n  index_artifacts_dir: ./shared\n",
    )
    settings = load_settings(config)

    assert settings.index_artifacts_dir == Path("./shared")
    assert settings.artifacts_dir == Path("./out")


def test_reasoning_effort_defaults_to_none(tmp_path):
    config = _write_config(tmp_path, "io:\n  artifacts_dir: ./artifacts\n")

    assert load_settings(config).lm_reasoning_effort is None


def test_study_arms_disable_thinking_for_short_binary_outputs():
    for config in (STUDY_21, STUDY_23):
        assert load_settings(config).lm_reasoning_effort == "none"
