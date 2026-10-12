from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from ptg.classify.codebook import DEFAULT_CODEBOOK_PATH, load_codebook
from ptg.domain.models import Codebook


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    artifacts_dir: Path
    workbook_path: Path
    sheet_template: str
    prev_k: int
    embed_model: str
    retrieval_top_k: int
    index_backend: str
    run_prefix: str
    lm_api_base: str | None
    lm_model: str | None
    binary_max_retries: int
    # Codebook / study-arm configuration. index_artifacts_dir is where the shared
    # retrieval index lives; artifacts_dir is the per-arm result namespace.
    codebook_path: Path
    active_labels: tuple[str, ...]
    index_artifacts_dir: Path
    # Optional per-run reasoning control; defaults keep every field with a default last.
    lm_reasoning_effort: str | None = None
    excluded_source_labels: tuple[str, ...] = ()
    lm_temperature: float = 0.0
    lm_top_p: float | None = None
    lm_max_tokens: int = 10
    lm_chat_template_kwargs: dict[str, bool] | None = None
    lm_seed: int | None = None


def _expand_env(val):
    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        key = val[2:-1]
        return os.getenv(key)
    return val


def _resolve_active_labels(raw: object, codebook: Codebook) -> tuple[str, ...]:
    """Validate configured active labels against the configured codebook.

    Labels are canonicalised (aliases allowed), must be unique after
    canonicalisation, and must exist in the codebook. Declared order is kept
    because downstream aggregation and reporting use it.
    """
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("codebook.active_labels must be a non-empty list of label names.")

    resolved: list[str] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            raise ValueError("codebook.active_labels contains a blank label.")
        canonical = codebook.canonicalize_label(text)
        if canonical is None:
            raise ValueError(
                f"Unknown active label {text!r}: not present in codebook {codebook.version!r}."
            )
        resolved.append(canonical)

    if len(resolved) != len(set(resolved)):
        duplicates = sorted({label for label in resolved if resolved.count(label) > 1})
        raise ValueError(f"codebook.active_labels contains duplicate labels: {duplicates}")

    return tuple(resolved)


def load_settings(path: str | os.PathLike = "config/default.yaml") -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Expand ${ENV} placeholders (very small, on-purpose)
    def rec(x):
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rec(v) for v in x]
        return _expand_env(x)

    cfg = rec(raw)

    io_cfg = cfg.get("io", {}) or {}
    data_dir = Path(io_cfg.get("data_dir", "./data"))
    artifacts_dir = Path(io_cfg.get("artifacts_dir", "./artifacts"))
    # The retrieval index is shared between study arms; results are not.
    index_dir_raw = io_cfg.get("index_artifacts_dir")
    index_artifacts_dir = (
        Path(index_dir_raw) if index_dir_raw not in (None, "") else artifacts_dir
    )

    workbook = Path(cfg.get("excel", {}).get("workbook", "./data/data-transcripts.xlsx"))
    sheet_template = cfg.get("excel", {}).get(
        "sheet_template", "Team {team} Time {meet} ({semester}25)"
    )

    prev_k = int(cfg.get("context", {}).get("prev_k", 2))

    embed_model = str(cfg.get("embedding", {}).get("model", os.getenv("EMBED_MODEL", "")))

    retrieval = cfg.get("retrieval", {}) or {}
    retrieval_top_k = int(retrieval.get("top_k", 5))
    index_backend = str(retrieval.get("index_backend", "faiss"))
    run_prefix = str(retrieval.get("run_prefix", "run_"))

    llm = cfg.get("llm", {}) or {}
    if not isinstance(llm, dict):
        raise ValueError("llm configuration must be a mapping")
    lm_api_base = llm.get("api_base")
    lm_model = llm.get("model")
    binary_max_retries = int(llm.get("binary_max_retries", 2))
    reasoning_effort = llm.get("reasoning_effort")
    lm_reasoning_effort = None if reasoning_effort is None else str(reasoning_effort)
    lm_temperature = float(llm.get("temperature", 0.0))
    lm_top_p = None if llm.get("top_p") is None else float(llm["top_p"])
    lm_max_tokens = int(llm.get("max_tokens", 10))
    template_kwargs = llm.get("chat_template_kwargs")
    lm_seed = None if llm.get("seed") is None else int(llm["seed"])
    if not 0 <= lm_temperature <= 2 or lm_top_p is not None and not 0 < lm_top_p <= 1:
        raise ValueError("llm temperature/top_p outside valid range")
    if lm_max_tokens < 1 or lm_max_tokens > 2048:
        raise ValueError("llm.max_tokens must be between 1 and 2048")
    if lm_seed is not None and not 0 <= lm_seed < 2**32:
        raise ValueError("llm.seed must be a nonnegative 32-bit integer")
    if template_kwargs is not None and (
        not isinstance(template_kwargs, dict)
        or any(k not in ("enable_thinking", "low_effort") or not isinstance(v, bool)
               for k, v in template_kwargs.items())
    ):
        raise ValueError("llm.chat_template_kwargs accepts only boolean thinking controls")

    codebook_cfg = cfg.get("codebook", {}) or {}
    if not isinstance(codebook_cfg, dict):
        raise ValueError("codebook configuration must be a mapping")
    codebook_path = Path(codebook_cfg.get("path") or DEFAULT_CODEBOOK_PATH)
    codebook = load_codebook(codebook_path)
    active_raw = codebook_cfg.get("active_labels")
    # Default: every label in the configured codebook, in codebook order.
    active_labels = (
        tuple(codebook.allowed_labels)
        if active_raw is None
        else _resolve_active_labels(active_raw, codebook)
    )
    excluded_raw = codebook_cfg.get("excluded_source_labels", [])
    if not isinstance(excluded_raw, list) or any(not isinstance(x, str) or not x.strip()
                                                  for x in excluded_raw):
        raise ValueError("codebook.excluded_source_labels must be a list of nonblank names")
    excluded_keys = [" ".join(x.casefold().split()) for x in excluded_raw]
    if len(excluded_keys) != len(set(excluded_keys)):
        raise ValueError("Duplicate excluded source labels")
    if any(codebook.canonicalize_label(label) is not None for label in excluded_raw):
        raise ValueError("An excluded source label overlaps an active codebook label")

    return Settings(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        workbook_path=workbook,
        sheet_template=sheet_template,
        prev_k=prev_k,
        embed_model=embed_model,
        retrieval_top_k=retrieval_top_k,
        index_backend=index_backend,
        run_prefix=run_prefix,
        lm_api_base=lm_api_base,
        lm_model=lm_model,
        binary_max_retries=binary_max_retries,
        codebook_path=codebook_path,
        active_labels=active_labels,
        index_artifacts_dir=index_artifacts_dir,
        lm_reasoning_effort=lm_reasoning_effort,
        excluded_source_labels=tuple(excluded_raw),
        lm_temperature=lm_temperature,
        lm_top_p=lm_top_p,
        lm_max_tokens=lm_max_tokens,
        lm_chat_template_kwargs=template_kwargs,
        lm_seed=lm_seed,
    )
