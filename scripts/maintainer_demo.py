"""Expose a fictional pipeline walkthrough without opening network connections.

Good: run from the checkout to inspect generated config, prompts and results.
Bad: use oracle answers to assess model quality, or replace fixtures with study data.
This wrapper reuses synthetic_demo.py; it does not implement a model server.
"""
from __future__ import annotations

import json
import socket
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd
import yaml

from ptg.classify import binary_classifier
from ptg.config import load_settings
from ptg.pipeline import run as pipeline
from synthetic_demo import ROOT, demo


def main() -> None:
    settings_seen = []
    jobs_seen = []
    requests_seen = []
    real_build = pipeline.build_index
    real_prompt = binary_classifier.build_binary_prompt
    real_transport = httpx.MockTransport

    def capture_build(settings, **kwargs):
        settings_seen.append(settings)
        return real_build(settings, **kwargs)

    def capture_prompt(job):
        if not jobs_seen and job.row_index == 0 and job.label_name == "Initiative":
            jobs_seen.append(job)
        return real_prompt(job)

    def capture_transport(handler):
        def respond(request):
            body = json.loads(request.content)
            if not requests_seen and "Target label: Initiative\n" in body["messages"][-1]["content"]:
                requests_seen.append({"url": str(request.url), "body": body})
            return handler(request)
        return real_transport(respond)

    def refuse_network(*args, **kwargs):
        raise RuntimeError("Maintainer demo refuses network connections")

    with patch.object(pipeline, "build_index", capture_build), \
            patch.object(binary_classifier, "build_binary_prompt", capture_prompt), \
            patch.object(httpx, "MockTransport", capture_transport), \
            patch.object(socket.socket, "connect", refuse_network), \
            patch.object(socket.socket, "connect_ex", refuse_network), \
            patch.object(socket, "create_connection", refuse_network):
        result = demo()

    settings, job = settings_seen[0], jobs_seen[0]
    out = ROOT / result["output_directory"]
    relative = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()
    cfg = {
        "io": {"data_dir": relative(out), "artifacts_dir": relative(settings.artifacts_dir),
               "index_artifacts_dir": relative(settings.index_artifacts_dir)},
        "excel": {"workbook": relative(settings.workbook_path), "sheet_template": settings.sheet_template},
        "codebook": {"path": relative(settings.codebook_path), "active_labels": list(settings.active_labels),
                     "excluded_source_labels": list(settings.excluded_source_labels)},
        "context": {"prev_k": settings.prev_k},
        "embedding": {"model": settings.embed_model},
        "retrieval": {"top_k": settings.retrieval_top_k, "index_backend": settings.index_backend,
                      "run_prefix": settings.run_prefix},
        "llm": {"api_base": settings.lm_api_base, "model": settings.lm_model,
                "binary_max_retries": settings.binary_max_retries,
                "temperature": settings.lm_temperature, "top_p": settings.lm_top_p,
                "max_tokens": settings.lm_max_tokens, "seed": settings.lm_seed,
                "reasoning_effort": settings.lm_reasoning_effort,
                "chat_template_kwargs": settings.lm_chat_template_kwargs},
    }
    # The original demo does not use data_dir; point this informational field at
    # the fictional output too. Every operational setting must round-trip.
    config_path = out / "demo-config.yaml"
    config_path.write_text(
        "# SYNTHETIC ONLY. Export/scoring work offline; infer/build-index need test doubles.\n"
        + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    loaded = load_settings(config_path)
    for field in fields(settings):
        if field.name == "data_dir":
            continue
        expected, actual = getattr(settings, field.name), getattr(loaded, field.name)
        if isinstance(expected, Path):
            assert expected.resolve() == actual.resolve(), field.name
        else:
            assert expected == actual, field.name

    system, rag_prompt = real_prompt(job)
    _, no_rag_prompt = real_prompt(replace(job, retrieved_examples=[]))
    (out / "prompt-with-retrieval.txt").write_text(rag_prompt + "\n", encoding="utf-8")
    (out / "prompt-without-retrieval.txt").write_text(no_rag_prompt + "\n", encoding="utf-8")
    (out / "request-example.json").write_text(json.dumps(requests_seen[0], indent=2) + "\n")
    assert requests_seen[0]["body"]["messages"] == [
        {"role": "system", "content": system}, {"role": "user", "content": rag_prompt}]

    raw_path = settings.artifacts_dir / f"{settings.run_prefix}raw_results.xlsx"
    wide_path = settings.artifacts_dir / f"{settings.run_prefix}wide_results.xlsx"
    raw_sheet = "Team 1 Time 1 (S26) Run 0"
    wide_sheet = "Team 1 Time 1 (S26)"
    raw = pd.read_excel(raw_path, sheet_name=raw_sheet, keep_default_na=False)
    selected = raw[(raw.row_index == job.row_index) & (raw.label == job.label_name)]
    assert len(selected) == 1 and selected.iloc[0]["parsed"] == "yes"
    wide = pd.read_excel(wide_path, sheet_name=wide_sheet, keep_default_na=False)
    assert len(wide) == 3 and wide.iloc[0][job.label_name] == "yes"
    walkthrough = {
        "kind": "SYNTHETIC plumbing walkthrough; no live model evaluated",
        "config": relative(config_path), "input_workbook": relative(settings.workbook_path),
        "target_sheet": wide_sheet, "dataframe_row_index": job.row_index, "excel_row": job.row_index + 2,
        "input_text": job.line_to_analyze, "target_label": job.label_name,
        "previous_lines": job.prev_lines, "retrieved_examples": len(job.retrieved_examples),
        "raw_workbook": relative(raw_path), "raw_sheet": raw_sheet,
        "raw_decision": selected[["row_index", "label", "raw_response", "parsed"]].to_dict("records")[0],
        "wide_workbook": relative(wide_path), "wide_sheet": wide_sheet,
        "wide_decision": wide.iloc[0][job.label_name],
        "no_retrieval_prompt_only": True,
        "note": "The no-retrieval file is a prompt illustration, not a second scored inference arm.",
        "network_guard": "socket connect/connect_ex/create_connection refused during demo execution",
    }
    (out / "walkthrough.json").write_text(json.dumps(walkthrough, indent=2) + "\n")
    print(json.dumps({**result, "walkthrough": relative(out / "walkthrough.json"),
                      "effective_config": relative(config_path),
                      "network_guard": True, "config_round_trip": True}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
