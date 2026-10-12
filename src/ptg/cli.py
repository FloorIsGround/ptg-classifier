from __future__ import annotations

import argparse
from dataclasses import replace

from dotenv import load_dotenv

from ptg.config import load_settings
from ptg.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="ptg")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-index", help="Build vector index + metadata artifacts")
    p_build.add_argument("--semester", required=True, help="e.g. S or F")
    p_build.add_argument("--teams", default="1,2")
    p_build.add_argument("--meets", default="1,2,3")

    p_infer = sub.add_parser("infer", help="Stage C: classify one sheet and write raw results")
    p_infer.add_argument("--team", type=int, required=True)
    p_infer.add_argument("--meet", type=int, required=True)
    p_infer.add_argument("--semester", required=True)
    p_infer.add_argument("--embed-workers", type=int, default=1)
    p_infer.add_argument("--embed-batch-size", type=int, default=32)
    p_infer.add_argument("--lm-workers", type=int, default=4, help="Concurrent LM Studio requests")
    p_infer.add_argument("--lm-timeout", type=float, default=120.0)
    p_infer.add_argument("--pilot-row-limit", type=int, default=None,
                         help="Use first N nonblank rows in an isolated *_pilot results directory")

    p_export = sub.add_parser("export", help="Stage D: aggregate raw results into wide sheet")
    p_export.add_argument("--team", type=int, required=True)
    p_export.add_argument("--meet", type=int, required=True)
    p_export.add_argument("--semester", required=True)
    p_export.add_argument("--pilot", type=int, metavar="N", default=None,
                          help="Read results of a --pilot-row-limit N run")

    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    settings = load_settings(args.config)

    if args.cmd == "build-index":
        from ptg.pipeline.run import build_index

        teams = [int(x) for x in args.teams.split(",") if x.strip()]
        meets = [int(x) for x in args.meets.split(",") if x.strip()]
        build_index(settings, semester=args.semester, teams=teams, meets=meets)

    elif args.cmd == "infer":
        from ptg.pipeline.run import infer_sheet

        if args.pilot_row_limit is not None:
            settings = replace(
                settings,
                artifacts_dir=settings.artifacts_dir.with_name(
                    f"{settings.artifacts_dir.name}_pilot{args.pilot_row_limit}"
                ),
            )
        infer_sheet(
            settings,
            team=args.team,
            meet=args.meet,
            semester=args.semester,
            embed_workers=args.embed_workers,
            embed_batch_size=args.embed_batch_size,
            lm_workers=args.lm_workers,
            lm_timeout=args.lm_timeout,
            pilot_row_limit=args.pilot_row_limit,
        )

    elif args.cmd == "export":
        from ptg.pipeline.run import export_sheet

        if args.pilot is not None:
            if args.pilot < 1:
                parser.error("--pilot requires a positive row limit")
            settings = replace(
                settings,
                artifacts_dir=settings.artifacts_dir.with_name(
                    f"{settings.artifacts_dir.name}_pilot{args.pilot}"
                ),
            )
        export_sheet(
            settings,
            team=args.team,
            meet=args.meet,
            semester=args.semester,
        )
