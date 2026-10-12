"""Local, crash-resumable per-label result ledger for long transcript inference.

Good: open a ledger with a fingerprint of the exact run inputs; write each finished
batch in one SQLite transaction and export only when all expected keys exist.
Bad: append incomplete batches to an Excel sheet and treat it as a completed run.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ptg.storage.raw_results import RawResult


class CheckpointStore:
    def __init__(self, path: Path, *, fingerprint: str, run_sheet: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        try:
            with self.connection:
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self.connection.execute(
                    """CREATE TABLE IF NOT EXISTS results (
                        row_index INTEGER NOT NULL, label TEXT NOT NULL,
                        timing TEXT NOT NULL, speaker TEXT NOT NULL, transcript TEXT NOT NULL,
                        raw_response TEXT NOT NULL, parsed TEXT NOT NULL,
                        PRIMARY KEY (row_index, label)
                    )"""
                )
                existing = dict(self.connection.execute("SELECT key, value FROM metadata"))
                if existing:
                    if existing.get("fingerprint") != fingerprint:
                        raise ValueError("Checkpoint input fingerprint mismatch; archive it before a new run")
                    self.run_sheet = existing["run_sheet"]
                else:
                    self.connection.executemany(
                        "INSERT INTO metadata(key, value) VALUES (?, ?)",
                        [("fingerprint", fingerprint), ("run_sheet", run_sheet)],
                    )
                    self.run_sheet = run_sheet
        except Exception:
            self.connection.close()
            raise

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.connection.close()

    def completed_keys(self) -> set[tuple[int, str]]:
        return {(int(i), label) for i, label in self.connection.execute(
            "SELECT row_index, label FROM results"
        )}

    def append(self, records: list[RawResult]) -> None:
        with self.connection:
            self.connection.executemany(
                """INSERT INTO results
                (row_index, label, timing, speaker, transcript, raw_response, parsed)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(r.row_index, r.label, str(r.timing), str(r.speaker), str(r.transcript),
                  str(r.raw_response), str(r.parsed)) for r in records],
            )

    def all_results(self) -> list[RawResult]:
        return [RawResult(*row) for row in self.connection.execute(
            """SELECT row_index, timing, speaker, transcript, label, raw_response, parsed
            FROM results ORDER BY row_index, label"""
        )]
