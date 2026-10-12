from __future__ import annotations

import pandas as pd

from ptg.domain.models import SheetRef, TranscriptRow
from ptg.ingest.normalize import build_prev_lines, normalize_speaker_col, normalize_transcript_col


def sheet_name(sheet: SheetRef, template: str) -> str:
    return template.format(team=sheet.team, meet=sheet.meet, semester=sheet.semester)


def load_sheet_df(
    workbook_path: str,
    sheet: SheetRef,
    sheet_template: str,
) -> tuple[str, pd.DataFrame]:
    name = sheet_name(sheet, sheet_template)
    df = pd.read_excel(workbook_path, sheet_name=name)
    return name, df


def normalize_sheet_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Speaker" not in df.columns or "Transcript" not in df.columns:
        raise ValueError("Sheet must include Speaker and Transcript columns")

    df = df.copy()
    df["Speaker"] = normalize_speaker_col(df["Speaker"])
    df["Transcript"] = normalize_transcript_col(df["Transcript"])
    return df


def iter_rows_from_df(df: pd.DataFrame, prev_k: int) -> list[TranscriptRow]:
    rows: list[TranscriptRow] = []
    for i in df.index:
        txt = str(df.at[i, "Transcript"]).strip()
        if not txt:
            continue
        rows.append(
            TranscriptRow(
                row_index=int(i),
                speaker=str(df.at[i, "Speaker"]),
                transcript=txt,
                prev_lines=build_prev_lines(df, int(i), prev_k),
            )
        )
    return rows
