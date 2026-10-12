from __future__ import annotations

import pandas as pd


def write_sheet(
    *,
    workbook_path: str,
    sheet_name: str,
    df: pd.DataFrame,
    out_sheet: str,
) -> None:
    # Writes a new sheet (or replaces if it exists)
    with pd.ExcelWriter(
        workbook_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, sheet_name=out_sheet, index=False)
