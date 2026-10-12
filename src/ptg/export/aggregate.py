from __future__ import annotations

import pandas as pd


def aggregate_runs(raw_df: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    """Convert long-format raw results to a wide per-row DataFrame.

    K=1: label columns contain "yes" | "no" | "" (empty = all runs failed).
    K>1: label columns contain a float confidence (yes / valid_runs, 0.0–1.0).

    Adds a ``Flags`` column listing (in codebook order) any labels where every
    run failed to parse a valid answer.
    """
    if raw_df.empty:
        cols = ["timing", "speaker", "transcript"] + list(label_names) + ["Flags"]
        return pd.DataFrame(columns=cols)

    run_ids = sorted(raw_df["run_id"].unique())
    K = len(run_ids)

    # Row identity: earliest run's values per row_index
    identity = (
        raw_df.sort_values("run_id")
        .groupby("row_index")[["timing", "speaker", "transcript"]]
        .first()
        .reset_index()
    )

    # Aggregate per (row_index, label)
    def _agg(group: pd.DataFrame) -> pd.Series:
        yes_count = (group["parsed"] == "yes").sum()
        valid_count = group["parsed"].isin(["yes", "no"]).sum()
        if valid_count == 0:
            return pd.Series({"value": "", "failed": True})
        if K == 1:
            value = "yes" if yes_count > 0 else "no"
        else:
            value = yes_count / valid_count
        return pd.Series({"value": value, "failed": False})

    agg = (
        raw_df.groupby(["row_index", "label"], sort=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    # Pivot to wide: one column per label
    pivot = agg.pivot(index="row_index", columns="label", values="value").reset_index()
    pivot.columns.name = None

    # Build Flags: failed labels in codebook order
    failed_df = agg[agg["failed"]][["row_index", "label"]]
    label_order = {name: i for i, name in enumerate(label_names)}
    flags_map: dict[int, list[str]] = {}
    for _, row in failed_df.iterrows():
        flags_map.setdefault(int(row["row_index"]), []).append(row["label"])

    def _sort_flags(labels: list[str]) -> str:
        return ", ".join(sorted(labels, key=lambda l: label_order.get(l, 999)))

    flags_series = pd.Series(
        {idx: _sort_flags(lbls) for idx, lbls in flags_map.items()},
        name="Flags",
    )
    flags_df = flags_series.reset_index().rename(columns={"index": "row_index"})

    # Merge identity + pivot + flags
    result = identity.merge(pivot, on="row_index", how="left")
    result = result.merge(flags_df, on="row_index", how="left")
    result["Flags"] = result["Flags"].fillna("")

    # Final column order: timing, speaker, transcript, labels (codebook order), Flags
    present_labels = [l for l in label_names if l in result.columns]
    result = result[["timing", "speaker", "transcript"] + present_labels + ["Flags"]]

    return result.reset_index(drop=True)
