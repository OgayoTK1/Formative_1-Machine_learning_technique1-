"""
src/preprocessing/aggregate_daily.py

Processes each raw daily file (sms-call-internet-mi-YYYY-MM-DD.txt) into a
compact aggregated Parquet file, one row per (square_id, timestamp_ms), with
Internet traffic summed across country codes.

This is the same load -> select columns -> downcast dtypes -> aggregate
logic validated on 2013-11-01 (4,842,625 raw rows -> 1,439,982 aggregated
rows, 92.6% reduction in resident memory footprint), applied to all 62 days
one file at a time so peak memory never exceeds what one day's raw file
plus its aggregate requires.

In the same pass, accumulates a running total of Internet traffic per
square_id across the full observation period, avoiding a second read of
all 62 files just to answer "which squares have the highest total traffic."

Usage (inside the Colab notebook, after PROJECT_ROOT is defined):

    import sys
    sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'preprocessing'))
    from aggregate_daily import run_pipeline

    summary = run_pipeline(
        raw_dir=PROJECT_ROOT / 'data' / 'raw',
        interim_dir=PROJECT_ROOT / 'data' / 'interim',
        processed_dir=PROJECT_ROOT / 'data' / 'processed',
    )
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path

import pandas as pd
import psutil

COLUMN_NAMES = [
    "square_id", "timestamp_ms", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet",
]

FILENAME_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def extract_date_from_filename(filename: str) -> str:
    match = FILENAME_DATE_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Could not extract a date from filename: {filename}")
    return match.group(1)


def process_daily_file(raw_path: Path, interim_dir: Path) -> dict:
    """
    Load one raw daily file, keep only square_id/timestamp/internet,
    aggregate across country code, and write the result as Parquet.

    Returns a record with row counts, timing, and memory observations
    for this file, so the memory/processing-time claims in the report
    are traceable back to this function rather than asserted.
    """
    date_str = extract_date_from_filename(raw_path.name)
    out_path = interim_dir / f"{date_str}.parquet"

    rss_before = _current_rss_mb()
    start = time.perf_counter()

    df = pd.read_csv(
        raw_path, sep="\t", header=None, names=COLUMN_NAMES,
        usecols=["square_id", "timestamp_ms", "internet"],
        dtype={"square_id": "int32", "timestamp_ms": "int64", "internet": "float32"},
    )
    df["internet"] = df["internet"].fillna(0.0)
    raw_row_count = len(df)

    df_agg = df.groupby(["square_id", "timestamp_ms"], as_index=False)["internet"].sum()
    agg_row_count = len(df_agg)

    interim_dir.mkdir(parents=True, exist_ok=True)
    df_agg.to_parquet(out_path, index=False)

    # per-square total for this day, used by the caller to update the
    # running full-period total without re-reading the file later
    daily_square_totals = df_agg.groupby("square_id")["internet"].sum()

    del df, df_agg
    gc.collect()

    elapsed = time.perf_counter() - start
    rss_after = _current_rss_mb()

    return {
        "date": date_str,
        "raw_row_count": raw_row_count,
        "aggregated_row_count": agg_row_count,
        "elapsed_seconds": round(elapsed, 2),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "output_path": str(out_path),
        "daily_square_totals": daily_square_totals,
    }


def run_pipeline(raw_dir: Path, interim_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """
    Process every raw daily file in raw_dir, writing aggregated Parquet
    files to interim_dir, and accumulating total Internet traffic per
    square across the full period into a single Parquet file in
    processed_dir.

    Returns a per-file summary DataFrame (also saved to processed_dir)
    for inclusion in the report's memory/timing evidence.
    """
    raw_files = sorted(raw_dir.glob("sms-call-internet-mi-*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"No raw files found in {raw_dir}")

    running_total = pd.Series(dtype="float64")  # index: square_id
    file_summaries = []

    for i, raw_path in enumerate(raw_files, start=1):
        print(f"[{i}/{len(raw_files)}] {raw_path.name} ... ", end="", flush=True)
        result = process_daily_file(raw_path, interim_dir)

        running_total = running_total.add(result["daily_square_totals"], fill_value=0.0)

        print(
            f"{result['aggregated_row_count']:,} rows, "
            f"{result['elapsed_seconds']}s, "
            f"RSS {result['rss_before_mb']}->{result['rss_after_mb']} MB"
        )

        file_summaries.append({k: v for k, v in result.items() if k != "daily_square_totals"})

    processed_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(file_summaries)
    summary_path = processed_dir / "daily_processing_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    totals_df = running_total.rename("total_internet_traffic").reset_index()
    totals_df.columns = ["square_id", "total_internet_traffic"]
    totals_df = totals_df.sort_values("total_internet_traffic", ascending=False).reset_index(drop=True)
    totals_path = processed_dir / "total_traffic_per_square.parquet"
    totals_df.to_parquet(totals_path, index=False)

    print(f"\nProcessed {len(raw_files)} files.")
    print(f"Per-file summary saved to: {summary_path}")
    print(f"Total traffic per square saved to: {totals_path}")
    print(f"\nTop 10 squares by total Internet traffic:")
    print(totals_df.head(10).to_string(index=False))

    return summary_df
