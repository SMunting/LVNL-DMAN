#!/usr/bin/env python3
"""Compare final runway slot assignments (rw_cur) between two snapshot parquet stores.

Usage examples:
  python schedule_diff.py --base ./output/snapshots/2024-08-06/snapshot_24.parquet \
                          --test ./output/snapshots/test_rules/2024-08-06/snapshot_24.parquet

  python schedule_diff.py --base base.parquet --test test.parquet --limit 30 --csv diff_out.csv

Logic:
  1. Load each parquet file into a DataFrame.
  2. Derive the *final schedule* by taking, for every flight_key, the last row by snapshot_time.
     (If snapshot_time absent, fall back to row order.)
  3. Extract rw_cur (int) and any helpful timing columns (tobt, ttot, tsat) for context.
  4. Outer-merge base & test; compute diff = test_rw_cur - base_rw_cur (positive => moved later slot).
  5. Sort by |diff| descending (then by diff to group negatives/positives deterministically).
  6. Print summary + table of differences. Optionally export to CSV.

Columns in output:
  flight_key, base_rw_cur, test_rw_cur, diff, base_tobt, test_tobt, base_status, test_status

Exit code is 0 unless a fatal error occurs.
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd
from pathlib import Path


def _load_final_schedule(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    
    try:
        df = pd.read_parquet(path)
    except:
        df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame(columns=["flight_key","rw_cur"])

    # Ensure snapshot_time is datetime for ordering if present
    if 'snapshot_time' in df.columns:
        try:
            df = df.copy()
            df['snapshot_time'] = pd.to_datetime(df['snapshot_time'])
        except Exception:
            pass
        df = df.sort_values('snapshot_time')
        final = df.groupby('flight_key', as_index=False).tail(1).copy()
    else:
        # Fallback: assume last occurrence in file per flight_key is final
        final = df.sort_index().groupby('flight_key', as_index=False).tail(1).copy()

    # # Normalize rw_cur to int where possible (avoid SettingWithCopyWarning by operating on copy)
    # if 'rw_cur' in final.columns:
    #     final.loc[:, 'rw_cur'] = pd.to_numeric(final['rw_cur'], errors='coerce').astype('Int64')
    # else:
    #     final.loc[:, 'rw_cur'] = pd.NA

    # Drop is_frozen early (user requested not to show)
    keep_cols = [c for c in ['flight_key','tobt','ttot','tsat','status'] if c in final.columns]
    final = final[keep_cols].copy()
    # Prefix columns (except flight_key) with label for merge disambiguation
    rename_map = {c: f"{label}_{c}" for c in keep_cols if c != 'flight_key'}
    final = final.rename(columns=rename_map)
    return final

# For rw_cur as collumn
# def compute_diff(base_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
#     merged = pd.merge(base_df, test_df, on='flight_key', how='outer', indicator=True)
#     # rw_cur difference: positive means test scheduled later
#     merged['base_rw_cur'] = merged.get('base_rw_cur')
#     merged['test_rw_cur'] = merged.get('test_rw_cur')
#     # Convert to numeric (Int64) again after merge
#     for col in ['base_rw_cur','test_rw_cur']:
#         if col in merged.columns:
#             merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('Int64')
#     merged['diff'] = merged['test_rw_cur'] - merged['base_rw_cur']
#     merged['abs_diff'] = merged['diff'].abs()
#     merged = merged.sort_values(['abs_diff','diff','flight_key'], ascending=[False, True, True])
#     return merged

# For ttot as collumn
def compute_diff(base_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(base_df, test_df, on='flight_key', how='outer', indicator=True)
    
    # Parse datetime safely
    for col in ['base_ttot', 'test_ttot']:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], errors='coerce')

    # Compute timedelta diff in minutes (float), positive => test is later
    merged['diff'] = (merged['test_ttot'] - merged['base_ttot']).dt.total_seconds() / 60
    merged['abs_diff'] = merged['diff'].abs()

    merged = merged.sort_values(['abs_diff', 'diff', 'flight_key'], ascending=[False, True, True])
    return merged



def format_int(val):
    if pd.isna(val):
        return "-"
    return str(int(val))


def main():
    ap = argparse.ArgumentParser(description="Compare final rw_cur assignments between two snapshot parquet files.")
    ap.add_argument('--base', required=True, help='Baseline (original) snapshot parquet path')
    ap.add_argument('--test', required=True, help='Test / new-rules snapshot parquet path')
    ap.add_argument('--limit', type=int, default=50, help='Show top N differences (default 50, 0=all)')
    ap.add_argument('--show-zeros', action='store_true', help='Include flights with zero diff in printed table')
    ap.add_argument('--csv', help='Optional path to write full diff CSV')
    args = ap.parse_args()

    base_path = Path(args.base)
    test_path = Path(args.test)

    try:
        base_final = _load_final_schedule(base_path, 'base')
        test_final = _load_final_schedule(test_path, 'test')
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    diff_df = compute_diff(base_final, test_final)

    total_flights = diff_df.shape[0]
    changed = diff_df[diff_df['diff'].fillna(0) != 0].shape[0]
    print(f"Compared final schedules: {total_flights} flights (changed rw_cur: {changed}, unchanged: {total_flights - changed})")

    # Prepare display subset
    show_df = diff_df
    if not args.show_zeros:
        show_df = show_df[show_df['diff'].fillna(0) != 0]

    if args.limit and args.limit > 0:
        show_df = show_df.head(args.limit)

    if show_df.empty:
        print("No differences to display (maybe identical schedules). Use --show-zeros to list all.")
    else:
        # Columns requested (omit is_frozen & test_status)
        # display_cols = [
        #     'flight_key','base_rw_cur','test_rw_cur','diff',
        #     'base_status',  # keep baseline status
        #     'base_tobt','test_tobt','base_ttot','test_ttot'
        # ]
        display_cols = [
            'flight_key', 'base_ttot', 'test_ttot', 'diff',
            'base_status', 'base_tobt', 'test_tobt'
        ]
        display_cols = [c for c in display_cols if c in show_df.columns]
        # Format a concise table
        rows = []
        for _, r in show_df.iterrows():
            rows.append({c: r.get(c) for c in display_cols})
        out_df = pd.DataFrame(rows, columns=display_cols)

        print("Top differences (sorted by |diff|, then diff):")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(out_df)

    if args.csv:
        try:
            diff_df.to_csv(args.csv, index=False)
            print(f"Full diff written to {args.csv}")
        except Exception as e:
            print(f"WARNING: Could not write CSV: {e}")

    # Removed largest shift summary per user request.


if __name__ == '__main__':
    main()
