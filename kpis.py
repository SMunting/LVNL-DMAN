#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KPI computation module.

This module derives KPIs from the raw update history (df_history) and the
final scheduled result set (df_final) without replaying historical scheduling.

Provided functions are intentionally pure (no side effects) so they can be
unit tested easily. All timestamps are expected in epoch seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

import config


# ---------------------------------------------------------------------------
# Snapshot reconstruction
# ---------------------------------------------------------------------------

SNAPSHOT_COLUMNS = [
    "snapshot_epoch",      # time of snapshot (epoch seconds)
    "flight_key",
    "acid",
    "sfplid",
    "tobt",               # latest known TOBT at snapshot
    "ctot",               # latest known CTOT (NaN if none)
    "taxi_time_minutes",  # resolved taxi duration per update
    "sched_ttot",         # deterministic sched_ttot = tobt + taxi (no CTOT logic)
]


def build_snapshots(df_history: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct per-snapshot state from update rows.

    Rules:
    - Each row in df_history is an update event (timesec field) providing new TOBT/CTOT values.
    - We treat every update row as a snapshot boundary (no extra sampling cadence yet).
    - For each flight we keep only the *latest* row per (flight_key, snapshot_epoch) if duplicates.
    - sched_ttot (baseline prediction) = tobt + taxi_time_minutes*60 (no CTOT window clipping here).
    - Rows lacking mandatory fields are dropped.
    """
    required = {"timesec", "flight_key", "tobt", "acid", "sfplid"}
    missing = required - set(df_history.columns)
    if missing:
        raise ValueError(f"df_history missing required columns: {missing}")

    snaps = df_history.copy()
    snaps = snaps.sort_values(["flight_key", "timesec"])  # chronological
    snaps.rename(columns={"timesec": "snapshot_epoch"}, inplace=True)

    if "taxi_time_minutes" not in snaps.columns:
        raise ValueError("df_history must include taxi_time_minutes column")

    snaps["taxi_time_minutes"] = pd.to_numeric(snaps["taxi_time_minutes"], errors="coerce")
    if snaps["taxi_time_minutes"].isna().any():
        missing_taxi = snaps.loc[snaps["taxi_time_minutes"].isna(), "flight_key"].unique()[:5]
        raise ValueError(
            "Missing taxi_time_minutes for flights: " + ', '.join(map(str, missing_taxi))
        )

    # Ensure numeric epoch seconds
    for col in ["snapshot_epoch", "tobt", "ctot"]:
        if col in snaps.columns:
            snaps[col] = pd.to_numeric(snaps[col], errors="coerce")

    # Drop rows without TOBT (can't form a prediction)
    snaps = snaps[snaps["tobt"].notna()]

    # Baseline sched_ttot = tobt + taxi
    snaps["sched_ttot"] = snaps["tobt"] + (snaps["taxi_time_minutes"] * 60)

    keep_cols = [c for c in SNAPSHOT_COLUMNS if c in snaps.columns]
    snaps = snaps[keep_cols]

    # Deduplicate in case multiple updates share same seconds
    snaps = snaps.drop_duplicates(["flight_key", "snapshot_epoch"], keep="last")

    return snaps.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per‑flight temporal stability
# ---------------------------------------------------------------------------

@dataclass
class FlightStability:
    flight_key: str
    acid: str
    sfplid: str
    updates: int
    first_prediction_epoch: int
    last_prediction_epoch: int
    horizon_first_min: float
    horizon_last_min: float
    total_shift_slots: int
    substantial_shifts: int
    final_planned_slot: int
    final_slot_center_epoch: Optional[float]
    exact_adherence: int  # 1/0
    near_adherence: int   # 1/0 within +/-1 slot
    pilot_annoyance: int  # heuristic flag


def _compute_day_slot_index(epoch_series: pd.Series, slot_duration_seconds: int) -> pd.Series:
    """Compute slot index relative to local day (seconds modulo 86400).

    This matches how final assigned runway slot indices (rw_cur) are defined
    elsewhere in the system (day-relative). Using absolute epoch based slot
    indices previously caused huge artificial errors when comparing with
    day-relative final slots (e.g., millions of slots offset)."""
    day_seconds = (epoch_series % 86400)
    return (day_seconds // slot_duration_seconds).astype("Int64")


def compute_flight_stability(
    snapshots: pd.DataFrame,
    df_final: pd.DataFrame,
    slot_duration_seconds: int,
    ) -> pd.DataFrame:
    """Compute per‑flight stability metrics.

    Method:
    - Use snapshots.sched_ttot as predicted takeoff at each update.
    - Convert to slot indices for comparison with final assigned slot (rw_cur).
    - Count total slot prediction shifts (absolute diff between consecutive predicted slot indices).
    - Substantial shift: |delta_slots| * slot_duration_minutes >= threshold.
    - Adherence: final assigned slot equals last predicted slot index at last snapshot.
    - Near adherence: within +/- NEAR_ADHERENCE_SLOT_TOLERANCE slots.
    - Pilot annoyance: >=2 substantial shifts AND last substantial shift occurred within PILOT_ANNOYANCE_WINDOW_MINUTES before final sched_ttot.
    """
    if snapshots.empty or df_final is None or df_final.empty:
        return pd.DataFrame()

    # Map final slot indices per flight
    if not {"flight_key", "rw_cur"}.issubset(df_final.columns):
        raise ValueError("df_final missing required columns flight_key/rw_cur")

    final_slots = df_final.set_index("flight_key")["rw_cur"].astype("Int64")

    # Per-flight grouping
    records: List[FlightStability] = []
    slot_minutes = slot_duration_seconds / 60.0
    substantial_threshold_slots = config.SUBSTANTIAL_SHIFT_THRESHOLD_MINUTES / slot_minutes

    # Precompute per-row predicted slot
    snapshots = snapshots.copy()
    snapshots["pred_slot"] = _compute_day_slot_index(snapshots["sched_ttot"], slot_duration_seconds)

    for flight_key, grp in snapshots.groupby("flight_key", sort=False):
        grp = grp.sort_values("snapshot_epoch")
        pred_slots = grp["pred_slot"].dropna().astype(int).to_list()
        if not pred_slots:
            continue
        # compute deltas between consecutive predictions
        deltas = np.diff(pred_slots)
        total_shift_slots = int(np.sum(np.abs(deltas))) if deltas.size else 0
        substantial_mask = np.abs(deltas) >= substantial_threshold_slots
        substantial_shifts = int(substantial_mask.sum())

        # Horizon minutes: time until (final) planned takeoff from first & last snapshot
        first_snapshot_epoch = int(grp["snapshot_epoch"].iloc[0])
        last_snapshot_epoch = int(grp["snapshot_epoch"].iloc[-1])
        latest_sched_ttot = grp["sched_ttot"].iloc[-1]
        horizon_first_min = (latest_sched_ttot - first_snapshot_epoch) / 60.0
        horizon_last_min = (latest_sched_ttot - last_snapshot_epoch) / 60.0

        final_slot = final_slots.get(flight_key, pd.NA)
        final_slot_int = int(final_slot) if pd.notna(final_slot) else -1
        last_pred_slot = pred_slots[-1]

        exact_adherence = 1 if final_slot_int == last_pred_slot and final_slot_int >= 0 else 0
        near_adherence = 1 if (final_slot_int >= 0 and abs(final_slot_int - last_pred_slot) <= config.NEAR_ADHERENCE_SLOT_TOLERANCE) else 0

        # Pilot annoyance heuristic
        pilot_annoyance = 0
        if substantial_shifts >= 2:
            # Find indices of substantial shifts
            subst_indices = np.where(substantial_mask)[0]
            # Each delta refers to transition from snapshot i -> i+1 (use i+1 for the time of shift adoption)
            shift_epochs = grp["snapshot_epoch"].iloc[[i+1 for i in subst_indices]].to_numpy()
            # Time difference between last substantial shift and final predicted sched_ttot
            if shift_epochs.size:
                last_shift_epoch = shift_epochs[-1]
                minutes_before_takeoff = (latest_sched_ttot - last_shift_epoch) / 60.0
                if minutes_before_takeoff <= config.PILOT_ANNOYANCE_WINDOW_MINUTES:
                    pilot_annoyance = 1

        records.append(
            FlightStability(
                flight_key=flight_key,
                acid=str(grp["acid"].iloc[-1]) if "acid" in grp.columns else "", 
                sfplid=str(grp["sfplid"].iloc[-1]) if "sfplid" in grp.columns else "",
                updates=len(grp),
                first_prediction_epoch=first_snapshot_epoch,
                last_prediction_epoch=last_snapshot_epoch,
                horizon_first_min=horizon_first_min,
                horizon_last_min=horizon_last_min,
                total_shift_slots=total_shift_slots,
                substantial_shifts=substantial_shifts,
                final_planned_slot=final_slot_int,
                final_slot_center_epoch=np.nan,  # optional, can be filled with slots mapping later
                exact_adherence=exact_adherence,
                near_adherence=near_adherence,
                pilot_annoyance=pilot_annoyance,
            )
        )

    return pd.DataFrame([r.__dict__ for r in records])


# ---------------------------------------------------------------------------
# Horizon accuracy (placeholder structure)
# ---------------------------------------------------------------------------

def _label_horizon_bucket(minutes: float) -> str:
    # Clamp negative horizons (can arise from late updates) to 0 bucket
    if minutes < 0:
        minutes = 0.0
    prev = 0
    for edge in config.HORIZONS_MINUTES:
        if minutes <= edge:
            return f"{prev}-{edge}m" if prev != edge else f"{edge}m"
        prev = edge
    return f">{config.HORIZONS_MINUTES[-1]}m"


def compute_horizon_accuracy(snapshots: pd.DataFrame, df_final: pd.DataFrame, slot_duration_seconds: int) -> Dict[str, pd.DataFrame]:
    """Compute per-snapshot prediction error vs final outcome and bucket by horizon.

    Returns dict with keys:
        snapshot: per snapshot row errors
        bucket: aggregated stats per horizon bucket
    """
    if snapshots.empty or df_final is None or df_final.empty:
        return {"snapshot": pd.DataFrame(), "bucket": pd.DataFrame()}

    final_slots = df_final.set_index("flight_key")["rw_cur"].astype("Int64")
    # Anchor horizon to FINAL scheduled sched_ttot from df_final (CTOT‑adjusted), not baseline last snapshot
    if "sched_ttot" not in df_final.columns:
        raise ValueError("df_final missing sched_ttot for horizon anchoring")
    final_sched_map = df_final.set_index("flight_key")["sched_ttot"]
    snapshots = snapshots.copy()
    # Filter snapshots to only flights present in final schedule to avoid NaN final_slot mapping
    final_flight_keys = set(df_final['flight_key'].unique())
    snapshots = snapshots[snapshots['flight_key'].isin(final_flight_keys)]
    snapshots["pred_slot"] = _compute_day_slot_index(snapshots["sched_ttot"], slot_duration_seconds)
    # Attach final values
    snapshots["final_planned_ttot"] = snapshots["flight_key"].map(final_sched_map)  # true final scheduled
    snapshots["final_slot"] = snapshots["flight_key"].map(final_slots)
    # Drop any rows without final_slot (safety)
    snapshots = snapshots[snapshots['final_slot'].notna()]
    # Horizon before (final) planned takeoff
    snapshots["horizon_min"] = (snapshots["final_planned_ttot"] - snapshots["snapshot_epoch"]) / 60.0
    # Errors
    snapshots["slot_error"] = snapshots["pred_slot"].astype("float") - snapshots["final_slot"].astype("float")
    snapshots["abs_slot_error"] = snapshots["slot_error"].abs()
    snapshots["time_error_min"] = (snapshots["sched_ttot"] - snapshots["final_planned_ttot"]) / 60.0
    snapshots["abs_time_error_min"] = snapshots["time_error_min"].abs()
    # Bucket label
    snapshots["horizon_bucket"] = snapshots["horizon_min"].apply(_label_horizon_bucket)

    if snapshots.empty:
        return {"snapshot": snapshots, "bucket": pd.DataFrame()}

    def iqr(s: pd.Series):
        return s.quantile(0.75) - s.quantile(0.25)

    # Determine global p95 clip threshold for abs slot error to suppress extreme outliers in aggregation plots
    clip_thr = snapshots["abs_slot_error"].quantile(0.95) if snapshots.shape[0] else 0.0
    snapshots["abs_slot_error_clipped"] = snapshots["abs_slot_error"].clip(upper=clip_thr)
    grouped = snapshots.groupby("horizon_bucket")
    rows = []
    rng = np.random.default_rng(config.BOOTSTRAP_RANDOM_SEED)
    alpha = 1 - config.BOOTSTRAP_CONFIDENCE
    lower_q = alpha/2
    upper_q = 1 - alpha/2
    for bucket, g in grouped:
        n = len(g)
        abs_slot_raw = g['abs_slot_error'].to_numpy()
        abs_slot = g['abs_slot_error_clipped'].to_numpy()
        abs_time = g['abs_time_error_min'].to_numpy()
        slot_error = g['slot_error'].to_numpy()
        time_error = g['time_error_min'].to_numpy()
        row = {
            'horizon_bucket': bucket,
            'snapshots': n,
            'flights': g['flight_key'].nunique(),
            'mean_abs_slot_error': abs_slot.mean(),  # clipped
            'mean_abs_slot_error_raw': abs_slot_raw.mean(),
            'median_abs_slot_error': np.median(abs_slot),
            'std_abs_slot_error': abs_slot.std(ddof=1) if n>1 else 0.0,
            'iqr_abs_slot_error': np.quantile(abs_slot,0.75)-np.quantile(abs_slot,0.25) if n>1 else 0.0,
            'p90_abs_slot_error': np.quantile(abs_slot,0.9) if n>0 else 0.0,
            'max_abs_slot_error': abs_slot.max() if n>0 else 0.0,
            'mean_signed_slot_error': slot_error.mean(),
            'bias_slots': slot_error.mean(),
            'mean_signed_time_error_min': time_error.mean(),
            'mean_abs_time_error_min': abs_time.mean(),
            'median_abs_time_error_min': np.median(abs_time),
            'p90_abs_time_error_min': np.quantile(abs_time,0.9) if n>0 else 0.0,
            'clip_threshold_abs_slot_error': clip_thr,
        }
        # Bootstrap CIs if enough samples
        if n >= config.BOOTSTRAP_MIN_SAMPLE:
            bs_iter = config.BOOTSTRAP_ITERATIONS
            # Pre-allocate arrays
            bs_mean_abs_slot = np.empty(bs_iter)
            bs_mean_abs_time = np.empty(bs_iter)
            bs_mean_signed_slot = np.empty(bs_iter)
            bs_mean_signed_time = np.empty(bs_iter)
            for i in range(bs_iter):
                idx = rng.integers(0, n, n)
                sample_abs_slot = abs_slot[idx]  # already clipped
                sample_abs_time = abs_time[idx]
                sample_signed_slot = slot_error[idx]
                sample_signed_time = time_error[idx]
                bs_mean_abs_slot[i] = sample_abs_slot.mean()
                bs_mean_abs_time[i] = sample_abs_time.mean()
                bs_mean_signed_slot[i] = sample_signed_slot.mean()
                bs_mean_signed_time[i] = sample_signed_time.mean()
            row['mean_abs_slot_error_ci_lower'] = np.quantile(bs_mean_abs_slot, lower_q)
            row['mean_abs_slot_error_ci_upper'] = np.quantile(bs_mean_abs_slot, upper_q)
            row['mean_abs_time_error_min_ci_lower'] = np.quantile(bs_mean_abs_time, lower_q)
            row['mean_abs_time_error_min_ci_upper'] = np.quantile(bs_mean_abs_time, upper_q)
            row['mean_signed_slot_error_ci_lower'] = np.quantile(bs_mean_signed_slot, lower_q)
            row['mean_signed_slot_error_ci_upper'] = np.quantile(bs_mean_signed_slot, upper_q)
            row['mean_signed_time_error_min_ci_lower'] = np.quantile(bs_mean_signed_time, lower_q)
            row['mean_signed_time_error_min_ci_upper'] = np.quantile(bs_mean_signed_time, upper_q)
        else:
            row['mean_abs_slot_error_ci_lower'] = np.nan
            row['mean_abs_slot_error_ci_upper'] = np.nan
            row['mean_abs_time_error_min_ci_lower'] = np.nan
            row['mean_abs_time_error_min_ci_upper'] = np.nan
            row['mean_signed_slot_error_ci_lower'] = np.nan
            row['mean_signed_slot_error_ci_upper'] = np.nan
            row['mean_signed_time_error_min_ci_lower'] = np.nan
            row['mean_signed_time_error_min_ci_upper'] = np.nan
        rows.append(row)
    agg = pd.DataFrame(rows)
    agg['meets_min_sample'] = agg['snapshots'] >= config.HORIZON_BUCKET_MIN_COUNT
    if config.HORIZON_BUCKET_DROP_UNDER_MIN:
        agg = agg[agg['meets_min_sample']].reset_index(drop=True)
    agg['cv_abs_slot_error'] = agg['std_abs_slot_error'] / agg['mean_abs_slot_error'].replace({0: np.nan})
    return {"snapshot": snapshots, "bucket": agg}


# ---------------------------------------------------------------------------
# Throughput metrics (placeholder structure)
# ---------------------------------------------------------------------------
def compute_throughput(df_final: pd.DataFrame, slots: pd.DataFrame, slot_duration_seconds: int,
                       snapshots: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute rolling departures (realised) vs baseline demand.

    realised: counts of final scheduled departures (df_final.sched_ttot)
    demand: counts of baseline predicted departures (last snapshot sched_ttot per flight)
            ignoring CTOT adjustments (since build_snapshots uses pure TOBT+taxi baseline).
    """
    if df_final is None or df_final.empty or not {"flight_key", "sched_ttot"}.issubset(df_final.columns):
        return pd.DataFrame()

    # Realised times (final schedule)
    realised_times = df_final[["flight_key", "sched_ttot"]].dropna().copy()
    realised_arr = realised_times["sched_ttot"].sort_values().to_numpy()
    if realised_arr.size == 0:
        return pd.DataFrame()

    # Demand baseline: use LAST snapshot per flight predicted sched_ttot if provided
    if snapshots is not None and not snapshots.empty and {"flight_key", "sched_ttot"}.issubset(snapshots.columns):
        demand_base = (snapshots.sort_values("snapshot_epoch")
                       .groupby("flight_key").tail(1)[["flight_key", "sched_ttot"]])
        demand_arr = demand_base["sched_ttot"].sort_values().to_numpy()
    else:
        demand_arr = realised_arr  # fallback

    window = config.THROUGHPUT_WINDOW_MINUTES * 60
    step = config.THROUGHPUT_STEP_MINUTES * 60
    start = min(realised_arr.min(), demand_arr.min()) // step * step
    end = max(realised_arr.max(), demand_arr.max())
    records = []
    t = start
    r_left = r_right = d_left = d_right = 0
    while t <= end:
        w_start = t
        w_end = t + window
        # advance pointers realised
        while r_right < realised_arr.size and realised_arr[r_right] < w_end:
            r_right += 1
        while r_left < realised_arr.size and realised_arr[r_left] < w_start:
            r_left += 1
        realised_count = r_right - r_left
        # demand pointers
        while d_right < demand_arr.size and demand_arr[d_right] < w_end:
            d_right += 1
        while d_left < demand_arr.size and demand_arr[d_left] < w_start:
            d_left += 1
        demand_count = d_right - d_left
        records.append({
            "window_start_epoch": w_start,
            "window_end_epoch": w_end,
            "realised_departures": realised_count,
            "realised_rate_per_hour": realised_count * 3600 / window,
            "demand_departures": demand_count,
            "demand_rate_per_hour": demand_count * 3600 / window,
        })
        t += step
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Slot quality metrics (placeholder structure)
# ---------------------------------------------------------------------------
def compute_slot_quality(snapshots: pd.DataFrame, df_final: pd.DataFrame, slot_duration_seconds: int) -> pd.DataFrame:
    if snapshots.empty or df_final is None or df_final.empty:
        return pd.DataFrame()
    # Last baseline prediction per flight
    last_pred = (snapshots.sort_values("snapshot_epoch")
                 .groupby("flight_key").tail(1)
                 [["flight_key", "sched_ttot"]].copy())
    last_pred["baseline_slot"] = _compute_day_slot_index(last_pred["sched_ttot"], slot_duration_seconds)
    final_slots = df_final.set_index("flight_key")["rw_cur"].astype("Int64")
    last_pred["final_slot"] = last_pred["flight_key"].map(final_slots)
    last_pred["slot_diff"] = last_pred["final_slot"].astype("float") - last_pred["baseline_slot"].astype("float")
    last_pred["abs_slot_diff"] = last_pred["slot_diff"].abs()
    last_pred["direction"] = np.where(last_pred["slot_diff"]==0, "unchanged",
        np.where(last_pred["slot_diff"]>0, "later", "earlier"))
    last_pred.insert(0, 'kpi_table_type', 'slot_quality')
    return last_pred.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation / summary
# ---------------------------------------------------------------------------

def summarize_stability(stability_df: pd.DataFrame) -> Dict[str, float]:
    if stability_df.empty:
        return {}
    return {
        "flights": int(len(stability_df)),
        "avg_updates": float(stability_df["updates"].mean()),
        "median_updates": float(stability_df["updates"].median()),
        "mean_total_shift_slots": float(stability_df["total_shift_slots"].mean()),
        "pct_substantial_shift_ge2": float((stability_df["substantial_shifts"] >= 2).mean()),
        "exact_adherence_rate": float(stability_df["exact_adherence"].mean()),
        "near_adherence_rate": float(stability_df["near_adherence"].mean()),
        "pilot_annoyance_rate": float(stability_df["pilot_annoyance"].mean()),
    }


__all__ = [
    "build_snapshots",
    "compute_flight_stability",
    "compute_horizon_accuracy",
    "compute_throughput",
    "compute_slot_quality",
    "summarize_stability",
]
