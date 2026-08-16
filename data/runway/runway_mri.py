import math
from typing import Dict, Optional, Sequence, Tuple, List

import pandas as pd
import numpy as np


rwy_mri = pd.read_csv('data/runway/runway_mri_aug_2024.csv', delimiter=';', parse_dates=['t_update', 't_start', 't_end'], low_memory=False)


# --------
# 2) Utilities
# --------
def _to_dt(s):
    return pd.to_datetime(s, utc=False)

def _clip_to_day(start, end, day):
    day_start = pd.Timestamp(day).normalize()
    day_end   = day_start + pd.Timedelta(days=1)
    s = max(start, day_start)
    e = min(end, day_end)
    if s >= e:
        return None, None
    return s, e

def _minute_index_for_day(day: str | pd.Timestamp) -> pd.DatetimeIndex:
    d0 = pd.Timestamp(day).normalize()
    d1 = d0 + pd.Timedelta(days=1)
    # closed='left' so 00:00 through 23:59
    return pd.date_range(d0, d1, freq="min", inclusive="left")

def _collect_runways_from_mri(rwy_mri: pd.DataFrame) -> list[str]:
    cols = ["landing_1", "landing_2", "takeoff_1", "takeoff_2"]
    vals = set()
    for c in cols:
        if c in rwy_mri.columns:
            vals.update(rwy_mri[c].dropna().astype(str).unique().tolist())
    # keep only plausible runway identifiers (numbers + optional letter)
    return sorted([v for v in vals if isinstance(v, str) and len(v) >= 2])


BLOCK_REASON = "mri:closed"


def _validate_minutes_index(minutes: pd.DatetimeIndex, *, day: Optional[pd.Timestamp] = None) -> Tuple[pd.Timestamp, pd.Timedelta]:
    if not isinstance(minutes, pd.DatetimeIndex):
        raise TypeError("Minutes index must be a pandas.DatetimeIndex")
    if len(minutes) == 0:
        raise ValueError("Minutes index is empty")
    inferred = minutes.freq or minutes.inferred_freq
    if inferred is None:
        raise ValueError("Minutes index must have a fixed frequency")
    freq = pd.tseries.frequencies.to_offset(inferred)
    if freq != pd.Timedelta(minutes=1):
        raise ValueError(f"Expected 1-minute frequency, got {freq}")
    if minutes.tz is not None:
        raise ValueError("Minutes index must be timezone-naive")
    idx_day = minutes[0].normalize()
    if day is not None and idx_day != day.normalize():
        raise ValueError(f"Minutes index day {idx_day.date()} does not match requested day {day.date()}")
    expected_len = int(pd.Timedelta(days=1) / pd.Timedelta(minutes=1))
    if len(minutes) != expected_len:
        raise ValueError(f"Minutes index must span exactly 1440 rows, got {len(minutes)}")
    return idx_day, freq


def _resolve_runway_list(takeoff_map: Dict[str, pd.Series], requested: Optional[Sequence[str]]) -> list[str]:
    if requested is None:
        return sorted(takeoff_map.keys())
    missing = [rw for rw in requested if rw not in takeoff_map]
    print(takeoff_map)
    if missing:
        raise KeyError(f"Runways not found in MRI takeoff map: {missing}")
    return [str(rw) for rw in requested]


def _slot_span_seconds(slot_idx: int, slot_duration_seconds: int) -> Tuple[int, int]:
    start = int(slot_idx * slot_duration_seconds)
    end = min(start + slot_duration_seconds, 86400)
    return start, end


def build_takeoff_capacity_blocks(
    maps: Dict[str, Dict[str, pd.Series]],
    minutes: pd.DatetimeIndex,
    slots: pd.DataFrame,
    *,
    day: pd.Timestamp,
    slot_duration_seconds: int,
    runway_filter: Optional[Sequence[str]] = None,
    reason: str = BLOCK_REASON,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct artificial capacity blocks for closed takeoff minutes.

    Returns a tuple of (blocks_df, summary_df).
    blocks_df contains synthetic rows ready to be merged into scheduling data.
    summary_df reports usable vs consumed capacity per slot/runway for diagnostics.
    """
    if 'takeoff' not in maps:
        raise KeyError("MRI maps must include a 'takeoff' entry")
    day = pd.Timestamp(day).normalize()
    idx_day, _ = _validate_minutes_index(minutes, day=day)
    takeoff_map = maps['takeoff']
    # runways = _resolve_runway_list(takeoff_map, runway_filter)
    runways = runway_filter#[0] if isinstance(runway_filter, (list, tuple)) and len(runway_filter) > 0 else runway_filter
    if slots.empty:
        raise ValueError("Slots dataframe is empty; cannot derive capacity blocks")
    if 'slot_size' not in slots.columns:
        raise KeyError("Slots dataframe must include 'slot_size' column")
    slot_capacity_map = slots['slot_size'].to_dict()

    minute_seconds = ((minutes - idx_day).total_seconds()).astype(int)
    minute_bounds = np.column_stack([
        minute_seconds,
        minute_seconds + 60
    ])

    block_rows: list[dict] = []
    summary_rows: list[dict] = []

    unix_epoch = pd.Timestamp("1970-01-01 00:00:00")
    day_epoch = int((day - unix_epoch) / pd.Timedelta(seconds=1))

    if not runways:
        return pd.DataFrame(block_rows), pd.DataFrame(summary_rows)

    takeoff_cache: Dict[str, np.ndarray] = {}
    closed_minutes_map: Dict[str, List[pd.Timestamp]] = {}
    for runway in runways:
        series = takeoff_map[runway]
        series = series.reindex(minutes)
        if series.isna().any():
            raise ValueError(f"MRI availability for runway {runway} contains NaN entries")
        arr = series.to_numpy(dtype=bool)
        takeoff_cache[runway] = arr
        closed_mask = ~arr
        closed_minutes_map[runway] = minutes[closed_mask].to_list()

    open_matrix = np.vstack([takeoff_cache[rw].astype(np.int8, copy=False) for rw in runways])
    open_counts = open_matrix.sum(axis=0)
    total_runways = len(runways)

    total_slots = int(math.ceil(86400 / slot_duration_seconds))

    for slot_idx in range(total_slots):
        slot_capacity = int(slot_capacity_map.get(slot_idx, slot_capacity_map.get(float(slot_idx), 0)))
        if slot_capacity <= 0:
            continue
        slot_start_sec, slot_end_sec = _slot_span_seconds(slot_idx, slot_duration_seconds)
        if slot_start_sec >= 86400:
            continue
        slot_span_seconds = max(1, slot_end_sec - slot_start_sec)
        slot_start_ts = day + pd.Timedelta(seconds=slot_start_sec)
        slot_end_ts = day + pd.Timedelta(seconds=slot_end_sec)

        minute_start_idx = int(slot_start_sec // 60)
        minute_end_idx = int(math.ceil(slot_end_sec / 60))
        minute_end_idx = min(minute_end_idx, len(minutes))

        minute_counts = open_counts[minute_start_idx:minute_end_idx]
        closed_minutes = int(np.sum(minute_counts == 0))

        open_seconds = 0.0
        for mi in range(minute_start_idx, minute_end_idx):
            if mi < 0 or mi >= len(minutes):
                continue
            m_open = int(open_counts[mi])
            if m_open <= 0:
                continue
            m_start_sec, m_end_sec = minute_bounds[mi]
            overlap_start = max(slot_start_sec, m_start_sec)
            overlap_end = min(slot_end_sec, m_end_sec)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                open_seconds += m_open * overlap

        total_capacity_seconds = float(total_runways * slot_span_seconds)
        open_seconds = min(open_seconds, total_capacity_seconds)
        closure_seconds = max(0.0, total_capacity_seconds - open_seconds)
        closure_ratio = 0.0 if total_capacity_seconds <= 0 else min(1.0, closure_seconds / total_capacity_seconds)
        usable_ratio = max(0.0, 1.0 - closure_ratio)

        if closed_minutes == 0:
            closure_seconds = 0.0
            closure_ratio = 0.0
            usable_ratio = 1.0

        if slot_capacity == 1:
            usable_capacity = 1 if usable_ratio > 0.5 else 0
        else:
            provisional = usable_ratio * slot_capacity
            usable_capacity = int(min(slot_capacity, max(0, math.ceil(provisional))))
        consumed_capacity = slot_capacity - usable_capacity

        avg_open_runways = open_seconds / slot_span_seconds if slot_span_seconds else 0.0

        summary_rows.append({
            'runway': runways[0] if total_runways == 1 else 'AGGREGATED',
            'runways_considered': ','.join(runways),
            'slot_idx': slot_idx,
            'slot_capacity_units': slot_capacity,
            'usable_capacity_units': usable_capacity,
            'consumed_capacity_units': consumed_capacity,
            'closure_ratio': closure_ratio,
            'closure_seconds': closure_seconds,
            'avg_open_runways': avg_open_runways,
            'slot_start': slot_start_ts,
            'slot_end': slot_end_ts,
            'closed_minutes_in_slot': closed_minutes,
        })

        if consumed_capacity <= 0:
            continue

        slot_epoch = day_epoch + slot_start_sec
        block_meta_runways = ','.join(runways)
        block_trwy = runways[0] if total_runways == 1 else None
        block_status = 'CLOSED' if usable_capacity <= 0 else 'BLOCKED'
        for seq in range(consumed_capacity):
            block_id = f"MRI_BLOCK__{day.strftime('%Y%m%d')}__{slot_idx:04d}__{seq}"
            block_rows.append({
                'flight_key': block_id,
                'block_id': block_id,
                'trwy': block_trwy,
                'is_mri_block': True,
                'block_reason': reason,
                'block_sequence': seq,
                'sfplid': -(slot_idx * 100 + seq + 1),
                'rw_slot': int(slot_idx),
                'rw_cur': int(slot_idx),
                'sched_ttot_s': slot_start_ts,
                'sched_ttot': int(slot_epoch),
                'tobt': np.nan,
                'tobt_s': pd.NaT,
                'ctot': np.nan,
                'ctot_s': pd.NaT,
                'taxi_time_minutes': 0.0,
                'timesec': int(slot_epoch),
                'difference_tobt-sobt': 0,
                'ctot_updates': 0,
                'ctot_cancelled': 0,
                'last_ctot': np.nan,
                'slot_shift': 0,
                'slot_shiftback': 0,
                'shift_day': 0,
                'earliest_ok_sec': -np.inf,
                'latest_ok_sec': np.inf,
                'status': block_status,
                'slot_capacity_units': slot_capacity,
                'usable_capacity_units': usable_capacity,
                'consumed_capacity_units': consumed_capacity,
                'closure_ratio': closure_ratio,
                'closure_seconds': closure_seconds,
                'avg_open_runways': avg_open_runways,
                'total_runways_considered': total_runways,
                'runways_considered': block_meta_runways,
                'block_slot_start': slot_start_ts,
                'block_slot_end': slot_end_ts,
                'closed_minutes_in_slot': closed_minutes,
            })

    blocks_df = pd.DataFrame(block_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.attrs['closed_minutes_map'] = closed_minutes_map
    return blocks_df, summary_df


def merge_slot_capacity_summary(slots: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Attach MRI usable capacity diagnostics to the slots dataframe."""
    if summary.empty:
        slots = slots.copy()
        slots['mri_usable_capacity'] = slots['slot_size'] if 'slot_size' in slots.columns else np.nan
        slots['mri_consumed_capacity'] = 0
        return slots
    summary_latest = summary[['slot_idx', 'usable_capacity_units', 'consumed_capacity_units']].drop_duplicates('slot_idx', keep='last')
    slots = slots.copy()
    usable_series = summary_latest.set_index('slot_idx')['usable_capacity_units']
    consumed_series = summary_latest.set_index('slot_idx')['consumed_capacity_units']
    slots['mri_usable_capacity'] = slots.index.map(usable_series)
    if 'slot_size' in slots.columns:
        slots['mri_usable_capacity'] = slots['mri_usable_capacity'].where(slots['mri_usable_capacity'].notna(), slots['slot_size'])
    slots['mri_consumed_capacity'] = slots.index.map(consumed_series).fillna(0).astype(int)
    return slots


# --------
# 3) Build minute-level MRI (open/closed) maps per runway & role
# --------
def build_mri_minute_maps(rwy_mri: pd.DataFrame, day: str, runways: list[str] | None):
    df = rwy_mri.copy()
    for c in ["t_start", "t_end", "t_update"]:
        if c in df.columns:
            df[c] = _to_dt(df[c])

    idx = _minute_index_for_day(day)
    # roles: "takeoff" and "landing"
    mri_maps = {
        "takeoff": { },  # runway -> Series[bool] indexed by minute
        "landing": { },
    }
    if runways is None:
        runways = _collect_runways_from_mri(df)

    # Initialize all to False
    for rw in runways:
        mri_maps["takeoff"][rw] = pd.Series(False, index=idx)
        mri_maps["landing"][rw] = pd.Series(False, index=idx)

    # For each MRI interval row, mark open minutes for each listed runway/role
    for _, row in df.iterrows():
        s, e = row["t_start"], row["t_end"]
        if pd.isna(s) or pd.isna(e):
            continue
        s, e = _clip_to_day(s, e, day)
        if s is None:
            continue

        for role, col in [("landing", "landing_1"), ("landing", "landing_2"),
                          ("takeoff", "takeoff_1"), ("takeoff", "takeoff_2")]:
            if col in df.columns:
                rw = row[col]
                if pd.notna(rw) and rw in mri_maps[role]:
                    # Set True for [s, e)
                    m_idx = mri_maps[role][rw].index
                    sl = (m_idx >= s) & (m_idx < e)
                    mri_maps[role][rw].loc[sl] = True

    return mri_maps, idx, runways
