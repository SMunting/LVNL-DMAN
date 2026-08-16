#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot Explorer / Schedule Timeline Inspector

Purpose
-------
Utility for loading a snapshot store (single-file layout produced by
InMemorySnapshotStore.to_disk()) and exploring schedule evolution.

Supports BOTH 'full' mode (row every minute) and 'compact' mode (first +
changes + freeze) because all queries operate by *forward filling* the
latest known state per flight up to a requested time.

Key Capabilities
----------------
1. Final schedule (state at last snapshot)
2. Schedule at an arbitrary time t (inclusive): last known state per flight
   with first appearance <= t.
3. Flight history / update log for a specific flight_key with classified
   events (first_seen, slot_change, ttot_change, status_change, freeze).
4. (Extensible) Diff between two times (placeholder stub provided for future use).

CLI Examples
------------
Final schedule:
    python snapshot_explorer.py --store ./output/snapshots/2024-08-06 --final

Schedule at time:
    python snapshot_explorer.py --store ./output/snapshots/2024-08-06 --at 2024-08-06T05:20

Flight history:
    python snapshot_explorer.py --store ./output/snapshots/2024-08-06 --flight FLIGHT_KEY

Multiple queries at once are allowed (processed in this order: at, final, flight).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable
import argparse
import pandas as pd
import sys

# Reuse status constants if available (fallback literals otherwise)
try:  # pragma: no cover - defensive import
    from snapshot_generator import FREEZE_STATUS, TAKEN_OFF_STATUS, PLANNED_STATUS  # type: ignore
except Exception:  # pragma: no cover
    FREEZE_STATUS = 'FROZEN'
    TAKEN_OFF_STATUS = 'TAKEN_OFF'
    PLANNED_STATUS = 'PLANNED'


SNAPSHOT_FILE_NAME = 'snapshots_all.parquet'
SNAPSHOT_CSV_FALLBACK = 'snapshots_all.csv'
# New pattern support: snapshot_<suffix>.parquet / snapshot_<suffix>.csv (suffix = all or runway id)


def _resolve_store_path(store: str | Path) -> Path:
    p = Path(store)
    if p.is_file():
        return p
    # Treat as directory. Priority:
    # 1. Explicit snapshot_*.parquet if exactly one OR snapshot_all.parquet
    # 2. snapshots_all.parquet (legacy)
    # 3. snapshot_*.csv / snapshots_all.csv (legacy)
    # 4. Fail.
    # User can disambiguate by passing file path directly.
    # Collect parquet candidates
    parquet_legacy = p / SNAPSHOT_FILE_NAME
    parquet_pattern = list(p.glob("snapshot_*.parquet"))
    if parquet_legacy.exists():
        return parquet_legacy
    if parquet_pattern:
        if len(parquet_pattern) == 1:
            return parquet_pattern[0]
        # Prefer snapshot_all.parquet among multiple
        for cand in parquet_pattern:
            if cand.name == "snapshot_all.parquet":
                return cand
        raise FileNotFoundError(f"Multiple snapshot_*.parquet files found: {[c.name for c in parquet_pattern]}. Specify one explicitly.")
    # CSV legacy/pattern
    csv_legacy = p / SNAPSHOT_CSV_FALLBACK
    csv_pattern = list(p.glob("snapshot_*.csv"))
    if csv_legacy.exists():
        return csv_legacy
    if csv_pattern:
        if len(csv_pattern) == 1:
            return csv_pattern[0]
        for cand in csv_pattern:
            if cand.name == "snapshot_all.csv":
                return cand
        raise FileNotFoundError(f"Multiple snapshot_*.csv files found: {[c.name for c in csv_pattern]}. Specify one explicitly.")
    raise FileNotFoundError(f"Could not find snapshot file in {p} (expected snapshot_<suffix>.parquet or {SNAPSHOT_FILE_NAME})")


def load_snapshot_dataframe(store: str | Path) -> pd.DataFrame:
    """Load snapshot rows into a DataFrame with parsed timestamps.

    Returns DataFrame sorted by (flight_key, snapshot_time).
    """
    src = _resolve_store_path(store)
    if src.suffix == '.parquet':
        # Multi-engine fallback mirroring writer logic
        errors: list[str] = []
        df = None  # type: ignore
        for eng in (None, 'pyarrow', 'fastparquet'):
            try:
                if eng == 'pyarrow':
                    # Capability check: skip if pyarrow lacks ArrayStatistics (older/ABI mismatch)
                    import pyarrow as pa  # type: ignore
                    if not hasattr(pa.lib, 'ArrayStatistics'):
                        raise RuntimeError('pyarrow missing ArrayStatistics capability, skipping engine')
                df = pd.read_parquet(src, engine=eng)  # type: ignore[arg-type]
                break
            except Exception as e:  # collect and try next
                errors.append(f"{eng or 'auto'}: {e}")
        if df is None:
            raise RuntimeError(f"Failed to read parquet with any engine: {' | '.join(errors)}")
    else:
        df = pd.read_csv(src)
    if 'snapshot_time' not in df.columns:
        raise ValueError('snapshot_time column missing in snapshot file')
    df['snapshot_time'] = pd.to_datetime(df['snapshot_time'])
    # Ensure expected minimal columns exist (fill if missing)
    for col in ['flight_key','status','rw_cur','ttot','ctot','tsat','is_frozen']:
        if col not in df.columns:
            # Fill with safe default
            df[col] = None
    # Parse datetime-ish columns (strings) if present
    for col in ['ttot','ctot','tsat','rw_cur_start','rw_cur_end']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    df.sort_values(['flight_key','snapshot_time'], inplace=True)
    return df


@dataclass
class ScheduleState:
    snapshot_time: pd.Timestamp
    schedule: pd.DataFrame  # One row per flight (latest state <= snapshot_time)


class SnapshotExplorer:
    """In-memory exploration helper for a snapshot DataFrame"""

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError('Snapshot DataFrame is empty')
        self.df = df.copy()
        self.df.sort_values(['flight_key','snapshot_time'], inplace=True)
        self.last_snapshot_time = self.df['snapshot_time'].max()
        self.first_snapshot_time = self.df['snapshot_time'].min()

    # ---------- Core helpers ----------
    def _state_up_to(self, t: pd.Timestamp) -> ScheduleState:
        if t < self.first_snapshot_time:
            # No flights yet
            empty = self.df.head(0).copy()
            return ScheduleState(snapshot_time=t, schedule=empty)
        # Filter once; compact mode may skip many minutes but we just need last row per flight
        view = self.df[self.df['snapshot_time'] <= t]
        if view.empty:
            empty = self.df.head(0).copy()
            return ScheduleState(snapshot_time=t, schedule=empty)
        # Take last row per flight_key
        last_rows = (view.sort_values(['flight_key','snapshot_time'])
                         .groupby('flight_key', as_index=False)
                         .tail(1))
        last_rows = last_rows.sort_values(['rw_cur','flight_key'])
        return ScheduleState(snapshot_time=t, schedule=last_rows.reset_index(drop=True))

    # ---------- Public API ----------
    def final_schedule(self) -> ScheduleState:
        return self._state_up_to(self.last_snapshot_time)

    def schedule_at(self, t: str | pd.Timestamp) -> ScheduleState:
        ts = pd.to_datetime(t)
        return self._state_up_to(ts)

    def flight_history(self, flight_key: str) -> pd.DataFrame:
        # hist = self.df[self.df['flight_key'] == flight_key].copy()
        hist = self.df[self.df['flight_key'].str.contains(flight_key, na=False)].copy() # also show vacated slots
        if hist.empty:
            return hist  # empty
        hist.sort_values('snapshot_time', inplace=True)
        # Classify events
        events: List[str] = []
        changes: List[List[str]] = []
        prev_slot = prev_ttot = prev_status = prev_is_frozen = prev_tobt = prev_tsat = None
        for _, row in hist.iterrows():
            event_types: List[str] = []
            slot_cur = row.get('rw_cur')
            ttot_cur = row.get('ttot')
            status_cur = row.get('status')
            tobt_cur = row.get('tobt')
            tsat_cur = row.get('tsat')
            is_frozen_cur = bool(row.get('is_frozen')) if 'is_frozen' in row else (status_cur in (FREEZE_STATUS, TAKEN_OFF_STATUS))
            if prev_slot is None:
                event_types.append('first_seen')
            else:
                if slot_cur != prev_slot:
                    event_types.append('slot_change')
                # ttot comparison using string iso (avoid NaT issues)
                if (ttot_cur is not pd.NaT) and (prev_ttot is not None) and pd.notna(ttot_cur) and pd.notna(prev_ttot) and ttot_cur != prev_ttot:
                    event_types.append('ttot_change')
                # TOBT change detection with TSAT tolerance window check
                if (tobt_cur is not pd.NaT) and (prev_tobt is not None) and pd.notna(tobt_cur) and pd.notna(prev_tobt) and tobt_cur != prev_tobt:
                    event_types.append('tobt_change')
                    # Check if TSAT was preserved (TOBT within ±5min window of TSAT)
                    if pd.notna(tsat_cur) and pd.notna(prev_tsat) and tsat_cur == prev_tsat:
                        # TSAT unchanged despite TOBT change -> within tolerance window
                        event_types.append('tsat_preserved')
                if status_cur != prev_status:
                    event_types.append('status_change')
                if (not prev_is_frozen) and is_frozen_cur:
                    event_types.append('freeze')
            if not event_types:
                event_types.append('unchanged')
            events.append('+'.join(event_types))
            changes.append(event_types)
            prev_slot = slot_cur
            prev_ttot = ttot_cur
            prev_status = status_cur
            prev_is_frozen = is_frozen_cur
            prev_tobt = tobt_cur
            prev_tsat = tsat_cur
        hist['event'] = events
        hist['event_components'] = changes
        return hist

    def diff(self, t_from: str | pd.Timestamp, t_to: str | pd.Timestamp) -> pd.DataFrame:
        """Compute slot/status changes between two times (placeholder).

        Returns a DataFrame of flights whose rw_cur OR status differs.
        """
        s_from = self.schedule_at(t_from).schedule.set_index('flight_key')
        s_to = self.schedule_at(t_to).schedule.set_index('flight_key')
        joined = s_from[['rw_cur','status']].rename(columns={'rw_cur':'rw_cur_from','status':'status_from'}) \
            .join(s_to[['rw_cur','status']].rename(columns={'rw_cur':'rw_cur_to','status':'status_to'}), how='outer')
        mask = (joined['rw_cur_from'] != joined['rw_cur_to']) | (joined['status_from'] != joined['status_to'])
        return joined[mask].reset_index()


# --------------------------- CLI Front-End ---------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Explore snapshot store schedule evolution')
    p.add_argument('--store', required=True, help='Path to snapshot folder OR snapshots_all.parquet file')
    p.add_argument('--final', action='store_true', help='Show final schedule')
    p.add_argument('--at', dest='at_time', help='Show schedule at (ISO or YYYY-MM-DDTHH:MM)')
    p.add_argument('--flight', dest='flight_key', help='Show history for a single flight_key')
    p.add_argument('--limit', type=int, default=50, help='Limit rows printed for schedules (default 50; 0 = no limit)')
    p.add_argument('--wide', action='store_true', help='Print all columns (do not trim)')
    return p.parse_args(list(argv) if argv is not None else None)


def _print_schedule(state: ScheduleState, limit: int, wide: bool, label: str) -> None:
    sched = state.schedule.copy()
    sched.sort_values('snapshot_time', inplace=True)
    # Include rwy and tobt explicitly in standard column ordering if present
    cols_order = [c for c in ['flight_key','rwy','rw_cur','status','is_frozen','tobt','ttot','ctot','tsat'] if c in sched.columns]
    extra = [c for c in sched.columns if c not in cols_order]
    if not wide:
        # Keep core first few extra diagnostics only
        sched = sched[cols_order + extra]
    if limit > 0 and len(sched) > limit:
        print(f"{label} (showing first {limit} of {len(sched)} flights; use --limit 0 for all):")
        print(sched.head(limit).to_string(index=False))
    else:
        print(f"{label} ({len(sched)} flights):")
        print(sched.to_string(index=False))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        df = load_snapshot_dataframe(args.store)
    except Exception as e:
        print(f"Error loading snapshot store: {e}")
        return 1
    explorer = SnapshotExplorer(df)

    # Schedule at specific time first (so user can compare with final if both requested)
    if args.at_time:
        state = explorer.schedule_at(args.at_time)
        _print_schedule(state, args.limit, args.wide, f"Schedule at {state.snapshot_time.isoformat()}")

    if args.final:
        final_state = explorer.final_schedule()
        _print_schedule(final_state, args.limit, args.wide, "Final schedule")

    if args.flight_key:
        hist = explorer.flight_history(args.flight_key)
        if hist.empty:
            print(f"No history found for flight_key={args.flight_key}")
        else:
            print(f"History for flight_key={args.flight_key} ({len(hist)} snapshot rows):")
            show = hist if args.limit == 0 else hist.head(args.limit)
            # Add rwy and tobt if available
            cols_hist = ['snapshot_time','rwy','rw_cur','status','is_frozen','tobt','tsat','ttot','ctot','event']
            cols_hist = [c for c in cols_hist if c in show.columns]
            print(show[cols_hist].to_string(index=False))
            if args.limit and len(hist) > args.limit:
                print(f"(truncated; total rows {len(hist)}; use --limit 0 for all)")

    if not any([args.at_time, args.final, args.flight_key]):
        print("No action specified (use one of --at / --final / --flight). See --help")
        return 2
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
