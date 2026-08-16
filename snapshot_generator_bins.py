#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snaopshot generator for minute-by-minute deterministic planning state replay.

This module now exposes a minute-by-minute snapshot generator used to replay a
full operating day while *freezing* all past decisions. The new public API:

    generate_day_snapshots(day_start_utc, day_end_utc, scheduler, data_source, minute_stride=1)

It returns a SnapshotStore containing immutable PlanningSnapshot objects.

Materialize deterministic replay state for later analysis.

All times are UTC (naive pandas Timestamps assumed as UTC). No local tz logic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Protocol, Iterable, Iterator, Any, Dict, List, Tuple, Optional, Union, Sequence
import hashlib
from pathlib import Path
import json
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

import global_vars
from data.runway.runway_mri import build_takeoff_capacity_blocks

# Safe epoch bounds (seconds since 1970-01-01)
SAFE_EPOCH_MIN = 0
SAFE_EPOCH_MAX = 4102444800  # 2100-01-01

# Feature flags / configuration (back-compat friendly)
ALLOW_LATE_REVIVAL = True  # Turn off to revert to monotonic freeze behaviour
REVIVE_ON_FIELDS = {"tobt", "ctot", "trwy"}  # Fields that can trigger revival
PRESERVE_VACATED_SLOTS = True  # Keep vacated slots as permanent inefficiencies
# When using the FULL pipeline, optionally enforce vacated capacity by reducing slot_size
# so newly freed slots can't be backfilled within the same snapshot recompute.
ENFORCE_VACATED_CAPACITY_ON_FULL = False
# Optionally prevent assigning flights to slots that are infeasible relative to now (start - taxi < now)
DISALLOW_PAST_SLOT_ASSIGNMENT = False

# ---------------------------------------------------------------------------
# Domain dataclasses & interfaces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:  # Minimal slot representation for KPI snapshotting
    flight_key: str
    rwy: Optional[str]
    ttot: Optional[pd.Timestamp]  # planned take-off (slot reference)
    tsat: Optional[pd.Timestamp]
    ctot: Optional[pd.Timestamp]
    status: str  # PLANNED | TAKEN_OFF | FROZEN
    rw_cur: Optional[int] = None  # numeric slot index if available
    rw_cur_start: Optional[pd.Timestamp] = None
    rw_cur_end: Optional[pd.Timestamp] = None
    tobt: Optional[pd.Timestamp] = None  # newly added: current TOBT
    is_capacity_block: bool = False
    block_reason: Optional[str] = None
    block_meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PlanningSnapshot:
    snapshot_time: pd.Timestamp                # UTC minute t
    frozen_slot_ids: Tuple[str, ...]           # flight_keys frozen up to t
    scheduled_slots: Tuple[Slot, ...]          # full schedule state (frozen + future)
    pending_queue: Tuple[str, ...]             # flight_keys not yet scheduled
    metadata: Dict[str, Any]


class SnapshotStore(Protocol):
    def get(self, t: pd.Timestamp) -> PlanningSnapshot: ...
    def times(self) -> List[pd.Timestamp]: ...
    def to_disk(self, folder: Path) -> None: ...


class InMemorySnapshotStore:
    """Default in-memory snapshot store with optional persistence.

    Persistence format:
        folder/metadata.json
        folder/snapshots.parquet (row group per snapshot) OR
        folder/snapshots/<HHMM>.parquet (fallback if pyarrow unavailable)
    """

    def __init__(self, snapshots: Dict[pd.Timestamp, PlanningSnapshot]):
        self._snaps = dict(sorted(snapshots.items()))

    # Aggregate deterministic id (sum of nanosecond values) – lightweight hash for tests
    @property
    def aggregate_time_sum_ns(self) -> int:
        return int(sum(ts.value for ts in self._snaps))

    def get(self, t: pd.Timestamp) -> PlanningSnapshot:
        return self._snaps[t]

    def times(self) -> List[pd.Timestamp]:
        return list(self._snaps.keys())

    def to_disk(self, folder: Path, *, mode: str = 'full', include_hashes: bool = False, suffix: str = 'all') -> None:
        """Persist all snapshots into a SINGLE file (preferred) with per-row metadata.

        Layout (new v2):
            folder/snapshots_{suffix}.parquet (or CSV fallback) – columns include:
                snapshot_time, flight_key, rwy, ttot, tsat, ctot, status,
                rw_cur, rw_cur_start, rw_cur_end,
                is_frozen, num_frozen, queue_len, (config_hash, data_cut_hash optional)

        mode:
            'full'    -> write every flight row at every snapshot (default, verbose)
            'compact' -> write only: first appearance of a flight AND the snapshot where it freezes
                         (if it freezes in same snapshot as appearance only one row)
        include_hashes: include configuration & data-cut hashes (set False to drop columns)
        suffix: string suffix for the filename (default 'all' -> snapshots_all.parquet)

        Old auxiliary metadata files (metadata.json, metadata_timeseries.csv, per-minute CSVs)
        are no longer written. Loader keeps backward compatibility.
        """
        folder.mkdir(parents=True, exist_ok=True)
        if mode not in ('full','compact'):
            raise ValueError("mode must be 'full' or 'compact'")
        records: List[Dict[str, Any]] = []
        seen: set = set()
        frozen_written: set = set()  # flights for which we already wrote the frozen row
        # For compact change-tracking
        last_slot: Dict[str, Any] = {}
        last_ttot: Dict[str, Any] = {}
        last_status: Dict[str, str] = {}
        for ts, snap in self._snaps.items():
            num_frozen = len(snap.frozen_slot_ids)
            queue_len = len(snap.pending_queue)
            cfg_hash = snap.metadata.get('config_hash') if include_hashes else None
            cut_hash = snap.metadata.get('data_cut_hash') if include_hashes else None
            frozen_set = set(snap.frozen_slot_ids)
            for sl in snap.scheduled_slots:
                is_frozen = sl.flight_key in frozen_set
                if mode == 'compact':
                    need_row = False
                    fk = sl.flight_key
                    # First appearance
                    if fk not in seen:
                        need_row = True
                    # Freeze event (first frozen record)
                    elif is_frozen and fk not in frozen_written:
                        need_row = True
                    else:
                        # Detect slot / ttot / status change since last written
                        slot_changed = (fk in last_slot and sl.rw_cur != last_slot[fk])
                        ttot_iso = sl.ttot.isoformat() if sl.ttot else None
                        ttot_changed = (fk in last_ttot and ttot_iso != last_ttot[fk])
                        status_changed = (fk in last_status and sl.status != last_status[fk])
                        if slot_changed or ttot_changed or status_changed:
                            need_row = True
                    if not need_row:
                        continue
                # Record row
                rec = {
                    'snapshot_time': ts.isoformat(),
                    'flight_key': sl.flight_key,
                    'rwy': sl.rwy,
                    'ttot': sl.ttot.isoformat() if sl.ttot else None,
                    'tsat': sl.tsat.isoformat() if sl.tsat else None,
                    'ctot': sl.ctot.isoformat() if sl.ctot else None,
                    'tobt': sl.tobt.isoformat() if sl.tobt else None,  # added
                    'status': sl.status,
                    'rw_cur': sl.rw_cur,
                    'rw_cur_start': sl.rw_cur_start.isoformat() if sl.rw_cur_start else None,
                    'rw_cur_end': sl.rw_cur_end.isoformat() if sl.rw_cur_end else None,
                    'is_frozen': is_frozen,
                    'num_frozen': num_frozen,
                    'queue_len': queue_len,
                }
                if include_hashes:
                    rec['config_hash'] = cfg_hash
                    rec['data_cut_hash'] = cut_hash
                records.append(rec)
                seen.add(sl.flight_key)
                if mode == 'compact' and is_frozen:
                    frozen_written.add(sl.flight_key)
                if mode == 'compact':
                    last_slot[sl.flight_key] = sl.rw_cur
                    last_ttot[sl.flight_key] = sl.ttot.isoformat() if sl.ttot else None
                    last_status[sl.flight_key] = sl.status
        if not records:
            return
        df = pd.DataFrame.from_records(records)
        target_parquet = folder / f'snapshots_{suffix}.parquet'
        # Attempt write with preferred engines (pyarrow -> fastparquet) and capability check
        write_errors: List[str] = []
        engines: List[str | None] = [None, 'pyarrow', 'fastparquet']  # None lets pandas choose
        for eng in engines:
            try:
                if eng == 'pyarrow':
                    try:
                        import pyarrow as pa  # type: ignore
                        # Skip if required attribute absent (older pyarrow triggering ArrayStatistics error pathway)
                        if not hasattr(pa.lib, 'ArrayStatistics'):
                            raise RuntimeError('pyarrow too old (missing ArrayStatistics); skipping engine')
                    except Exception as ie:  # import or capability
                        write_errors.append(f"pyarrow import/capability: {ie}")
                        continue
                df.to_parquet(target_parquet, index=False, engine=None if eng is None else eng)
                break
            except Exception as e:
                write_errors.append(f"engine {eng or 'auto'}: {e}")
        else:
            # All attempts failed -> CSV fallback
            target_csv = folder / f'snapshots_{suffix}.csv'
            df.to_csv(target_csv, index=False)
            print("Parquet write failed on all engines; errors: " + " | ".join(write_errors) + f"; wrote CSV fallback -> {target_csv}")


def load_snapshot_store(folder: Path) -> InMemorySnapshotStore:
    """Load snapshots from either new single-file layout (v2) or older layouts.

    Resolution order:
        1. snapshots_all.parquet / snapshots_all.csv (new design)
        2. legacy snapshots.parquet + metadata_timeseries.csv
        3. legacy per-minute HHMM.(parquet|csv) + metadata_timeseries.csv
    """
    single_parquet = folder / 'snapshots_all.parquet'
    single_csv = folder / 'snapshots_all.csv'
    df_slots: Optional[pd.DataFrame] = None
    meta_lookup: Dict[str, Dict[str, Any]] = {}

    if single_parquet.exists() or single_csv.exists():
        # New layout
        try:
            df_slots = pd.read_parquet(single_parquet) if single_parquet.exists() else None
        except Exception:
            df_slots = None
        if df_slots is None and single_csv.exists():
            df_slots = pd.read_csv(single_csv)
    else:
        # Legacy paths
        meta_csv = folder / 'metadata_timeseries.csv'
        has_meta = meta_csv.exists()
        df_meta = pd.read_csv(meta_csv) if has_meta else pd.DataFrame()
        slots_parquet = folder / 'snapshots.parquet'
        if slots_parquet.exists():
            df_slots = pd.read_parquet(slots_parquet)
        else:
            per_files = sorted([p for p in folder.iterdir() if p.is_file() and p.stem.isdigit() and len(p.stem)==4 and p.suffix in ('.parquet','.csv')])
            if per_files:
                parts = []
                for pf in per_files:
                    part = pd.read_parquet(pf) if pf.suffix=='.parquet' else pd.read_csv(pf)
                    parts.append(part)
                if parts:
                    df_slots = pd.concat(parts, ignore_index=True)
        if has_meta and not df_meta.empty:
            meta_lookup = {row.snapshot_time: {'config_hash': getattr(row,'config_hash',None), 'data_cut_hash': getattr(row,'data_cut_hash',None)} for _, row in df_meta.iterrows()}

    if df_slots is None or df_slots.empty:
        raise FileNotFoundError(f"No snapshot data found in {folder}")

    required_cols = {'snapshot_time','flight_key','status'}
    missing = required_cols - set(df_slots.columns)
    if missing:
        raise ValueError(f"Snapshot store missing required columns: {missing}")

    snaps: Dict[pd.Timestamp, PlanningSnapshot] = {}
    for ts_iso, df_group in df_slots.groupby('snapshot_time'):
        ts = pd.Timestamp(ts_iso)
        scheduled: List[Slot] = []
        for _, r in df_group.iterrows():
            scheduled.append(Slot(
                flight_key=r.flight_key,
                rwy=r.rwy if 'rwy' in r and pd.notna(r.rwy) else None,
                ttot=pd.Timestamp(r.ttot) if 'ttot' in r and pd.notna(r.ttot) else None,
                tsat=pd.Timestamp(r.tsat) if 'tsat' in r and pd.notna(r.tsat) else None,
                ctot=pd.Timestamp(r.ctot) if 'ctot' in r and pd.notna(r.ctot) else None,
                status=r.status,
                rw_cur=int(r.rw_cur) if 'rw_cur' in r and pd.notna(r.rw_cur) else None,
                rw_cur_start=pd.Timestamp(r.rw_cur_start) if 'rw_cur_start' in r and pd.notna(r.rw_cur_start) else None,
                rw_cur_end=pd.Timestamp(r.rw_cur_end) if 'rw_cur_end' in r and pd.notna(r.rw_cur_end) else None,
                tobt=pd.Timestamp(r.tobt) if 'tobt' in r and pd.notna(r.tobt) else None,  # added
            ))
        # Prefer explicit is_frozen column if present
        if 'is_frozen' in df_group.columns:
            frozen_ids = tuple(sorted(df_group.loc[df_group['is_frozen']==True,'flight_key'].tolist()))  # noqa: E712
        else:
            frozen_ids = tuple(sorted(sl.flight_key for sl in scheduled if sl.status in ('FROZEN','TAKEN_OFF')))
        meta = {
            'config_hash': df_group['config_hash'].iloc[0] if 'config_hash' in df_group.columns else None,
            'data_cut_hash': df_group['data_cut_hash'].iloc[0] if 'data_cut_hash' in df_group.columns else None,
        }
        if not meta['config_hash'] and ts_iso in meta_lookup:
            meta.update(meta_lookup[ts_iso])
        snaps[ts] = PlanningSnapshot(
            snapshot_time=ts,
            frozen_slot_ids=frozen_ids,
            scheduled_slots=tuple(sorted(scheduled, key=lambda s: (s.rw_cur if s.rw_cur is not None else 1e9, s.flight_key))),
            pending_queue=tuple(),
            metadata=meta,
        )
    return InMemorySnapshotStore(snaps)


def iter_snapshots(store: SnapshotStore) -> Iterator[PlanningSnapshot]:
    for t in store.times():
        yield store.get(t)


# ---------------------------------------------------------------------------
# Interfaces for dependency injection (scheduler & data source)
# ---------------------------------------------------------------------------

class SchedulerInterface(Protocol):
    def warm_start(self, frozen_slots: Iterable[Slot]) -> None: ...
    def schedule(self, flights: Iterable[Dict[str, Any]], now: pd.Timestamp) -> List[Slot]: ...


class DataSourceInterface(Protocol):
    def load_up_to(self, t: pd.Timestamp) -> pd.DataFrame: ...  # raw messages (history)
    def to_flights(self, messages: pd.DataFrame) -> List[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

FREEZE_STATUS = 'FROZEN'
PLANNED_STATUS = 'PLANNED'
TAKEN_OFF_STATUS = 'TAKEN_OFF'


def _hash_dict(d: Dict[str, Any]) -> str:
    enc = json.dumps(d, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(enc).hexdigest()


def _config_hash() -> str:
    cfg = {
        'slot_duration': global_vars.SLOT_DURATION_SECONDS,
        'slot_size': global_vars.SLOT_SIZE,
        'taxi_time_strategy': 'per_flight_stand_runway_matrix_v1',
        'ctot_min_margin': global_vars.CTOT_MIN_MARGIN,
        'ctot_max_margin': global_vars.CTOT_MAX_MARGIN,
        'version': 1,
    }
    return _hash_dict(cfg)


def is_past(slot: Slot, t: pd.Timestamp) -> bool:
    """Check if a slot should be frozen (i.e., its TSAT window has passed).
    
    A slot is considered past (should be frozen) when:
    1. Status is TAKEN_OFF, or
    2. The TSAT window has expired: now > tsat + 5min
    
    This uses the TSAT window rather than TTOT to determine freeze timing,
    which aligns with the VACATED detection logic.
    """
    if slot.status == TAKEN_OFF_STATUS:
        return True
    # Use TSAT + 5min window to determine freeze timing (not TTOT)
    if slot.tsat and slot.tsat + pd.Timedelta(minutes=5) < t:
        return True
    return False


def check_tsat_missed(
    now: pd.Timestamp,
    old_tsat: Optional[pd.Timestamp],
    new_tobt: Optional[pd.Timestamp],
    taxi_time_minutes: Optional[float] = None,
    tolerance_minutes: int = 5,
    old_ctot: Optional[pd.Timestamp] = None,
    new_ctot: Optional[pd.Timestamp] = None,
    fields_changed: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Check if a TSAT window was missed based on the pseudo code logic:
    
    if now < tsat - 5min:
        tsat_missed = False  # planning-phase update
    else:
        if earliest_possible_new_tsat > tsat + 5min:
             tsat_missed = True
        else:
             tsat_missed = False
    
    Additionally, if a CTOT change occurs after the TSAT window has expired,
    it also triggers VACATED (since the flight was already frozen and the
    CTOT change is outside our control).
    
    Args:
        now: Current snapshot time
        old_tsat: Previously assigned TSAT
        new_tobt: New/updated TOBT from incoming message
        taxi_time_minutes: Taxi time in minutes (not used for TSAT calculation in this context)
        tolerance_minutes: Tolerance window in minutes (default: 5)
        old_ctot: Previously assigned CTOT (optional)
        new_ctot: New/updated CTOT from incoming message (optional)
        fields_changed: List of fields that changed (optional, for determining vacate reason)
    
    Returns:
        Tuple of (tsat_missed: bool, vacate_reason: Optional[str])
        - tsat_missed: True if TSAT was missed (should trigger VACATED status)
        - vacate_reason: String describing why vacated (e.g., 'tobt_change', 'ctot_change')
    """
    if old_tsat is None:
        return False, None
    
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    planning_window_start = old_tsat - tolerance
    tsat_window_end = old_tsat + tolerance
    
    # Step 1: Check if still in planning phase (now < tsat - 5min)
    if now < planning_window_start:
        return False, None  # Planning-phase update, TSAT not missed
    
    # Step 2: We're past the planning phase (now >= tsat - 5min)
    # Check for TOBT-based TSAT miss
    if new_tobt is not None:
        # In the bins-based scheduler, TSAT = max(slot_start - taxi, TOBT)
        # Since slot assignment may change, we use new_tobt as an approximation for the
        # earliest possible new TSAT. This is a simplification: actual new TSAT depends on
        # which slot gets assigned, but new_tobt represents the minimum TSAT achievable.
        earliest_possible_new_tsat = new_tobt
        
        if earliest_possible_new_tsat > tsat_window_end:
            return True, 'tobt_change'  # TSAT missed - new TSAT would be more than 5min after old
    
    # Step 3: Check for CTOT-based vacate (CTOT changes after TSAT window expired)
    # If now > tsat + 5min (window has fully expired) and CTOT changed, trigger vacate
    if now > tsat_window_end:
        if new_ctot is not None and old_ctot is not None and new_ctot != old_ctot:
            return True, 'ctot_change'  # Vacated due to CTOT change after window expired
        elif new_ctot is not None and old_ctot is None:
            return True, 'ctot_change'  # New CTOT assigned after window expired
    
    return False, None  # TSAT not missed - update within tolerance window


# ---------------------------------------------------------------------------
# Concrete adapters using existing codebase for production integration
# ---------------------------------------------------------------------------

def _deduplicate_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest message per flight_key using timesec ordering only.

    Rationale: Earlier version biased rows with ATD which caused premature
    'TAKEN_OFF' states to influence snapshot freezing. We only need TOBT/CTOT
    for planning; ATD is purely for post-hoc performance analysis. Therefore
    we ignore ATD entirely here and just take the newest update (largest timesec).
    """
    if df.empty:
        return df
    # Ensure numeric sorting (coerce to numeric just in case)
    if 'timesec' in df.columns:
        df = df.copy()
        df['timesec'] = pd.to_numeric(df['timesec'], errors='coerce')
    # Sort newest first so drop_duplicates keeps latest
    df_sorted = df.sort_values(['flight_key','timesec','sfplid'], ascending=[True, False, False])
    dedup = df_sorted.drop_duplicates(subset=['flight_key'], keep='first').copy()
    return dedup


class HistoricalCSVDataSource:
    """Data source backed by a pre-loaded history DataFrame (df_history).

    Expects columns: timesec (epoch s), flight_key, tobt, ctot, tsat, ttot, trwy, atd, sfplid, acid
    """
    def __init__(self, df_history: pd.DataFrame):
        self.df_history = df_history.copy()
        if 'flight_key' not in self.df_history.columns:
            raise ValueError('df_history must include flight_key column')
        if 'taxi_time_minutes' not in self.df_history.columns:
            raise ValueError('df_history must include taxi_time_minutes column for snapshot generation')
        self.df_history['taxi_time_minutes'] = pd.to_numeric(
            self.df_history['taxi_time_minutes'], errors='coerce'
        )
        if self.df_history['taxi_time_minutes'].isna().any():
            missing = self.df_history.loc[
                self.df_history['taxi_time_minutes'].isna(), 'flight_key'
            ].unique()[:5]
            raise ValueError(
                "Taxi time minutes missing for flights: " + ', '.join(map(str, missing))
            )
        # One-time epoch sanitation (faster than repeating every snapshot).
        for col in ('tobt','ctot','atd'):
            if col in self.df_history.columns:
                self.df_history[col] = pd.to_numeric(self.df_history[col], errors='coerce')
                invalid_mask = (
                    (self.df_history[col] < SAFE_EPOCH_MIN) |
                    (self.df_history[col] > SAFE_EPOCH_MAX) |
                    (~np.isfinite(self.df_history[col])) |
                    (self.df_history[col] == 0)
                )
                # Treat 0 as invalid placeholder -> NaN
                if invalid_mask.any():
                    self.df_history.loc[invalid_mask, col] = np.nan

    def load_up_to(self, t: pd.Timestamp) -> pd.DataFrame:
        # Include all events from the entire minute to match change detection
        # which floors timestamps to minutes. An event at 22:14:48 triggers
        # a recompute at 22:14:00 minute, so we need to include data from
        # the full minute (22:14:00 to 22:14:59.999...).
        # Use ceiling to get next minute start, then use < to include all of current minute.
        next_minute = (t + pd.Timedelta(minutes=1)).floor('min')
        epoch = int(next_minute.timestamp())
        return self.df_history[self.df_history['timesec'] < epoch].copy()

    # def load_up_to(self, t: pd.Timestamp) -> pd.DataFrame:
    #     df = self.df_history[self.df_history['timesec'] <= t]
    #     idx = df.groupby('flight_key')['timesec'].idxmax()
    #     return df.loc[idx]

    def to_flights(self, messages: pd.DataFrame) -> List[Dict[str, Any]]:
        dedup = _deduplicate_messages(messages)
        flights: List[Dict[str, Any]] = []
        if dedup.empty:
            return flights
        # Basic derived fields
        for _, r in dedup.iterrows():
            flights.append({
                'flight_key': r.flight_key,
                'acid': r.acid,
                'tobt': r.get('tobt'),
                'ctot': r.get('ctot'),
                'trwy': str(r.get('trwy')) if pd.notna(r.get('trwy')) else None,
                'atd': r.get('atd'),
                'taxi_time_minutes': float(r.get('taxi_time_minutes')) if pd.notna(r.get('taxi_time_minutes')) else None,
            })
        return flights


class SimpleSchedulerAdapter:
    """Very lightweight deterministic scheduler used only for snapshot generation.

    Now enriched with priority rules consistent with forward scheduler:
      1. ctot_updates >= 10
      2. slot shifts > 2
      3. (placeholder) STW violation if shifted (not available -> neutral)
      4. CTOT cancellation late-impact condition (approx using last_ctot + cancellation flag)
      5. 1–2 previous shifts
      6. Earlier TOBT (ascending)
      7. difference_tobt-sobt (descending rank if present)
      8. sfplid (ascending)
    Lower tuple sorts earlier (kept in earlier slot). Excess flights in an overfilled slot are pushed forward.
    """
    def __init__(self):
        self._frozen: Dict[str, Slot] = {}
        self._blocks: Dict[str, Slot] = {}
        self._blocked_slots: set[int] = set()  # permanently vacated slot indices (inefficiencies)
        # Track dynamic shift counts (non-frozen planning volatility)
        self._shift_counts: Dict[str, int] = {}
        self._prev_assignment: Dict[str, int] = {}

    def block_slot(self, slot_idx: Optional[int]) -> None:
        if slot_idx is not None:
            self._blocked_slots.add(int(slot_idx))

    def set_capacity_blocks(self, blocks: Iterable[Slot]) -> None:
        self._blocks = {b.flight_key: b for b in blocks}

    def warm_start(self, frozen_slots: Iterable[Slot]) -> None:
        self._frozen.clear()
        for s in frozen_slots:
            if getattr(s, 'is_capacity_block', False):
                self._blocks[s.flight_key] = s
            else:
                self._frozen[s.flight_key] = s
            # Frozen flights do not accumulate shift counts

    def _priority_tuple(self, f: Dict[str, Any]) -> tuple:
        # Helper safely fetch
        def gv(k, default=None):
            return f.get(k, default)
        # Rule 1
        p1 = 0 if (gv('ctot_updates', 0) is not None and gv('ctot_updates', 0) >= 10) else 1
        # Rule 2
        shifts = self._shift_counts.get(f['flight_key'], 0)
        p2 = 0 if shifts > 2 else 1
        # Rule 3 (STW violation if shifted) – unavailable -> neutral 1
        p3 = 1
        # Rule 4 cancellation late impact: if cancelled & (planned slot start after last_ctot) -> priority
        last_ctot = gv('last_ctot')
        ctot_cancelled = gv('ctot_cancelled', 0)
        # Interpret last_ctot epoch seconds if numeric
        try:
            now_slot_ref = gv('_slot_start_epoch')  # injected later
            cond4 = (ctot_cancelled and last_ctot and now_slot_ref and now_slot_ref > last_ctot)
        except Exception:
            cond4 = False
        p4 = 0 if cond4 else 1
        # Rule 5 (1–2 previous shifts): map to tier (original forward used 3/2/1)
        if shifts > 1:
            p5 = 1
        elif shifts == 1:
            p5 = 2
        else:
            p5 = 3
        # Rule 6 earlier TOBT
        tobt = gv('tobt') or 0
        # Rule 7 difference_tobt-sobt (want larger difference first per original descending rank)
        diff = gv('difference_tobt-sobt')
        p7 = -(diff if diff is not None and not pd.isna(diff) else -0)  # negate for descending effect via ascending sort
        # Rule 8 flight id (sfplid) fallback to flight_key
        fid = gv('sfplid') or f['flight_key']
        return (p1, p2, p3, p4, p5, tobt, p7, str(fid), f['flight_key'])

    def schedule(self, flights: Iterable[Dict[str, Any]], now: pd.Timestamp) -> List[Slot]:
        flights_list = list(flights)
        scheduled_map: Dict[str, Slot] = {}
        scheduled_map.update(self._blocks)
        scheduled_map.update(self._frozen)
        scheduled: List[Slot] = list(scheduled_map.values())

        if not flights_list:
            return sorted(scheduled, key=lambda s: (s.rw_cur if s.rw_cur is not None else 1e9, s.flight_key))

        slot_duration = global_vars.SLOT_DURATION_SECONDS
        capacity = getattr(global_vars, 'AIRCRAFT_PER_SLOT', 6)
        base_midnight = pd.Timestamp(now.normalize())

        # Build enriched working copies (aggregate last row per flight_key)
        latest: Dict[str, Dict[str, Any]] = {}
        for f in flights_list:
            fk = f['flight_key']
            # Keep most recent TOBT (assume higher timesec later if available)
            if fk not in latest or ('timesec' in f and 'timesec' in latest[fk] and f['timesec'] > latest[fk]['timesec']):
                latest[fk] = f

        candidates = []
        for fk, f in latest.items():
            if fk in self._frozen:
                continue
            tobt_epoch = f.get('tobt')
            if not tobt_epoch or pd.isna(tobt_epoch):
                continue
            tobt_ts = pd.to_datetime(tobt_epoch, unit='s')
            secs_midnight = (tobt_ts - tobt_ts.normalize()).total_seconds()
            base_slot = int(secs_midnight // slot_duration)
            # Skip blocked slots only during final assignment (not here)
            slot_start = base_midnight + pd.Timedelta(seconds=base_slot * slot_duration)
            f['_base_slot'] = base_slot
            f['_slot_start_epoch'] = int(slot_start.timestamp())
            candidates.append(f)

        if not candidates:
            return sorted(scheduled, key=lambda s: (s.rw_cur if s.rw_cur is not None else 1e9, s.flight_key))

        # Compute priorities
        for f in candidates:
            f['_priority'] = self._priority_tuple(f)

        # Sort globally by priority (stable)
        candidates.sort(key=lambda f: f['_priority'])

        # Assign respecting capacity (push forward if needed)
        slot_load: Dict[int, int] = {}
        # Seed with frozen flights
        for sl in scheduled:
            if sl.rw_cur is not None:
                slot_load[sl.rw_cur] = slot_load.get(sl.rw_cur, 0) + 1

        assigned_slots: Dict[str, int] = {}
        for f in candidates:
            target = f['_base_slot']
            # Advance until free capacity & not blocked
            while (target in self._blocked_slots) or (slot_load.get(target, 0) >= capacity):
                target += 1
            slot_load[target] = slot_load.get(target, 0) + 1
            assigned_slots[f['flight_key']] = target

        # Update shift counts
        for fk, new_slot in assigned_slots.items():
            prev = self._prev_assignment.get(fk)
            if prev is not None and prev != new_slot:
                self._shift_counts[fk] = self._shift_counts.get(fk, 0) + 1
            elif prev is None:
                self._shift_counts.setdefault(fk, 0)
            self._prev_assignment[fk] = new_slot

        # Build Slot objects
        for f in candidates:
            fk = f['flight_key']
            slot_idx = assigned_slots[fk]
            slot_start = base_midnight + pd.Timedelta(seconds=slot_idx * slot_duration)
            slot_end = slot_start + pd.Timedelta(seconds=slot_duration)
            tobt_epoch = f.get('tobt')
            tobt_ts = pd.to_datetime(tobt_epoch, unit='s') if tobt_epoch else None
            taxi_minutes = f.get('taxi_time_minutes')
            if taxi_minutes is None or pd.isna(taxi_minutes):
                raise ValueError(f"Missing taxi_time_minutes for flight {fk} in simple scheduler")
            taxi_delta = pd.Timedelta(minutes=float(taxi_minutes))
            candidate_tsat = slot_start - taxi_delta
            tsat = tobt_ts if (tobt_ts and tobt_ts > candidate_tsat) else candidate_tsat
            sl = Slot(
                flight_key=fk,
                rwy=f.get('trwy'),
                ttot=slot_start,
                tsat=tsat,
                ctot=pd.to_datetime(f['ctot'], unit='s') if f.get('ctot') and not pd.isna(f.get('ctot')) else None,
                status=PLANNED_STATUS,
                rw_cur=slot_idx,
                rw_cur_start=slot_start,
                rw_cur_end=slot_end,
                tobt=tobt_ts
            )
            scheduled.append(sl)

        # Final stable ordering
        scheduled.sort(key=lambda s: (s.rw_cur if s.rw_cur is not None else 1e9, s.flight_key))
        return scheduled


# ---------------------------------------------------------------------------
# Snapshot generation core
# ---------------------------------------------------------------------------

FULL_PIPELINE = object()  # sentinel indicating we must run the real scheduling pipeline


def _prepare_capacity_blocks(
    closure_context: Optional[Dict[str, Any]],
    slots: pd.DataFrame,
    *,
    active_runways: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize MRI-based capacity blocks for the provided slot template."""
    if not closure_context:
        return pd.DataFrame(), pd.DataFrame()

    def _normalize(values: Optional[Sequence[str]]) -> list[str]:
        if not values:
            return []
        return [str(v).strip() for v in values if v and str(v).strip()]

    def _unique_preserve(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in values:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    requested = _unique_preserve(_normalize(closure_context.get('requested_runways')))
    active = _unique_preserve(_normalize(active_runways) or _normalize(closure_context.get('active_runways')))

    if requested:
        runways = requested
    elif active:
        runways = active
    else:
        runways = _unique_preserve(_normalize(closure_context.get('runways_from_map')))

    # Only use the specific runway set relevant for this scheduling run.
    if not runways:
        raise ValueError("MRI closure context did not provide any runways")

    blocks_df, summary_df = build_takeoff_capacity_blocks(
        closure_context['maps'],
        closure_context['minutes'],
        slots,
        day=closure_context['day'],
        slot_duration_seconds=int(getattr(global_vars, 'SLOT_DURATION_SECONDS', 600)),
        runway_filter=runways,
    )
    return blocks_df, summary_df


def _build_state_dataframe(messages: pd.DataFrame) -> pd.DataFrame:
    """Create a per-minute flight state DataFrame from truncated history.

    Must emulate the important columns produced by load_TOBT_data_optimized for scheduling.
    Only uses information available up to current snapshot (messages already filtered).
    Bad / out-of-range epoch values are coerced to NaN to avoid overflow in pandas.to_datetime.
    """
    if messages.empty:
        return pd.DataFrame(columns=['flight_key'])
    df = _deduplicate_messages(messages)
    if df.empty:
        return df

    # Helper: sanitize epoch columns & collect invalid counts (overflow / out-of-range / non-finite)
    invalid_counts: Dict[str, int] = {}

    def _sanitize_epoch(col: str):
        if col not in df.columns:
            invalid_counts[col] = 0
            return
        # Coerce to numeric; infinities preserved so we can detect and null them
        df[col] = pd.to_numeric(df[col], errors='coerce')
        mask_invalid = (
            (df[col] < SAFE_EPOCH_MIN) |
            (df[col] > SAFE_EPOCH_MAX) |
            (~np.isfinite(df[col]))
        )
        invalid_counts[col] = int(mask_invalid.sum())
        if invalid_counts[col]:
            df.loc[mask_invalid, col] = np.nan

    for _c in ['tobt', 'ctot', 'atd']:
        _sanitize_epoch(_c)
    df.attrs['invalid_epoch_counts'] = invalid_counts

    # Safe datetime conversion
    def _safe_to_datetime(series: pd.Series) -> pd.Series:
        if series.empty:
            return series
        # Explicitly work on float copy to avoid pandas internal multiply on object/infinite
        s = series.astype('float64')
        # pandas still may emit RuntimeWarning for nan * 1e9; silence via np.errstate
        with np.errstate(all='ignore'):
            try:
                converted = pd.to_datetime(s, unit='s', errors='coerce')
            except Exception:
                converted = pd.to_datetime(pd.Series([np.nan]*len(s)), errors='coerce')
        return converted

    df['tobt_s'] = _safe_to_datetime(df['tobt']) if 'tobt' in df.columns else pd.NaT
    df['ctot_s'] = _safe_to_datetime(df['ctot']) if 'ctot' in df.columns else pd.NaT
    # If verbose later: we can surface invalid counts; store in attrs already.

    # Provide placeholder SOBT difference + CTOT updates counters if absent (algorithms expect them)
    if 'difference_tobt-sobt' not in df.columns:
        df['difference_tobt-sobt'] = 0
    if 'ctot_updates' not in df.columns:
        df['ctot_updates'] = 0

    # Provide timesec (message/event time) column if missing; approximate with TOBT epoch as fallback
    if 'timesec' not in df.columns:
        df['timesec'] = df['tobt'].fillna(0).astype('Int64')

    # Ensure sfplid exists
    if 'sfplid' not in df.columns:
        df['sfplid'] = range(1, len(df) + 1)

    if 'taxi_time_minutes' not in df.columns:
        raise ValueError('taxi_time_minutes column required for snapshot state build')
    df['taxi_time_minutes'] = pd.to_numeric(df['taxi_time_minutes'], errors='coerce')
    if df['taxi_time_minutes'].isna().any():
        missing = df.loc[df['taxi_time_minutes'].isna(), 'flight_key'].unique()[:5]
        raise ValueError(
            'Missing taxi_time_minutes for flights: ' + ', '.join(map(str, missing))
        )
    # Earliest feasible TTOT from TOBT+taxi, then clamp into CTOT window if present.
    df['sched_ttot'] = df['tobt'] + (df['taxi_time_minutes'] * 60)

    ctot_min = global_vars.CTOT_MIN_MARGIN * 60
    ctot_max = global_vars.CTOT_MAX_MARGIN * 60
    with_ctot = df['ctot'].notna()
    df['earliest_ok'] = np.where(with_ctot, df['ctot'] - ctot_min, -np.inf)
    df['latest_ok'] = np.where(with_ctot, df['ctot'] + ctot_max, np.inf)

    # Respect CTOT window strictly: TTOT cannot be before earliest_ok or after latest_ok.
    df.loc[with_ctot, 'sched_ttot'] = df.loc[with_ctot, ['sched_ttot','earliest_ok']].max(axis=1)
    over_mask = with_ctot & (df['sched_ttot'] > df['latest_ok'])
    df.loc[over_mask, 'sched_ttot'] = df.loc[over_mask, 'latest_ok']

    df['sched_ttot_s'] = _safe_to_datetime(df['sched_ttot'])
    secs_midnight = (df['sched_ttot_s'].dt.hour * 3600 +
                     df['sched_ttot_s'].dt.minute * 60 +
                     df['sched_ttot_s'].dt.second)
    df['rw_slot'] = (secs_midnight // global_vars.SLOT_DURATION_SECONDS).astype('Int64')

    for col in ['slot_shift','slot_shiftback','ctot_cancelled','last_ctot','prev_tobt_expired','shift_day']:
        if col not in df.columns:
            df[col] = np.nan if col != 'slot_shift' else 0

    midnight_epoch = df['sched_ttot'] - (df['sched_ttot'] % 86400)
    df['earliest_ok_sec'] = np.where(with_ctot, (df['earliest_ok'] - midnight_epoch) % 86400, -np.inf)
    df['latest_ok_sec'] = np.where(with_ctot, (df['latest_ok'] - midnight_epoch) % 86400, np.inf)
    return df


def  _run_full_scheduler(
    df_state: pd.DataFrame,
    frozen_assignments: Dict[str, int],
    blocked_slot_indices: Optional[set[int]] = None,
    now_ts: Optional[pd.Timestamp] = None,
    capacity_blocks: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the actual scheduling pipeline on df_state, preserving frozen slots.

    Returns (df_final_unique, slots, df_dep_day)
    """
    from slot_manager import initialize_slots, correct_df_newrw, adjust_df_cur_optimized
    from flight_scheduler import adapt_obp_vectorized, adapt_obp_moveback_vectorized
    # Initialize slots each run (idempotent given global config)
    slots = initialize_slots()
    # Respect permanently vacated capacity by reducing slot_size for those indices.
    # Without this, the scheduler could backfill a just-vacated slot, and later we also
    # append the synthetic VACATED__ entry for analysis, leading to apparent overfill.
    if ENFORCE_VACATED_CAPACITY_ON_FULL and blocked_slot_indices:
        for idx in blocked_slot_indices:
            if idx in slots.index:
                # Clamp at >= 0
                new_size = max(0, int(slots.at[idx, 'slot_size']) - 1)
                slots.at[idx, 'slot_size'] = new_size
    if df_state.empty and (capacity_blocks is None or capacity_blocks.empty):
        return df_state, slots, df_state
    # Provide initial current slot assignment = rw_slot
    df_run = df_state.copy()
    if capacity_blocks is not None and not capacity_blocks.empty:
        blocks_aug = capacity_blocks.copy()
        df_run = pd.concat([df_run, blocks_aug], ignore_index=True, sort=False)
    # Normalize dtypes early to avoid Int64 overhead (assumes rw_slot non-null after preprocessing)
    if 'rw_slot' in df_run.columns:
        try:
            df_run['rw_slot'] = df_run['rw_slot'].astype('int32')
        except Exception:
            pass  # leave as-is if nulls present
    # Optionally disallow assignment to past slots: clamp original slot to earliest feasible
    # slot whose start time minus taxi_time is >= now.
    if 'rw_slot' in df_run.columns and now_ts is not None and DISALLOW_PAST_SLOT_ASSIGNMENT:
        try:
            slot_sec = int(getattr(global_vars, 'SLOT_DURATION_SECONDS', 600))
            if 'taxi_time_minutes' not in df_run.columns:
                raise ValueError('taxi_time_minutes column required when disallowing past slots')
            taxi_minutes = pd.to_numeric(df_run['taxi_time_minutes'], errors='coerce')
            if taxi_minutes.isna().any():
                missing = df_run.loc[taxi_minutes.isna(), 'flight_key'].unique()[:5]
                raise ValueError(
                    'Missing taxi_time_minutes for flights: ' + ', '.join(map(str, missing))
                )
            base_mid = now_ts.normalize()
            secs_since_midnight = (now_ts - base_mid).total_seconds()
            earliest_start_sec = secs_since_midnight + (taxi_minutes * 60.0)
            earliest_slot_idx = np.ceil(earliest_start_sec / slot_sec)
            mask_valid = (~np.isnan(earliest_slot_idx)) & df_run['rw_slot'].notna()
            if mask_valid.any():
                current_slots = df_run.loc[mask_valid, 'rw_slot'].astype(float).to_numpy()
                earliest_slots = earliest_slot_idx[mask_valid]
                df_run.loc[mask_valid, 'rw_slot'] = np.maximum(current_slots, earliest_slots)
            try:
                df_run['rw_slot'] = df_run['rw_slot'].astype('int32')
            except Exception:
                pass
        except Exception:
            pass
    df_run['rw_cur'] = df_run['rw_slot']
    df_run = correct_df_newrw(df_run, verbose=0)
    # Apply previous frozen assignments BEFORE scheduling so algorithm treats them as in place
    for fk, slot_idx in frozen_assignments.items():
        mask = df_run['flight_key'] == fk
        if mask.any():
            df_run.loc[mask, 'rw_cur'] = slot_idx
    # Scheduling (forward) – keep full column set for downstream phases
    # Minimal dep-day frame to reduce copy cost but retain needed mutable counters
    base_cols = ['flight_key','rw_cur', 'rw_slot'] # added runway slot here, so to avoid crash
    for extra in ['slot_shift','shift_day','slot_shiftback']:
        if extra in df_run.columns:
            base_cols.append(extra)
    if 'is_mri_block' in df_run.columns:
        block_mask_values = df_run['is_mri_block'].to_numpy(dtype=bool, na_value=False)
        mask_blocks = pd.Series(block_mask_values, index=df_run.index)
        df_dep_day = df_run.loc[~mask_blocks, base_cols].copy()
    else:
        df_dep_day = df_run[base_cols].copy()
    # Ensure required columns exist
    for col in ['slot_shift','shift_day','slot_shiftback']:
        if col not in df_dep_day.columns:
            df_dep_day[col] = 0
    df_run, slots, df_dep_day = adapt_obp_vectorized(slots, df_run, [], 0, df_dep_day, verbose=0)
    # Adjust – must retain full df_dep_day (includes sched_ttot_s etc.)
    import time as _time
    df_run, df_dep_day = adjust_df_cur_optimized(df_run, df_dep_day, 0, _time.time(), _time.time() - 600)
    # Moveback phase
    _, _, slots, df_run, df_dep_day = adapt_obp_moveback_vectorized(slots, df_run, 0, df_dep_day, verbose=0)
    # Deduplicate by flight (latest slot_shift wins)
    df_run_sorted = df_run.sort_values(['flight_key','slot_shift'], ascending=[True, False])
    df_final = df_run_sorted.drop_duplicates(['flight_key'], keep='first')
    # Final dtype normalization
    for col in ['rw_cur','rw_slot']:
        if col in df_final.columns and pd.api.types.is_integer_dtype(df_final[col]):
            try:
                df_final[col] = df_final[col].astype('int32')
            except Exception:
                pass
    # Post-processing capacity repair: enforce frozen assignments as immutable and resolve collisions
    try:
        if frozen_assignments and 'rw_cur' in df_final.columns and 'flight_key' in df_final.columns and 'slot_size' in slots.columns:
            # Restore frozen placements for real flights (ignore synthetic VACATED__ which aren't in df_final)
            for fk, sidx in list(frozen_assignments.items()):
                if fk.startswith('VACATED__'):
                    continue
                mask = (df_final['flight_key'] == fk)
                if mask.any() and pd.notna(sidx):
                    try:
                        df_final.loc[mask, 'rw_cur'] = int(sidx)
                    except Exception:
                        pass
            # Build counts and capacity maps
            capacity_map = slots['slot_size'].to_dict()
            counts = df_final['rw_cur'].dropna().astype(int).value_counts().to_dict()
            frozen_keys = set(k for k in frozen_assignments.keys())
            # Resolve overfilled slots by pushing non-frozen flights forward
            changed = 0
            for s, c in sorted(list(counts.items())):
                cap = capacity_map.get(s)
                if cap is None or c <= cap:
                    continue
                excess = c - cap
                # Select non-frozen occupants in this slot in a stable order
                mask_s = (df_final['rw_cur'].astype(int) == int(s))
                cand_idx = df_final[mask_s & (~df_final['flight_key'].isin(frozen_keys))].index.tolist()
                if not cand_idx:
                    continue  # nothing to push; should not happen if frozen respected capacity originally
                # Push first 'excess' non-frozen flights
                for idx in cand_idx[:excess]:
                    cur_slot = int(df_final.at[idx, 'rw_cur'])
                    target = cur_slot + 1
                    while True:
                        if target not in capacity_map:
                            break
                        if counts.get(target, 0) < capacity_map.get(target, 0):
                            # move
                            df_final.at[idx, 'rw_cur'] = target
                            counts[cur_slot] = max(0, counts.get(cur_slot, 1) - 1)
                            counts[target] = counts.get(target, 0) + 1
                            changed += 1
                            break
                        target += 1
    except Exception:
        pass
    # Post-processing feasibility repair: push flights to later slots only if they cannot reach
    # the runway before the current slot ends (i.e., slot_end < now + taxi_time).
    # A flight can use a slot as long as it arrives before the slot ends, not when it starts.
    try:
        if now_ts is not None and 'rw_cur' in df_final.columns and {'slot_start_sec','slot_size'}.issubset(slots.columns):
            slot_start_map = slots['slot_start_sec'].to_dict()
            # Prefer explicit slot_end_sec column, otherwise derive from slot_start + duration
            if 'slot_end_sec' in slots.columns:
                slot_end_map = slots['slot_end_sec'].to_dict()
            else:
                # Compute slot end from slot start + duration
                slot_duration = int(getattr(global_vars, 'SLOT_DURATION_SECONDS', 600))
                slot_end_map = {k: (v + slot_duration) % 86400 for k, v in slot_start_map.items()}
            capacity_map = slots['slot_size'].to_dict()
            # Build current counts per slot
            counts = df_final['rw_cur'].dropna().astype(int).value_counts().to_dict()
            # Determine feasibility threshold (seconds since midnight)
            base_mid = now_ts.normalize()
            now_sec = int((now_ts - base_mid).total_seconds())
            # Flights we must not move (frozen allocations)
            frozen_keys = set(frozen_assignments.keys()) if isinstance(frozen_assignments, dict) else set()
            moves = 0
            for idx, row in df_final.iterrows():
                fk = str(row.get('flight_key'))
                if fk in frozen_keys:
                    continue
                cur_slot = int(row['rw_cur']) if pd.notna(row['rw_cur']) else None
                if cur_slot is None:
                    continue
                end_sec = slot_end_map.get(cur_slot)
                if end_sec is None:
                    continue
                taxi_minutes = row.get('taxi_time_minutes')
                if taxi_minutes is None or pd.isna(taxi_minutes):
                    raise ValueError(
                        f"Missing taxi_time_minutes for flight {fk} during feasibility repair"
                    )
                # Get TOBT to determine when flight will actually start taxiing
                tobt_val = row.get('tobt')
                if tobt_val is not None and pd.notna(tobt_val):
                    # tobt is in epoch seconds, convert to seconds since midnight
                    tobt_epoch = float(tobt_val)
                    tobt_midnight_epoch = tobt_epoch - (tobt_epoch % 86400)
                    tobt_sec = tobt_epoch - tobt_midnight_epoch
                else:
                    # If no TOBT, fallback to now
                    tobt_sec = now_sec
                # A flight can only start taxiing at max(now, TOBT)
                # If TOBT is in future, flight will depart on time (at TOBT + taxi)
                # If TOBT is in past but flight still here, earliest departure is now + taxi
                effective_start = max(now_sec, tobt_sec)
                earliest_ready = effective_start + float(taxi_minutes) * 60.0
                # If slot ends before earliest_ready, push forward (flight can't reach runway in time)
                if end_sec < earliest_ready:
                    target = cur_slot
                    # Advance until we find a slot whose end >= earliest_ready AND has capacity
                    while True:
                        target += 1
                        if target not in slot_end_map:
                            break
                        if slot_end_map[target] < earliest_ready:
                            continue
                        cap = capacity_map.get(target)
                        if cap is None:
                            continue
                        if counts.get(target, 0) < cap:
                            # apply move
                            counts[cur_slot] = max(0, counts.get(cur_slot, 1) - 1)
                            counts[target] = counts.get(target, 0) + 1
                            df_final.at[idx, 'rw_cur'] = target
                            # update slot_shift if present
                            if 'slot_shift' in df_final.columns and 'rw_slot' in df_final.columns:
                                orig = row.get('rw_slot') if pd.notna(row.get('rw_slot')) else cur_slot
                                try:
                                    df_final.at[idx, 'slot_shift'] = int(target) - int(orig)
                                except Exception:
                                    pass
                            moves += 1
                        # Whether moved or capacity not available, stop searching on first slot at/after earliest_ready
                        break
    except Exception:
        pass
    # Diagnostic: warn if any slot is over capacity (helps catch push logic regressions)
    try:
        if 'rw_cur' in df_final.columns and 'slot_size' in slots.columns:
            counts = df_final['rw_cur'].dropna().astype(int).value_counts().to_dict()
            for s, c in sorted(counts.items()):
                try:
                    cap = int(slots.at[s, 'slot_size']) if s in slots.index else None
                except Exception:
                    cap = None
                if cap is not None and c > cap:
                    print(f"WARNING: Overfill detected in slot {s}: {c} flights > capacity {cap}")
    except Exception:
        pass

    if 'taxi_time_minutes' not in df_final.columns and 'taxi_time_minutes' in df_state.columns:
        taxi_lookup = df_state[['flight_key', 'taxi_time_minutes']].drop_duplicates('flight_key')
        df_final = df_final.merge(taxi_lookup, on='flight_key', how='left')
        if 'taxi_time_minutes' not in df_dep_day.columns:
            df_dep_day = df_dep_day.merge(taxi_lookup, on='flight_key', how='left')

    return df_final, slots, df_dep_day


def _df_to_slots(df_final: pd.DataFrame, slots_df: pd.DataFrame) -> List[Slot]:
    """Vectorized conversion of schedule DataFrame to Slot objects."""
    if df_final.empty or 'rw_cur' not in df_final.columns:
        return []
    v = df_final[df_final['rw_cur'].notna()].copy()
    if v.empty:
        return []

    # --- Safety: sanitize raw epoch columns before datetime conversion to avoid overflow ---
    def _sanitize_epoch(series: pd.Series) -> pd.Series:
        if series is None or series.empty:
            return series
        # Coerce to numeric first
        s = pd.to_numeric(series, errors='coerce')
        # Always upcast to float to avoid downstream overflow when pandas multiplies by 1e9
        # during datetime conversion (int32 inputs would overflow under numpy.seterr(over='raise')).
        s = s.astype('float64', copy=False)
        mask_invalid = (
            (s < SAFE_EPOCH_MIN) |
            (s > SAFE_EPOCH_MAX) |
            (~np.isfinite(s))
        )
        if mask_invalid.any():
            s = s.where(~mask_invalid, np.nan)
        return s

    slot_idx_arr = v['rw_cur'].astype(int).to_numpy()
    fk_arr = v['flight_key'].astype(str).to_numpy()
    rwy_arr = v.get('trwy')
    if rwy_arr is not None:
        rwy_arr = rwy_arr.astype(str).fillna('').to_numpy()
    else:
        rwy_arr = np.array([''] * len(v))
    block_mask_arr = v['is_mri_block'].to_numpy(dtype=bool, na_value=False) if 'is_mri_block' in v.columns else np.zeros(len(v), dtype=bool)
    sched_dt = v.get('sched_ttot_s')
    if sched_dt is None:
        sched_dt = pd.to_datetime(v['tobt'], unit='s', errors='coerce')
    base_dates = sched_dt.dt.normalize().to_numpy()
    start_map = slots_df['slot_starttime'].to_dict() if 'slot_starttime' in slots_df.columns else {}
    end_map = slots_df['slot_endtime'].to_dict() if 'slot_endtime' in slots_df.columns else {}
    start_times = []
    end_times = []
    for s, b in zip(slot_idx_arr, base_dates):
        st = start_map.get(s)
        et = end_map.get(s)
        if st and et:
            b_date = pd.Timestamp(b).date()
            start_times.append(pd.Timestamp.combine(b_date, st))
            end_times.append(pd.Timestamp.combine(b_date, et))
        else:
            start_times.append(pd.NaT)
            end_times.append(pd.NaT)
    start_times = np.array(start_times, dtype='datetime64[ns]')
    end_times = np.array(end_times, dtype='datetime64[ns]')
    # Safe TOBT conversion
    if 'tobt' in v.columns:
        safe_tobt = _sanitize_epoch(v['tobt'])
        try:
            with np.errstate(all='ignore'):
                tobt_dt = pd.to_datetime(safe_tobt.astype('object'), unit='s', errors='coerce')
        except Exception as exc:
            tobt_values = safe_tobt.to_numpy(dtype='float64', copy=False)
            finite_mask = np.isfinite(tobt_values)
            finite_vals = safe_tobt[finite_mask]
            stats = {
                'count': int(finite_vals.size),
                'max': float(finite_vals.max()) if not finite_vals.empty else None,
                'min': float(finite_vals.min()) if not finite_vals.empty else None,
            }
            sample_vals = safe_tobt.head(5).tolist()
            raise RuntimeError(
                f"Failed to convert TOBT epochs to datetime: stats={stats}, sample={sample_vals}, dtype={safe_tobt.dtype}"
            ) from exc
    else:
        tobt_dt = pd.Series([pd.NaT]*len(v))
    if 'taxi_time_minutes' not in v.columns:
        raise ValueError('taxi_time_minutes column required to compute TSAT')
    taxi_minutes = pd.to_numeric(v['taxi_time_minutes'], errors='coerce')
    if taxi_minutes.isna().any():
        missing = v.loc[taxi_minutes.isna(), 'flight_key'].unique()[:5]
        raise ValueError(
            'Missing taxi_time_minutes for flights: ' + ', '.join(map(str, missing))
        )
    taxi_deltas = pd.to_timedelta(taxi_minutes, unit='m').to_numpy(dtype='timedelta64[ns]')
    candidate_tsat = start_times - taxi_deltas
    tobt_arr = tobt_dt.to_numpy(dtype='datetime64[ns]')
    valid_tobt = ~pd.isna(tobt_dt)
    tsat_arr = candidate_tsat.copy()
    mask_adv = valid_tobt.to_numpy() & (tobt_arr > candidate_tsat)
    tsat_arr[mask_adv] = tobt_arr[mask_adv]
    
    # Rule 1: TTOT = TSAT + taxi_time (not slot start)
    ttot_arr = tsat_arr + taxi_deltas
    
    if 'ctot' in v.columns:
        safe_ctot = _sanitize_epoch(v['ctot'])
        try:
            with np.errstate(all='ignore'):
                ctot_arr = pd.to_datetime(safe_ctot.astype('object'), unit='s', errors='coerce').to_numpy(dtype='datetime64[ns]')
        except Exception:
            # Fallback: all NaT if still failing (extreme corruption)
            ctot_arr = np.array([np.datetime64('NaT')] * len(v))
    else:
        ctot_arr = np.array([np.datetime64('NaT')] * len(v))
    
    # Rule 2: Respect CTOT windows - clamp TTOT to [CTOT - 5min, CTOT + 10min]
    ctot_min_margin = pd.Timedelta(minutes=global_vars.CTOT_MIN_MARGIN)  # 5 min
    ctot_max_margin = pd.Timedelta(minutes=global_vars.CTOT_MAX_MARGIN)  # 10 min
    for i in range(len(v)):
        if not pd.isna(ctot_arr[i]):
            ctot_ts = pd.Timestamp(ctot_arr[i])
            earliest_allowed = ctot_ts - ctot_min_margin
            latest_allowed = ctot_ts + ctot_max_margin
            current_ttot = pd.Timestamp(ttot_arr[i]) if not pd.isna(ttot_arr[i]) else None
            if current_ttot is not None:
                if current_ttot < earliest_allowed:
                    ttot_arr[i] = earliest_allowed
                elif current_ttot > latest_allowed:
                    ttot_arr[i] = latest_allowed
    
    # Rule 3: TTOT must remain within the assigned runway bin
    for i in range(len(v)):
        if not pd.isna(ttot_arr[i]) and not pd.isna(start_times[i]) and not pd.isna(end_times[i]):
            ttot_ts = pd.Timestamp(ttot_arr[i])
            bin_start = pd.Timestamp(start_times[i])
            bin_end = pd.Timestamp(end_times[i])
            if ttot_ts < bin_start:
                ttot_arr[i] = bin_start
            elif ttot_ts >= bin_end:
                # Constrain to just before bin end (1 second before)
                ttot_arr[i] = bin_end - pd.Timedelta(seconds=1)
    
    # Rule 4: No two flights may share the exact same TTOT
    # Group by slot and resolve conflicts with +1 second increments until unique
    order = np.lexsort((fk_arr, slot_idx_arr))  # Process in slot order, then by flight key
    used_ttots: set = set()
    for i in order:  # Process in slot order
        if pd.isna(ttot_arr[i]):
            continue
        ttot_ts = pd.Timestamp(ttot_arr[i])
        # Resolve conflicts deterministically by pushing forward
        while ttot_ts in used_ttots:
            ttot_ts = ttot_ts + pd.Timedelta(seconds=1)
        ttot_arr[i] = ttot_ts
        used_ttots.add(ttot_ts)
    
    status_arr = v['status'].astype(str).to_numpy() if 'status' in v.columns else np.array([PLANNED_STATUS] * len(v))
    out: List[Slot] = []
    for idx in order:
        row = v.iloc[idx]
        is_block = bool(block_mask_arr[idx])
        start_ts = None if pd.isna(start_times[idx]) else pd.Timestamp(start_times[idx])
        end_ts = None if pd.isna(end_times[idx]) else pd.Timestamp(end_times[idx])
        tsat_ts = None if is_block or pd.isna(tsat_arr[idx]) else pd.Timestamp(tsat_arr[idx])
        # Use the computed TTOT from ttot_arr (derived from TSAT + taxi_time, with CTOT/bin clamping)
        ttot_ts = None if is_block or pd.isna(ttot_arr[idx]) else pd.Timestamp(ttot_arr[idx])
        ctot_ts = None if pd.isna(ctot_arr[idx]) else pd.Timestamp(ctot_arr[idx])
        tobt_ts = None if pd.isna(tobt_dt.iloc[idx]) else pd.Timestamp(tobt_dt.iloc[idx])
        status_val = status_arr[idx] if isinstance(status_arr[idx], str) and status_arr[idx] else (PLANNED_STATUS if not is_block else 'BLOCKED')
        if is_block and status_val.upper() == PLANNED_STATUS:
            status_val = 'BLOCKED'
        block_reason = None
        block_meta = None
        if is_block:
            raw_reason = row.get('block_reason')
            block_reason = str(raw_reason) if raw_reason and not pd.isna(raw_reason) else None
            block_meta = {
                'slot_capacity': int(row.get('slot_capacity_units')) if pd.notna(row.get('slot_capacity_units')) else None,
                'usable_capacity': int(row.get('usable_capacity_units')) if pd.notna(row.get('usable_capacity_units')) else None,
                'consumed_capacity': int(row.get('consumed_capacity_units')) if pd.notna(row.get('consumed_capacity_units')) else None,
            }
            if pd.notna(row.get('closure_ratio')):
                block_meta['closure_ratio'] = float(row.get('closure_ratio'))
            if pd.notna(row.get('closure_seconds')):
                block_meta['closure_seconds'] = float(row.get('closure_seconds'))
            if pd.notna(row.get('avg_open_runways')):
                block_meta['avg_open_runways'] = float(row.get('avg_open_runways'))
            if pd.notna(row.get('total_runways_considered')):
                block_meta['total_runways_considered'] = int(row.get('total_runways_considered'))
            runways_considered_val = row.get('runways_considered')
            if isinstance(runways_considered_val, str) and runways_considered_val:
                block_meta['runways_considered'] = runways_considered_val
            block_meta = {k: v for k, v in block_meta.items() if v is not None}
            if not block_meta:
                block_meta = None
        out.append(Slot(
            flight_key=fk_arr[idx],
            rwy=rwy_arr[idx] if rwy_arr[idx] else None,
            ttot=ttot_ts,
            tsat=tsat_ts,
            ctot=ctot_ts,
            status=str(status_val),
            rw_cur=int(slot_idx_arr[idx]),
            rw_cur_start=start_ts,
            rw_cur_end=end_ts,
            tobt=tobt_ts,
            is_capacity_block=is_block,
            block_reason=block_reason,
            block_meta=block_meta,
        ))
    return out


def _change_minutes_from_history(
    data_source: DataSourceInterface,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    *,
    schedule_cols: Optional[Sequence[str]] = None,
    coalesce_gap: int = 0,
) -> Optional[list[pd.Timestamp]]:
    """Return minute timestamps where schedule-relevant data changes.

    Filters to changes in specified columns per flight; always includes first
    message for each flight. Optional coalescing merges dense sequences.
    """
    if schedule_cols is None:
        schedule_cols = ('tobt','ctot','trwy','ctot_cancelled','last_ctot')
    for attr in ("df_history", "df", "_df", "history", "history_df"):
        hist = getattr(data_source, attr, None)
        if isinstance(hist, pd.DataFrame):
            if {'timesec','flight_key'}.issubset(hist.columns):
                start_sec = int(day_start.timestamp())
                end_sec = int((day_end + pd.Timedelta(minutes=1)).timestamp())
                subset = hist[(hist['timesec'] >= start_sec) & (hist['timesec'] < end_sec)].copy()
                if subset.empty:
                    return [day_start]
                needed = [c for c in schedule_cols if c in subset.columns]
                if not needed:
                    mins = pd.to_datetime(subset['timesec'], unit='s', errors='coerce').dt.floor('min')
                    uniq = sorted(ts for ts in mins.dropna().unique() if day_start <= ts <= day_end)
                else:
                    subset.sort_values(['flight_key','timesec'], inplace=True, kind='mergesort')
                    change_masks = []
                    for col in needed:
                        prev = subset.groupby('flight_key', sort=False)[col].shift(1)
                        change_masks.append(subset[col] != prev)
                    change_any = np.logical_or.reduce(change_masks)
                    first_mask = subset.groupby('flight_key', sort=False)['timesec'].rank(method='first') == 1
                    event_rows = subset[change_any | first_mask]
                    mins = pd.to_datetime(event_rows['timesec'], unit='s', errors='coerce').dt.floor('min')
                    uniq = sorted(ts for ts in mins.dropna().unique() if day_start <= ts <= day_end)
                if not uniq or uniq[0] != day_start:
                    uniq.insert(0, day_start)
                if coalesce_gap > 0 and len(uniq) > 2:
                    coalesced = []
                    last_added = None
                    for ts in uniq:
                        if last_added is None or (ts - last_added) >= pd.Timedelta(minutes=coalesce_gap):
                            coalesced.append(ts)
                            last_added = ts
                    uniq = coalesced
                return uniq
            else:
                return None
    return None


def generate_day_snapshots(
    day_start_utc: pd.Timestamp,
    day_end_utc: pd.Timestamp,
    scheduler: Union[SchedulerInterface, object],
    data_source: DataSourceInterface,
    minute_stride: int = 1,
    *,
    verbose: bool = False,
    event_driven: bool = True,
    closure_context: Optional[Dict[str, Any]] = None,
) -> InMemorySnapshotStore:
    """Generate minute-by-minute snapshots using either simple or full pipeline.

    If scheduler is FULL_PIPELINE sentinel, the real scheduling algorithm (capacity & CTOT)
    is invoked each minute with only data up to t, and frozen past slots preserved.
    Set verbose=True to emit per-snapshot console lines.
    """
    if day_end_utc < day_start_utc:
        raise ValueError('day_end_utc must be >= day_start_utc')
    if minute_stride <= 0:
        raise ValueError('minute_stride must be positive')

    use_full = (scheduler is FULL_PIPELINE)
    capacity_blocks_df: pd.DataFrame = pd.DataFrame()
    block_slots_for_simple: List[Slot] = []
    slots_template: Optional[pd.DataFrame] = None
    if closure_context:
        slots_template = getattr(global_vars, 'slots', None)
        if slots_template is None or getattr(slots_template, 'empty', True):
            from slot_manager import initialize_slots as _init_slots
            slots_template = _init_slots()
        else:
            slots_template = slots_template.copy()
        try:
            capacity_blocks_df, _ = _prepare_capacity_blocks(
                closure_context,
                slots_template,
                active_runways=closure_context.get('active_runways'),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to build MRI capacity blocks: {exc}")

    if not use_full and hasattr(scheduler, 'set_capacity_blocks'):
        if capacity_blocks_df.empty:
            scheduler.set_capacity_blocks([])  # type: ignore[attr-defined]
        else:
            if slots_template is None:
                from slot_manager import initialize_slots as _init_slots
                slots_template = _init_slots()
            block_slots_for_simple = _df_to_slots(capacity_blocks_df, slots_template)
            scheduler.set_capacity_blocks(block_slots_for_simple)  # type: ignore[attr-defined]

    snapshots: Dict[pd.Timestamp, PlanningSnapshot] = {}
    config_hash = _config_hash()
    frozen_assignments: Dict[str, int] = {}
    frozen_meta: Dict[str, Slot] = {}
    # Track last known planning fields for frozen flights to detect late changes
    frozen_last_plan: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}
    # Track permanently blocked slot indices (inefficiencies) for simple scheduler
    blocked_slot_indices: set[int] = set()
    # Track flights that have been revived at least once (for freeze monotonic relaxation)
    revived_flights: set[str] = set()
    # Record meta about revivals per snapshot (list of dicts) -> stored in snapshot.metadata['revivals'] for debugging
    revivals_log: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}

    # Pre-compute change minutes if enabled
    change_minutes: list[pd.Timestamp] | None = None
    change_set: set[pd.Timestamp] = set()
    if event_driven:
        change_minutes = _change_minutes_from_history(
            data_source, day_start_utc, day_end_utc,
            schedule_cols=('tobt','ctot','trwy','ctot_cancelled','last_ctot'),
            coalesce_gap=0,
        )
        if change_minutes is None:
            if verbose:
                print("[EVENT] Could not derive change minutes -> fallback to full per-minute recompute.")
            event_driven = False
        else:
            change_set = set(change_minutes)
            if verbose:
                total = int(((day_end_utc - day_start_utc) / pd.Timedelta(minutes=1)) + 1)
                print(f"[EVENT] Detected {len(change_minutes)} change minutes out of {total} total "
                      f"({len(change_minutes)/total:.1%} recomputation density).")

    prev_snapshot: Optional[PlanningSnapshot] = None
    schedule_runs = 0
    current = day_start_utc

    # Single progress bar counting ONLY recompute (change) minutes; simple & minimal.
    total_recompute = len(change_minutes) if (event_driven and change_minutes is not None) else int(((day_end_utc - day_start_utc) / pd.Timedelta(minutes=minute_stride)) + 1)
    pbar = None
    if not verbose and sys.stderr.isatty():  # keep quiet in non-TTY / verbose mode
        pbar = tqdm(total=total_recompute, desc="Snapshots", unit="snaps" if change_minutes else "min")

    # Lightweight rolling hash state (flight_key XOR timesec) to avoid full JSON hashing every recompute
    last_hash_simple = 0
    SIMPLE_HASH_MOD = (1 << 64) - 1
    last_message_count = 0

    while current <= day_end_utc:
        recompute = (not event_driven) or (current in change_set)
        if recompute:
            # Full scheduler path
            messages = data_source.load_up_to(current)
            # Build quick lookup of latest TOBT/CTOT (epoch) for late-change detection
            late_updates_lookup: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
            if not messages.empty and {'flight_key','tobt','ctot'}.issubset(messages.columns):
                # take newest per flight_key (messages already <= current)
                msg_sorted = messages.sort_values(['flight_key','timesec'], ascending=[True, False]) if 'timesec' in messages.columns else messages
                dedup = msg_sorted.drop_duplicates('flight_key', keep='first')
                for _, r in dedup.iterrows():
                    tobt_epoch = int(r.tobt) if 'tobt' in r and pd.notna(r.tobt) else None
                    ctot_epoch = int(r.ctot) if 'ctot' in r and pd.notna(r.ctot) else None
                    late_updates_lookup[str(r.flight_key)] = (tobt_epoch, ctot_epoch)
            # Unified late update detection (simple + full pipeline) if feature enabled
            if ALLOW_LATE_REVIVAL and late_updates_lookup:
                revived_entries: List[Dict[str, Any]] = []
                to_unfreeze: list[str] = []
                tsat_missed_flights: set[str] = set()  # Track which flights missed their TSAT window
                for fk, sl in list(frozen_meta.items()):
                    if getattr(sl, 'is_capacity_block', False):
                        continue
                    if fk.startswith('VACATED__'):
                        continue  # synthetic placeholder
                    if fk not in late_updates_lookup:
                        continue
                    new_tobt_epoch, new_ctot_epoch = late_updates_lookup[fk]
                    old_tobt = sl.tobt
                    old_ctot = sl.ctot
                    old_tsat = sl.tsat  # Get the frozen flight's TSAT
                    new_tobt_ts = pd.to_datetime(new_tobt_epoch, unit='s') if new_tobt_epoch else None
                    new_ctot_ts = pd.to_datetime(new_ctot_epoch, unit='s') if new_ctot_epoch else None
                    fields_changed: List[str] = []
                    if 'tobt' in REVIVE_ON_FIELDS and new_tobt_ts is not None and (old_tobt is None or new_tobt_ts != old_tobt):
                        if new_tobt_ts >= current:
                            fields_changed.append('tobt')
                    if 'ctot' in REVIVE_ON_FIELDS and new_ctot_ts is not None and (old_ctot is None or new_ctot_ts != old_ctot):
                        if new_ctot_ts >= current:
                            fields_changed.append('ctot')
                    # Runway change detection only if scheduler uses it (presence of trwy in messages) -> simple path; for full path ignore for now
                    # (Extend later if full pipeline integrates runway choice logic)
                    if 'trwy' in REVIVE_ON_FIELDS and not fields_changed:
                        # Only detect if runway field available in messages and differs (approx via latest row in messages for fk)
                        if 'trwy' in messages.columns:
                            latest_trwy = messages[messages['flight_key']==fk].sort_values('timesec', ascending=False).head(1)['trwy']
                            if not latest_trwy.empty:
                                new_trwy = str(latest_trwy.iloc[0]) if pd.notna(latest_trwy.iloc[0]) else None
                                if new_trwy != sl.rwy:
                                    # Treat runway-only change as revival if future tobt / ctot also present (or unchanged)
                                    if (new_tobt_ts and new_tobt_ts >= current) or (new_ctot_ts and new_ctot_ts >= current):
                                        fields_changed.append('trwy')
                    if fields_changed:
                        # Check if TSAT was missed using the new TSAT-window based logic
                        # Pseudo code: 
                        #   if now < tsat - 5min: tsat_missed = False (planning phase)
                        #   else: if earliest_possible_new_tsat > tsat + 5min: tsat_missed = True
                        # Also triggers VACATED on CTOT changes after TSAT window expires
                        tsat_missed, vacate_reason = check_tsat_missed(
                            now=current,
                            old_tsat=old_tsat,
                            new_tobt=new_tobt_ts,
                            tolerance_minutes=5,
                            old_ctot=old_ctot,
                            new_ctot=new_ctot_ts,
                            fields_changed=fields_changed,
                        )
                        if tsat_missed:
                            tsat_missed_flights.add(fk)
                        # Block its existing slot if TSAT was missed (capacity loss)
                        if PRESERVE_VACATED_SLOTS and sl.rw_cur is not None and tsat_missed:
                            blocked_slot_indices.add(int(sl.rw_cur))
                        to_unfreeze.append(fk)
                        revived_entries.append({
                            'flight_key': fk,
                            'from_slot': sl.rw_cur,
                            'changed_fields': tuple(fields_changed),
                            'snapshot_time': current,
                            'tsat_missed': tsat_missed,
                            'vacate_reason': vacate_reason,
                        })
                if to_unfreeze:
                    for fk in to_unfreeze:
                        revived_flights.add(fk)
                        sl_prev = frozen_meta.pop(fk, None)
                        frozen_assignments.pop(fk, None)
                        # Only create VACATED placeholder if TSAT was actually missed
                        if sl_prev and sl_prev.rw_cur is not None and PRESERVE_VACATED_SLOTS and fk in tsat_missed_flights:
                            vac_fk = f"VACATED__{fk}__{sl_prev.rw_cur}"
                            vac_slot = Slot(
                                flight_key=vac_fk,
                                rwy=sl_prev.rwy,
                                ttot=sl_prev.ttot,
                                tsat=sl_prev.tsat,
                                ctot=sl_prev.ctot,
                                status='VACATED',
                                rw_cur=sl_prev.rw_cur,
                                rw_cur_start=sl_prev.rw_cur_start,
                                rw_cur_end=sl_prev.rw_cur_end,
                                tobt=sl_prev.tobt,
                            )
                            frozen_meta[vac_fk] = vac_slot
                            frozen_assignments[vac_fk] = sl_prev.rw_cur
                    # Inform simple scheduler about blocked indices
                    if not use_full and hasattr(scheduler, 'block_slot'):
                        for idx in blocked_slot_indices:
                            try:
                                scheduler.block_slot(idx)  # type: ignore
                            except Exception:
                                pass
                if revived_entries:
                    revivals_log[current] = revived_entries
            if use_full:
                df_state = _build_state_dataframe(messages)
                # Pass blocked_slot_indices so the scheduler reduces capacity on vacated slots
                df_final, slots_df, _ = _run_full_scheduler(
                    df_state,
                    frozen_assignments,
                    blocked_slot_indices,
                    now_ts=current,
                    capacity_blocks=capacity_blocks_df,
                )
                scheduled_slots = _df_to_slots(df_final, slots_df)
            else:
                flights = data_source.to_flights(messages)
                scheduler.warm_start(frozen_meta.values())  # type: ignore
                scheduled_slots = scheduler.schedule(flights, now=current)  # type: ignore

            new_sched: list[Slot] = []
            for sl in scheduled_slots:
                # Preserve already frozen flights
                if sl.flight_key in frozen_meta:
                    sl = Slot(**asdict(frozen_meta[sl.flight_key]))
                else:
                    if is_past(sl, current):
                        sl = Slot(
                            flight_key=sl.flight_key,
                            rwy=sl.rwy,
                            ttot=sl.ttot,
                            tsat=sl.tsat,
                            ctot=sl.ctot,
                            status=FREEZE_STATUS if sl.status != TAKEN_OFF_STATUS else TAKEN_OFF_STATUS,
                            rw_cur=sl.rw_cur,
                            rw_cur_start=sl.rw_cur_start,
                            rw_cur_end=sl.rw_cur_end,
                            tobt=sl.tobt,  # added
                        )
                        frozen_meta[sl.flight_key] = sl
                        if use_full and sl.rw_cur is not None:
                            frozen_assignments[sl.flight_key] = sl.rw_cur
                        # Track baseline plan for change detection
                        frozen_last_plan[sl.flight_key] = (sl.tobt, sl.ctot)
                new_sched.append(sl)
            # Add any synthetic VACATED__ frozen slots created during revival that are not in scheduled list
            for fk, sl in frozen_meta.items():
                if fk.startswith('VACATED__') and all(s.flight_key != fk for s in new_sched):
                    new_sched.append(sl)

            if use_full:
                scheduled_ids = {s.flight_key for s in new_sched}
                pending = tuple(sorted(
                    fk for fk in (df_state['flight_key'].tolist() if not df_state.empty else [])
                    if fk not in scheduled_ids
                ))
            else:
                pending = tuple()

            # Hash only on recompute minutes
            if messages.empty:
                data_cut_hash = hashlib.sha256(b'empty').hexdigest()
                last_hash_simple = 0
                last_message_count = 0
            else:
                # Fast simple rolling hash (commutative) on flight_key + timesec, fallback to full every 50 recomputes
                if {'flight_key','timesec'}.issubset(messages.columns):
                    simple_vals = (messages['flight_key'].astype(str) + ':' + messages['timesec'].astype(str)).values
                    cur_hash = last_hash_simple
                    for v in simple_vals:
                        cur_hash ^= hash(v) & SIMPLE_HASH_MOD
                        cur_hash = ((cur_hash << 13) | (cur_hash >> 51)) & SIMPLE_HASH_MOD  # tiny mixing
                    last_hash_simple = cur_hash
                    # Periodic strong hash to avoid collision accumulation or structural changes
                    if (schedule_runs % 50) == 0 or messages.shape[0] < last_message_count:
                        key_cols = ['flight_key', 'timesec', 'tobt', 'ctot']
                        avail = [c for c in key_cols if c in messages.columns]
                        stable = messages[avail].sort_values(avail).to_json(
                            orient='records', date_unit='s', date_format='epoch'
                        )
                        data_cut_hash = hashlib.sha256(stable.encode()).hexdigest()
                    else:
                        data_cut_hash = f"simple-{last_hash_simple:016x}"
                    last_message_count = messages.shape[0]
                else:
                    key_cols = ['flight_key', 'timesec', 'tobt', 'ctot']
                    avail = [c for c in key_cols if c in messages.columns]
                    stable = messages[avail].sort_values(avail).to_json(
                        orient='records', date_unit='s', date_format='epoch'
                    )
                    data_cut_hash = hashlib.sha256(stable.encode()).hexdigest()

            snap = PlanningSnapshot(
                snapshot_time=current,
                frozen_slot_ids=tuple(sorted(frozen_meta.keys())),
                scheduled_slots=tuple(sorted(
                    new_sched, key=lambda s: (s.rw_cur if s.rw_cur is not None else 10**9, s.flight_key)
                )),
                pending_queue=pending,
                metadata={
                    'config_hash': config_hash,
                    'data_cut_hash': data_cut_hash,
                    'mode': 'full' if use_full else 'simple',
                    'event_recompute': True,
                    'revivals': revivals_log.get(current, [])
                }
            )
            prev_snapshot = snap
            schedule_runs += 1
        else:
            # Carry snapshot forward; only advance freezing
            ps = prev_snapshot
            if ps is None:
                # No schedule yet (no messages before this minute)
                snap = PlanningSnapshot(
                    snapshot_time=current,
                    frozen_slot_ids=tuple(),
                    scheduled_slots=tuple(),
                    pending_queue=tuple(),
                    metadata={
                        'config_hash': config_hash,
                        'data_cut_hash': hashlib.sha256(b'empty').hexdigest(),
                        'mode': 'full' if use_full else 'simple',
                        'event_recompute': False
                    }
                )
                prev_snapshot = snap
            else:
                frozen_ids = set(ps.frozen_slot_ids)
                updated_slots: list[Slot] = []
                newly_frozen = False
                for sl in ps.scheduled_slots:
                    if sl.flight_key in frozen_ids:
                        updated_slots.append(sl)
                        continue
                    # Use TSAT + 5min window to determine freeze timing (not TTOT)
                    if sl.tsat is not None and sl.tsat + pd.Timedelta(minutes=5) < current:
                        frozen_ids.add(sl.flight_key)
                        newly_frozen = True
                        nf = Slot(
                            flight_key=sl.flight_key,
                            rwy=sl.rwy,
                            ttot=sl.ttot,
                            tsat=sl.tsat,
                            ctot=sl.ctot,
                            status=FREEZE_STATUS if sl.status != TAKEN_OFF_STATUS else TAKEN_OFF_STATUS,
                            rw_cur=sl.rw_cur,
                            rw_cur_start=sl.rw_cur_start,
                            rw_cur_end=sl.rw_cur_end,
                            tobt=sl.tobt,  # added
                        )
                        frozen_meta[sl.flight_key] = nf
                        updated_slots.append(nf)
                    else:
                        updated_slots.append(sl)

                snap = PlanningSnapshot(
                    snapshot_time=current,
                    frozen_slot_ids=tuple(sorted(frozen_ids)),
                    scheduled_slots=ps.scheduled_slots if not newly_frozen else tuple(updated_slots),
                    pending_queue=ps.pending_queue,
                    metadata={
                        'config_hash': config_hash,
                        'data_cut_hash': ps.metadata.get('data_cut_hash', ''),
                        'mode': ps.metadata.get('mode', ''),
                        'event_recompute': False,
                        'freeze_update': newly_frozen
                    }
                )
                prev_snapshot = snap

        snapshots[current] = prev_snapshot
        if verbose:
            tag = "R" if prev_snapshot.metadata.get('event_recompute') else "C"
            print(f"[SNAPSHOT{tag}] t={current.isoformat()} frozen={len(prev_snapshot.frozen_slot_ids)} "
                  f"total={len(prev_snapshot.scheduled_slots)}")

        current += pd.Timedelta(minutes=minute_stride)
        if pbar is not None and recompute:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Monotonic frozen set assertion
    prev = set()
    for ts in sorted(snapshots):
        cur = set(snapshots[ts].frozen_slot_ids)
        # Allow regression ONLY for revived flights (original key removed, replaced by VACATED__ key)
        removed = prev - cur
        illegal = {fk for fk in removed if not fk.startswith('VACATED__') and fk not in revived_flights}
        if illegal:
            raise AssertionError(f"Frozen set regression detected for non-revived flights: {illegal}")
        prev = cur

    if verbose and event_driven and change_minutes is not None:
        total = len(snapshots)
        print(f"[EVENT] Scheduler executed {schedule_runs} times ({schedule_runs/total:.1%} of minutes).")

    return InMemorySnapshotStore(snapshots)

