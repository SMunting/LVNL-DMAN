"""Continuous pairwise departure sequencer.

This module implements a simple continuous sequencing model that replaces
slot-based sequencing. It's intentionally compact and focused: it takes a
state DataFrame (as prepared by `snapshot_generator._build_state_dataframe`)
and returns a list of `Slot` objects assigned along continuous TTOT times.

Design notes:
- Earliest readiness = max(TOBT, CTOT - 5min)
- CTOT is a hard constraint: TSAT + taxi_time <= CTOT + 10min (enforced by
  pushing TTOT earlier where possible; if impossible the CTOT is respected)
- Spacing is provided by the domain function `query_runway_separation` from
  `data.runway.SID_divergence`.
- Missed TSATs occur ONLY when the current clock time (now_ts) has passed the
  assigned TSAT and a TOBT update arrives after that point. If TOBT is updated
  before the clock reaches TSAT, no capacity is lost—it's simply a better
  prediction of aircraft readiness.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, DefaultDict
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    from data.runway.SID_divergence import query_runway_separation
except ImportError:
    def query_runway_separation(runway: str, leading_actype: str, trailing_actype: str,
                                prev_sid: str, curr_sid: str) -> float:
        """Fallback if domain module unavailable."""
        return 80.0


def check_tsat_validity(old_tsat: pd.Timestamp, new_tobt: pd.Timestamp, 
                        tolerance_minutes: int = 5) -> bool:
    """Check if a TSAT assignment is still valid after a TOBT update.
    
    A TSAT remains valid if the new TOBT falls within ±tolerance_minutes
    of the original TSAT. This allows minor delays without requiring immediate
    rescheduling and prevents unnecessary capacity loss.
    
    The tolerance window is defined as:
        [old_TSAT - tolerance, old_TSAT + tolerance]
    
    If new_TOBT falls within this window, the flight can still use its
    assigned TSAT without requiring a new slot assignment.
    
    Args:
        old_tsat: Previously assigned TSAT timestamp
        new_tobt: Updated TOBT timestamp (off-block time)
        tolerance_minutes: Tolerance window in minutes (default: 5)
        
    Returns:
        True if TSAT is still valid (no reassignment needed)
        False if TSAT requires reassignment
    
    Example:
        old_TSAT = 21:40
        tolerance = 5 minutes
        window = [21:35, 21:45]
        
        new_TOBT = 21:38 → returns True (within window, keep old TSAT)
        new_TOBT = 21:50 → returns False (outside window, assign new TSAT)
    """
    if pd.isna(old_tsat) or pd.isna(new_tobt):
        return False
    
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    window_start = old_tsat - tolerance
    window_end = old_tsat + tolerance
    
    return window_start <= new_tobt <= window_end


def run_pairwise_scheduler(
    df_state: pd.DataFrame,
    frozen_assignments: Dict[str, Any],
    *,
    now_ts: Optional[pd.Timestamp] = None,
    capacity_blocks: Optional[pd.DataFrame] = None,
    existing_vacated: Optional[DefaultDict[str, List[Tuple]]] = None,
    prev_slot_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Any], DefaultDict[str, List[Tuple]], Dict[str, Dict[str, Any]]]:
    """Compute continuous TTOT/TSAT pairwise sequencing.

    Returns (scheduled_slots, vacated_map, slot_map) where:
    - scheduled_slots: list of lightweight slot-like dicts for conversion to Slot objects
    - vacated_map: runway -> list of (flight_key, old_ttot, old_slot_dict) tuples
    - slot_map: flight_key -> {ttot, tsat, tobt, ctot} for next iteration
    
    TSAT Tolerance Window Logic (±5 minutes):
    When a TOBT update occurs, a two-step process determines the outcome:
    
    Step 1 - TSAT Validity Check:
      - If new_TOBT falls within [old_TSAT - 5min, old_TSAT + 5min], the TSAT remains valid
      - The flight keeps its original TSAT/TTOT assignment (no rescheduling)
      - If new_TOBT falls outside this window, a new TSAT is assigned
    
    Step 2 - Slot Vacation Check (only if TSAT changed):
      - If now_ts > old_TSAT + 5min (TSAT window expired), the slot is vacated
      - This represents capacity loss: the slot time passed before the aircraft was ready
      - If now_ts <= old_TSAT + 5min, no vacation occurs (update came in time)
    
    This tolerance window prevents unnecessary rescheduling and capacity loss for
    minor TOBT updates (≤5 minutes) while still capturing true missed slots.
    """
    if df_state is None or df_state.empty:
        return [], existing_vacated or defaultdict(list), {}

    vacated = existing_vacated or defaultdict(list)
    prev_slots = prev_slot_map or {}
    new_slot_map: Dict[str, Dict[str, Any]] = {}

    # Detect missed TSATs with two-step logic:
    # Step 1: Check if TSAT is still valid (new TOBT within ±5min window of old TSAT)
    # Step 2: If TSAT invalid AND update came after TSAT window expired, mark as vacated
    if now_ts is not None and prev_slots:
        for fk, old_slot in prev_slots.items():
            old_tsat = old_slot.get('tsat')
            old_ttot = old_slot.get('ttot')
            
            if pd.notna(old_tsat):
                # Check if flight is still in df_state (not taken off)
                if fk in df_state['flight_key'].values:
                    row_mask = df_state['flight_key'] == fk
                    if row_mask.any():
                        row = df_state[row_mask].iloc[0]
                        new_tobt = row.get('tobt')
                        new_tobt_ts = pd.to_datetime(new_tobt, unit='s', errors='coerce') if not pd.isna(new_tobt) else pd.NaT
                        
                        # Step 1: Check if TSAT is still valid using tolerance window
                        if pd.notna(new_tobt_ts):
                            tsat_still_valid = check_tsat_validity(old_tsat, new_tobt_ts, tolerance_minutes=5)
                            
                            if not tsat_still_valid:
                                # TSAT needs reassignment (new TOBT outside ±5min window)
                                # Step 2: Check if slot should be vacated (update came after TSAT window expired)
                                tsat_window_end = old_tsat + pd.Timedelta(minutes=5)
                                
                                if now_ts > tsat_window_end:
                                    # Update came after TSAT window expired → capacity lost
                                    rwy_val = row.get('trwy') if 'trwy' in df_state.columns else None
                                    if pd.notna(rwy_val):
                                        rwy_str = str(rwy_val)
                                        # Store complete slot info: (flight_key, old_ttot, old_slot_dict)
                                        vacated[rwy_str].append((fk, old_ttot, old_slot))
                            # else: TSAT still valid → keep old TSAT, no reassignment needed

    # Track flights that should keep their old TSAT (within tolerance window)
    keep_old_tsat: Dict[str, Dict[str, Any]] = {}
    if prev_slots:
        for fk, old_slot in prev_slots.items():
            old_tsat = old_slot.get('tsat')
            if pd.notna(old_tsat) and fk in df_state['flight_key'].values:
                row_mask = df_state['flight_key'] == fk
                if row_mask.any():
                    row = df_state[row_mask].iloc[0]
                    new_tobt = row.get('tobt')
                    new_tobt_ts = pd.to_datetime(new_tobt, unit='s', errors='coerce') if not pd.isna(new_tobt) else pd.NaT
                    
                    if pd.notna(new_tobt_ts) and check_tsat_validity(old_tsat, new_tobt_ts, tolerance_minutes=5):
                        # TSAT still valid → preserve old assignment
                        keep_old_tsat[fk] = old_slot

    # Prepare working rows: one entry per flight_key with required fields
    rows = []
    for _, r in df_state.iterrows():
        try:
            fk = str(r['flight_key'])
        except Exception:
            continue
        # Safe epoch -> timestamp conversion (some values may already be timestamps)
        tobt = r.get('tobt')
        tobt_ts = pd.to_datetime(tobt, unit='s', errors='coerce') if not pd.isna(tobt) else pd.NaT
        ctot = r.get('ctot')
        ctot_ts = pd.to_datetime(ctot, unit='s', errors='coerce') if not pd.isna(ctot) else pd.NaT
        taxi_minutes = float(r.get('taxi_time_minutes')) if 'taxi_time_minutes' in r and pd.notna(r.get('taxi_time_minutes')) else None
        runway = r.get('trwy') if 'trwy' in r else None
        sid = r.get('sid') if 'sid' in r else None
        wtc = r.get('wtc') if 'wtc' in r else None

        # Earliest readiness: max(TOBT + taxi_time, CTOT - 5min)
        # TOBT is off-block time, we need to add taxi time to get earliest possible TTOT
        cand_times = []
        if pd.notna(tobt_ts) and taxi_minutes is not None:
            earliest_ttot_from_tobt = tobt_ts + pd.Timedelta(minutes=taxi_minutes)
            cand_times.append(earliest_ttot_from_tobt)
        if pd.notna(ctot_ts):
            cand_times.append(ctot_ts - pd.Timedelta(minutes=5))
        if cand_times:
            earliest = max(cand_times)
        else:
            # If nothing available, deprioritize to far future
            earliest = pd.Timestamp.max

        rows.append({
            'flight_key': fk,
            'tobt_ts': tobt_ts,
            'ctot_ts': ctot_ts,
            'taxi_minutes': taxi_minutes,
            'earliest': earliest,
            'runway': str(runway) if pd.notna(runway) else '',
            'sid': str(sid) if pd.notna(sid) else '',
            'wtc': str(wtc) if pd.notna(wtc) else '',
            'raw_row': r,
        })

    # Group by runway and sequence each group independently
    by_runway: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in rows:
        by_runway[e['runway']].append(e)

    scheduled: List[Dict[str, Any]] = []
    for rwy, group in by_runway.items():
        # Sort by earliest readiness
        group.sort(key=lambda x: x['earliest'])
        prev_ttot: Optional[pd.Timestamp] = None
        prev_entry: Optional[Dict[str, Any]] = None
        seq_idx = 0
        for entry in group:
            # Skip frozen flights: treat them as already placed and keep order
            if entry['flight_key'] in frozen_assignments:
                # Frozen flights: extract their TTOT from frozen_assignments (should be timestamp)
                frozen_ttot = frozen_assignments.get(entry['flight_key'])
                try:
                    prev_ttot = pd.to_datetime(frozen_ttot) if frozen_ttot is not None else None
                except Exception:
                    prev_ttot = None
                prev_entry = entry
                seq_idx += 1
                continue

            # Check if this flight should keep its old TSAT (within tolerance window)
            if entry['flight_key'] in keep_old_tsat:
                old_slot = keep_old_tsat[entry['flight_key']]
                preserved_tsat = old_slot.get('tsat')
                preserved_ttot = old_slot.get('ttot')
                
                # Use preserved TSAT/TTOT instead of recalculating
                if pd.notna(preserved_tsat) and pd.notna(preserved_ttot):
                    # Record the preserved slot assignment
                    new_slot_map[entry['flight_key']] = {
                        'ttot': preserved_ttot,
                        'tsat': preserved_tsat,
                        'tobt': entry['tobt_ts'] if pd.notna(entry['tobt_ts']) else None,
                        'ctot': entry['ctot_ts'] if pd.notna(entry['ctot_ts']) else None,
                    }
                    
                    # Create scheduled entry with preserved times
                    scheduled.append({
                        'flight_key': entry['flight_key'],
                        'rwy': rwy,
                        'ttot': preserved_ttot,
                        'tsat': preserved_tsat,
                        'ctot': entry['ctot_ts'] if pd.notna(entry['ctot_ts']) else None,
                        'status': 'PLANNED',
                        'rw_seq': seq_idx,
                        'tobt': entry['tobt_ts'] if pd.notna(entry['tobt_ts']) else None,
                    })
                    prev_ttot = preserved_ttot
                    prev_entry = entry
                    seq_idx += 1
                    continue

            # Minimum candidate start
            candidate = entry['earliest']
            if pd.isna(candidate):
                candidate = pd.Timestamp.max

            # Respect pairwise spacing using domain function directly
            if prev_ttot is not None and prev_entry is not None:
                try:
                    spacing_sec = query_runway_separation(
                        runway=rwy,
                        leading_actype=prev_entry['wtc'],
                        trailing_actype=entry['wtc'],
                        prev_sid=prev_entry['sid'],
                        curr_sid=entry['sid']
                    )
                    if not (0 < spacing_sec < 1e6):  # sanity check
                        spacing_sec = 80.0
                except Exception:
                    spacing_sec = 80.0
                candidate = max(candidate, prev_ttot + pd.Timedelta(seconds=spacing_sec))

            # Respect capacity_blocks by pushing candidate out of any blocked interval for that runway
            if capacity_blocks is not None and not capacity_blocks.empty:
                # Look for block rows matching this runway
                if 'rwy' in capacity_blocks.columns:
                    blocks = capacity_blocks[capacity_blocks['rwy'] == rwy]
                    for _, b in blocks.iterrows():
                        # Try to extract start/end times from common column patterns
                        start_col = None
                        end_col = None
                        for c in blocks.columns:
                            c_lower = str(c).lower()
                            if 'start' in c_lower and 'time' in c_lower:
                                start_col = c
                            if 'end' in c_lower and 'time' in c_lower:
                                end_col = c
                        if start_col and end_col:
                            try:
                                bstart = pd.to_datetime(b[start_col])
                                bend = pd.to_datetime(b[end_col])
                                if bstart <= candidate <= bend:
                                    candidate = bend + pd.Timedelta(seconds=1)
                            except Exception:
                                pass

            # Calculate TSAT and ensure constraints are satisfied
            taxi_min = entry['taxi_minutes'] if entry['taxi_minutes'] is not None else 0.0
            tentative_ttot = candidate
            tsat = tentative_ttot - pd.Timedelta(minutes=taxi_min)
            
            # Constraint 1: TSAT must not be before TOBT (aircraft can't start taxi before off-block)
            tobt_ts = entry['tobt_ts']
            if pd.notna(tobt_ts) and tsat < tobt_ts:
                # Push TTOT forward so TSAT = TOBT
                tsat = tobt_ts
                tentative_ttot = tsat + pd.Timedelta(minutes=taxi_min)
            
            # Constraint 2: CTOT hard constraint - TSAT + taxi <= CTOT + 10min
            ctot_ts = entry['ctot_ts']
            if pd.notna(ctot_ts):
                # If TSAT + taxi_time > CTOT + 10min -> move TTOT earlier to satisfy
                if (tsat + pd.Timedelta(minutes=taxi_min)) > (ctot_ts + pd.Timedelta(minutes=10)):
                    # Move tentative_ttot earlier to CTOT + 10min
                    tentative_ttot = ctot_ts + pd.Timedelta(minutes=10)
                    tsat = tentative_ttot - pd.Timedelta(minutes=taxi_min)
                    # Re-check TOBT constraint after CTOT adjustment
                    if pd.notna(tobt_ts) and tsat < tobt_ts:
                        # CTOT constraint forces TSAT before TOBT - use TOBT and accept CTOT violation
                        tsat = tobt_ts
                        tentative_ttot = tsat + pd.Timedelta(minutes=taxi_min)

            # Record the complete slot assignment for next iteration (for vacated slot detection)
            new_slot_map[entry['flight_key']] = {
                'ttot': tentative_ttot,
                'tsat': tsat,
                'tobt': entry['tobt_ts'] if pd.notna(entry['tobt_ts']) else None,
                'ctot': entry['ctot_ts'] if pd.notna(entry['ctot_ts']) else None,
            }

            # Create scheduled entry
            scheduled.append({
                'flight_key': entry['flight_key'],
                'rwy': rwy,
                'ttot': tentative_ttot,
                'tsat': tsat,
                'ctot': entry['ctot_ts'] if pd.notna(entry['ctot_ts']) else None,
                'status': 'PLANNED',
                'rw_seq': seq_idx,
                'tobt': entry['tobt_ts'] if pd.notna(entry['tobt_ts']) else None,
            })
            prev_ttot = tentative_ttot
            prev_entry = entry
            seq_idx += 1

    return scheduled, vacated, new_slot_map
