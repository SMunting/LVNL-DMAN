#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CTOT (Calculated Take-Off Time) analysis module.

This module contains functions for analyzing CTOT updates and cancellations
for flight scheduling optimization.
"""

import pandas as pd
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=128)
def conditions_flight_cached(group_tuple):
    """
    Cached version of conditions_flight with optimized calculation of CTOT stats
    
    Args:
        group_tuple: Tuple representation of group data for caching
        
    Returns:
        Dict with CTOT update information
    """
    # Convert tuple to DataFrame with minimal columns
    needed_columns = ['ctot', 'flight_key']
    group_df = pd.DataFrame(group_tuple, columns=needed_columns)
    
    # Prepare result dictionary
    result = {}
    
    # Calculate CTOT updates efficiently
    ctot_values = np.array(group_df['ctot'].values, dtype=float)
    non_nan_count = np.sum(~np.isnan(ctot_values))
    result['ctot_updates'] = non_nan_count
    
    # Check CTOT status
    if non_nan_count > 0:
        # Find non-NaN values efficiently
        non_nan_indices = np.where(~np.isnan(ctot_values))[0]
        if len(non_nan_indices) > 0:
            last_non_nan_idx = non_nan_indices[-1]
            result['last_ctot'] = ctot_values[last_non_nan_idx]
            # Check if last value in series is NaN (cancelled)
            result['ctot_cancelled'] = 1 if np.isnan(ctot_values[-1]) else 0
        else:
            result['last_ctot'] = np.nan
            result['ctot_cancelled'] = 0
    else:
        result['last_ctot'] = np.nan
        result['ctot_cancelled'] = 0
        
    # Add flight_key for DataFrame assembly
    if len(group_df) > 0:
        result['flight_key'] = group_df['flight_key'].iloc[0]
    
    return result


def conditions_obp_optimized(df):
    """
    Highly optimized version of conditions_obp that minimizes pandas overhead
    
    Args:
        df: DataFrame with flight information
        
    Returns:
        DataFrame with added CTOT statistics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Extract unique flight keys
    unique_flight_keys = df['flight_key'].dropna().unique()
    
    # Pre-allocate results list for better memory efficiency
    results = []
    
    # Process each unique flight key
    for flight_key in unique_flight_keys:
        # Use NumPy boolean indexing for faster filtering
        mask = (df['flight_key'].values == flight_key)
        
        # Extract only needed columns for processing
        group_data = df.loc[mask, ['ctot', 'flight_key']]
        
        if group_data.empty:
            continue
            
        # Convert to tuple format for caching
        group_tuple = tuple(map(tuple, group_data.values))
        
        # Process with cached function
        result_dict = conditions_flight_cached(group_tuple)
        
        # Append to results
        results.append(result_dict)
    
    # Convert results to DataFrame in one operation
    if results:
        result_df = pd.DataFrame(results)
        
        # Join with original DataFrame efficiently using flight_key
        merge_cols = ['flight_key']
        update_cols = ['ctot_updates', 'ctot_cancelled', 'last_ctot']
        
        # Only merge the necessary columns for better performance
        merge_df = result_df[merge_cols + update_cols]
        
        # Use left join to preserve all original rows
        df = pd.merge(df, merge_df, on=merge_cols, how='left')
        
        return df
        
    return df


def check_ctot_violations(df_final, slots, verbose=0):
    """
    Check for CTOT violations in the final schedule and report conflicts.
    
    Args:
        df_final: Final scheduled flights DataFrame
        slots: Slots DataFrame with timing information
        verbose: Verbosity level
    
    Returns:
        List of CTOT violations for reporting
    """
    import global_vars
    import pandas as pd
    # Removed unused datetime/time imports
    
    # Only check flights with CTOT
    ctot_flights = df_final[df_final['ctot'].notna()].copy()
    
    if ctot_flights.empty:
        return []
    
    violations = []
    
    # Get CTOT margins from global vars
    ctot_min_margin = global_vars.CTOT_MIN_MARGIN  # -5 minutes
    ctot_max_margin = global_vars.CTOT_MAX_MARGIN  # +10 minutes
    
    for _, flight in ctot_flights.iterrows():
        try:
            assigned_slot = int(flight['rw_cur'])
            slot_duration = global_vars.SLOT_DURATION_SECONDS
            if 'slot_start_sec' in slots.columns:
                slot_start_seconds = slots.at[assigned_slot, 'slot_start_sec']
                slot_center_seconds = slots.at[assigned_slot, 'slot_center_sec'] if 'slot_center_sec' in slots.columns else (slot_start_seconds + slot_duration/2)
                slot_end_seconds = (slot_start_seconds + slot_duration) % 86400
            else:
                slot_start_seconds = assigned_slot * slot_duration
                slot_center_seconds = slot_start_seconds + slot_duration/2
                slot_end_seconds = slot_start_seconds + slot_duration

            def sec_to_hms(sec):
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"

            slot_start_time = sec_to_hms(slot_start_seconds)
            slot_end_time = sec_to_hms(slot_end_seconds)

            ctot_val = flight['ctot']
            if pd.isna(ctot_val):
                continue
            ctot_epoch = int(ctot_val)

            # Prefer relative second window if available to avoid unit mismatch
            if 'earliest_ok_sec' in flight and 'latest_ok_sec' in flight and pd.notna(flight['earliest_ok_sec']) and pd.notna(flight['latest_ok_sec']):
                window_start_sec = float(flight['earliest_ok_sec'])
                window_end_sec = float(flight['latest_ok_sec'])
                # Treat the assigned slot as a time INTERVAL [start, end].
                # A violation occurs only if the entire slot lies before or after the window.
                # This prevents false "LATE" flags when the slot starts within the window
                # but the slot center happens to fall outside (coarse slots case).
                if slot_end_seconds < window_start_sec:  # slot completely before window
                    violation_type = "EARLY"
                    violation_minutes = (window_start_sec - slot_end_seconds) / 60.0
                elif slot_start_seconds > window_end_sec:  # slot completely after window
                    violation_type = "LATE"
                    violation_minutes = (slot_start_seconds - window_end_sec) / 60.0
                else:
                    violation_type = None  # Overlaps window => acceptable
                # For reporting convert window bounds to HH:MM:SS using relative seconds
                window_start = window_start_sec
                window_end = window_end_sec
                ctot_str = pd.to_datetime(ctot_epoch, unit='s').time().strftime('%H:%M:%S')
                window_start_str = sec_to_hms(window_start)
                window_end_str = sec_to_hms(window_end)
            else:
                # Fallback: use epoch based comparison (convert slot center to epoch via earliest_ok if present)
                if 'earliest_ok' in flight and 'earliest_ok_sec' in flight and pd.notna(flight['earliest_ok']) and pd.notna(flight['earliest_ok_sec']):
                    midnight_epoch = float(flight['earliest_ok']) - float(flight['earliest_ok_sec'])
                    slot_center_epoch = midnight_epoch + slot_center_seconds
                else:
                    slot_center_epoch = slot_center_seconds  # likely mismatched, may yield large minutes
                window_start = ctot_epoch - ctot_min_margin * 60
                window_end = ctot_epoch + ctot_max_margin * 60
                if slot_center_epoch < window_start:
                    violation_type = "EARLY"
                    violation_minutes = (window_start - slot_center_epoch)/60
                elif slot_center_epoch > window_end:
                    violation_type = "LATE"
                    violation_minutes = (slot_center_epoch - window_end)/60
                else:
                    violation_type = None
                ctot_str = pd.to_datetime(ctot_epoch, unit='s').time().strftime('%H:%M:%S')
                window_start_str = pd.to_datetime(window_start, unit='s').time().strftime('%H:%M:%S')
                window_end_str = pd.to_datetime(window_end, unit='s').time().strftime('%H:%M:%S')

            if violation_type:
                violation = {
                    'callsign': flight.get('acid', ''),
                    'sfplid': flight.get('sfplid', ''),
                    'ctot': ctot_str,
                    'ctot_window_start': window_start_str,
                    'ctot_window_end': window_end_str,
                    'assigned_slot': f"{slot_start_time}-{slot_end_time}",
                    'violation_type': violation_type,
                    'violation_minutes': round(float(abs(violation_minutes)), 1)
                }
                violations.append(violation)
                if verbose > 0:
                    print(f"DEBUG: CTOT violation {violation}")
        except Exception as e:
            if verbose > 0:
                print(f"DEBUG: Error evaluating CTOT violation: {e}")
            continue
    
    return violations

def print_ctot_violations(violations):
    """
    Print CTOT violation warnings in a clear format.
    
    Args:
        violations: List of violation dictionaries
    """
    if not violations:
        return
    
    print("\n" + "="*80)
    print("  SCHEDULING CONFLICT DETECTED - CTOT VIOLATIONS  ")
    print("="*80)
    print(f"Found {len(violations)} flight(s) scheduled outside their CTOT window:")
    print()
    
    # Sort violations by severity (violation minutes)
    violations_sorted = sorted(violations, key=lambda x: x['violation_minutes'], reverse=True)
    
    print("CALLSIGN   CTOT       CTOT WINDOW        ASSIGNED SLOT      TYPE   VIOLATION")
    print("-" * 80)
    
    for v in violations_sorted:
        print(f"{v['callsign']:<10} {v['ctot']:<10} {v['ctot_window_start']}-{v['ctot_window_end']:<12} "
              f"{v['assigned_slot']:<18} {v['violation_type']:<6} {v['violation_minutes']:>6.0f} min")
    
    print("-" * 80)
    print("   RECOMMENDATION: Review capacity constraints or adjust CTOT assignments")
    print("   These flights may need manual intervention or CTOT renegotiation")
    print("="*80)