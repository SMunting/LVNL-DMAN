#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Slot management module for flight scheduling system.

This module handles runway slot management, time calculations, and slot adjustments.
"""

import pandas as pd
import global_vars

# Simple module-level cache for slot mapping dictionaries to avoid rebuilding every call
_SLOT_MAP_CACHE = {
    'slots_id': None,
    'slot_endtime_map': None,
    'slot_starttime_map': None,
    'slot_start_sec_map': None,
    'slot_end_sec_map': None,
    'slot_center_sec_map': None,
}


def correct_df_newrw(df, verbose=0):
    """
    Optimized version of the correct_df_newrw function
    Updates runway slot endtimes and starttimes for flights with shifted days
    Uses vectorized operations for better performance
    
    Now handles cases where rw_cur column doesn't exist yet
    
    Args:
        df: DataFrame with flight data
        verbose: Verbosity level for additional output (default: 0)
        
    Returns:
        DataFrame with updated slot time calculations
    """
    # Fast-path cache: if slot timing columns already present and no day shift risk, skip recompute
    # (All rw_cur within single day and no shift_day flags set.)
    if {'rw_cur_endtime','rw_cur_starttime','rw_cur_end_sec','rw_cur_start_sec'}.issubset(df.columns):
        if 'rw_cur' in df.columns:
            # Use existing columns if all indices within current day and no shift_day set
            if df['rw_cur'].max(skipna=True) < global_vars.TOTAL_SLOTS and (('shift_day' not in df.columns) or (df['shift_day'].fillna(0).sum() == 0)):
                return df
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Check if rw_cur column exists, if not initialize it from rw_slot
    if 'rw_cur' not in df.columns:
        df['rw_cur'] = df['rw_slot'].copy()
    
    # Get the total slots per day based on current configuration (module-level import already present)
    slots_per_day = global_vars.TOTAL_SLOTS
    
    # Handle day shifts - use actual slots per day instead of hardcoded value
    day_shift_mask = df['rw_cur'] >= slots_per_day
    if day_shift_mask.any():
        df.loc[day_shift_mask, 'shift_day'] = 1
        df.loc[day_shift_mask, 'rw_cur'] = df.loc[day_shift_mask, 'rw_cur'] - slots_per_day
    
    # Get slots from global variable
    slots = global_vars.slots
    
    # Reuse cached dictionaries when slots object unchanged
    sid = id(slots)
    if _SLOT_MAP_CACHE['slots_id'] != sid:
        _SLOT_MAP_CACHE['slots_id'] = sid
        _SLOT_MAP_CACHE['slot_endtime_map'] = slots['slot_endtime'].to_dict()
        _SLOT_MAP_CACHE['slot_starttime_map'] = slots['slot_starttime'].to_dict()
        _SLOT_MAP_CACHE['slot_start_sec_map'] = slots['slot_start_sec'].to_dict() if 'slot_start_sec' in slots.columns else {}
        _SLOT_MAP_CACHE['slot_end_sec_map'] = slots['slot_end_sec'].to_dict() if 'slot_end_sec' in slots.columns else {}
        _SLOT_MAP_CACHE['slot_center_sec_map'] = slots['slot_center_sec'].to_dict() if 'slot_center_sec' in slots.columns else {}
    slot_endtime_map = _SLOT_MAP_CACHE['slot_endtime_map']
    slot_starttime_map = _SLOT_MAP_CACHE['slot_starttime_map']
    slot_start_sec_map = _SLOT_MAP_CACHE['slot_start_sec_map']
    slot_end_sec_map = _SLOT_MAP_CACHE['slot_end_sec_map']
    slot_center_sec_map = _SLOT_MAP_CACHE['slot_center_sec_map']
    
    # Debug the mapping between slot numbers and times
    if verbose > 0:
        print("DEBUG: Mapping slot numbers to times:")
        print(f"DEBUG: Slot 79: {slot_starttime_map.get(79.0)}-{slot_endtime_map.get(79.0)}")
        print(f"DEBUG: Slot 80: {slot_starttime_map.get(80.0)}-{slot_endtime_map.get(80.0)}")
        print(f"DEBUG: Slot 81: {slot_starttime_map.get(81.0)}-{slot_endtime_map.get(81.0)}")
        print(f"DEBUG: Slot 82: {slot_starttime_map.get(82.0)}-{slot_endtime_map.get(82.0)}")
    
    # Convert to integers to ensure proper mapping
    df['rw_cur_int'] = df['rw_cur'].astype(int)
    
    # Use vectorized mapping instead of iloc operations
    df['rw_cur_endtime'] = df['rw_cur_int'].map(slot_endtime_map)
    df['rw_cur_starttime'] = df['rw_cur_int'].map(slot_starttime_map)
    # Add second-based references for precise comparisons
    df['rw_cur_start_sec'] = df['rw_cur_int'].map(slot_start_sec_map)
    df['rw_cur_end_sec'] = df['rw_cur_int'].map(slot_end_sec_map)
    df['rw_cur_center_sec'] = df['rw_cur_int'].map(slot_center_sec_map)
    
    # Convert date and times to strings efficiently
    date_str = df['sched_ttot_s'].dt.date.astype(str)
    end_str = df['rw_cur_endtime'].astype(str)
    start_str = df['rw_cur_starttime'].astype(str)
    
    # Combine and convert to timestamp in one operation
    df['rw_cur_endtime_base'] = pd.to_datetime(
        date_str + ' ' + end_str
    ).astype('datetime64[s]').astype('int')/1000000000
    
    df['rw_cur_starttime_base'] = pd.to_datetime(
        date_str + ' ' + start_str
    ).astype('datetime64[s]').astype('int')/1000000000
    
    # Debug to verify distinct rw_cur values and their corresponding times
    if verbose > 0:
        print("DEBUG: Distinct rw_cur values and their times:")
        for slot in sorted(df['rw_cur_int'].unique()):
            times = df[df['rw_cur_int'] == slot][['rw_cur_starttime', 'rw_cur_endtime']].iloc[0]
            print(f"DEBUG: Slot {slot}: {times['rw_cur_starttime']}-{times['rw_cur_endtime']}")
    
    # Remove temporary column
    df = df.drop(columns=['rw_cur_int'])
    
    return df


def adjust_df_cur_optimized(df, df_dep_day, loop_slot, loop_time, loop_time_old):
    """
    Optimized version of adjust_df_cur using vectorized operations
    
    Args:
        df: DataFrame with flight data
        df_dep_day: DataFrame with departure day data
        loop_slot: Current slot being processed
        loop_time: Current time being processed
        loop_time_old: Previous time being processed
        
    Returns:
        Tuple of updated (df, df_dep_day) DataFrames
    """
    # Create boolean masks for all conditions at once for better performance
    mask1 = ((df['rw_slot'] > df['rw_cur']) & 
             (df['rw_slot'] >= loop_slot) & 
             (df['timesec'] >= loop_time))
    
    mask2 = ((df['rw_slot'] > df['rw_cur']) & 
             (df['timesec'] < loop_time))
    
    mask3 = ((df['rw_slot'] < df['rw_cur']) & 
             (df['rw_slot'] > loop_slot) & 
             (df['timesec'] >= loop_time))
    
    # Combine masks and apply updates in one go
    update_mask = mask1 | mask2 | mask3
    
    if update_mask.any():
        df.loc[update_mask, 'rw_cur'] = df.loc[update_mask, 'rw_slot']
    
    # Optimize the merge operation
    # Only extract the columns we actually need for the merge
    key_columns = ['flight_key']
    merge_df = df[key_columns + ['rw_cur']].copy()
    
    # Use efficient merge method
    merged = pd.merge(
        df_dep_day, 
        merge_df,
        on=key_columns,
        how='left', 
        suffixes=('_x', '_y')
    )
    
    # Update rw_cur column efficiently with boolean indexing
    update_mask = merged['rw_cur_y'].notna()
    if update_mask.any():
        merged.loc[update_mask, 'rw_cur_x'] = merged.loc[update_mask, 'rw_cur_y']
    
    # Clean up and finalize
    result_df = merged.drop(columns=['rw_cur_y']).rename(columns={'rw_cur_x': 'rw_cur'})
    
    return df, result_df


def initialize_slots(slot_duration_seconds=None, aircraft_per_slot=None, taxi_time_minutes=None,
                  ctot_min_margin=None, ctot_max_margin=None):
    """
    Initialize the slots DataFrame with time information
    
    Args:
        slot_duration_seconds: Optional custom slot duration in seconds
        aircraft_per_slot: Optional custom number of aircraft allowed per slot
    taxi_time_minutes: Deprecated; retained for backward compatibility but ignored
        ctot_min_margin: Optional custom minutes before CTOT still allowed
        ctot_max_margin: Optional custom minutes after CTOT still allowed
        
    Returns:
        DataFrame containing slot information
    """
    # Use the centralized slot configuration from global_vars
    slots = global_vars.configure_slots(
        duration_seconds=slot_duration_seconds,
        aircraft_per_slot=aircraft_per_slot,
        ctot_min_margin=ctot_min_margin,
        ctot_max_margin=ctot_max_margin
    )
    
    return slots
