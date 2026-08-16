#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for flight scheduling system.

This module contains helper functions for data display and reporting.
"""

import pandas as pd
import numpy as np
import datetime

# Columns that hold epoch-seconds timestamps (as opposed to day-relative
# second offsets like '*_sec'/'*_start_sec', which are left untouched).
EPOCH_SECOND_COLUMNS = [
    'timesec', 'eobt', 'etd', 'retd', 'atd', 'aobt', 'etot', 'tsat', 'ttot',
    'asrt', 'sobt', 'tobt', 'ctot', 'sched_ttot', 'earliest_ok', 'latest_ok',
    'last_ctot', 'last_ctot_x', 'last_ctot_y',
]

# Plausible Unix epoch-seconds bounds (year 2000 - 2100), used to avoid
# mis-converting columns that merely happen to share a name but aren't epoch.
_MIN_PLAUSIBLE_EPOCH = 946684800
_MAX_PLAUSIBLE_EPOCH = 4102444800


def convert_epoch_columns_to_datetime(df, columns=EPOCH_SECOND_COLUMNS):
    """Return a copy of df with epoch-seconds columns rendered as datetime strings.

    Only numeric columns are converted (columns already holding formatted
    datetime text, e.g. main_pairwise.py's 'tobt'/'ctot', are left as-is).
    +/-inf (used as "no bound") and NaN become blank/NaT.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        finite = df[col].where(np.isfinite(df[col]))
        non_null = finite.dropna()
        if not non_null.empty and not non_null.between(_MIN_PLAUSIBLE_EPOCH, _MAX_PLAUSIBLE_EPOCH).all():
            continue
        df[col] = pd.to_datetime(finite, unit='s', errors='coerce')
    return df


def print_flight_results(df, global_slots=None, verbose=0, extra_columns=None):
    """
    Print a nicely formatted table of flight results showing:
    - Callsign
    - Latest TOBT time
    - Latest CTOT time
    - Runway slot time window
    - Difference between end slot time and TOBT in minutes
    - Optional additional columns
    
    Args:
        df: DataFrame containing flight scheduling data
        global_slots: Optional slots DataFrame for mapping slot numbers to times
        verbose: Verbosity level for debug output
        extra_columns: List of additional DataFrame column names to display
    """
    # Debug distinct rw_cur values and their times
    if verbose > 0:
        print("DEBUG: Before final output, checking distinct slot times:")
    for slot in sorted(df['rw_cur'].unique()):
        slot_info = df[df['rw_cur'] == slot][['rw_cur_starttime', 'rw_cur_endtime']].iloc[0]
        if verbose > 0:
            print(f"DEBUG: Slot {slot}: {slot_info['rw_cur_starttime']}-{slot_info['rw_cur_endtime']}")
        
    # Fix the slot time mapping issue - using the global_slots parameter if provided
    if global_slots is not None:
        if verbose > 0:
            print("DEBUG: Re-mapping slot times from global slots DataFrame")
        # Convert to integer indices for reliable mapping
        slot_map = {}
        for idx in global_slots.index:
            slot_map[float(idx)] = {
                'start': global_slots.at[idx, 'slot_starttime'],
                'end': global_slots.at[idx, 'slot_endtime']
            }
        
        # Apply the mapping to each flight
        for idx, row in df.iterrows():
            slot_num = row['rw_cur']
            if pd.notna(slot_num) and slot_num in slot_map:
                df.at[idx, 'rw_cur_starttime'] = slot_map[slot_num]['start']
                df.at[idx, 'rw_cur_endtime'] = slot_map[slot_num]['end']
                
        # Debug after remapping
        if verbose > 0:
            print("DEBUG: After remapping, slot times:")
        for slot in sorted(df['rw_cur'].unique()):
            slot_info = df[df['rw_cur'] == slot][['rw_cur_starttime', 'rw_cur_endtime']].iloc[0]
            if verbose > 0:
                print(f"DEBUG: Slot {slot}: {slot_info['rw_cur_starttime']}-{slot_info['rw_cur_endtime']}")
    
    # Prepare column headers
    base_headers = ["CALLSIGN", "TOBT", "CTOT", "RWY SLOT", "MARGIN (min)"]
    
    # Add extra column headers if provided
    if extra_columns:
        # Validate extra columns exist in DataFrame
        valid_extra_columns = [col for col in extra_columns if col in df.columns]
        if len(valid_extra_columns) != len(extra_columns):
            missing = [col for col in extra_columns if col not in df.columns]
            if verbose > 0:
                print(f"DEBUG: Missing columns in DataFrame: {missing}")
        
        # Create headers for extra columns (truncate if too long)
        extra_headers = [col.upper()[:10] for col in valid_extra_columns]
        all_headers = base_headers + extra_headers
    else:
        valid_extra_columns = []
        all_headers = base_headers
    
    # Create header format string
    header_format = " ".join(["{:<10}" if i < 3 else "{:<17}" if i == 3 else "{:<10}" 
                             for i in range(len(all_headers))])
    
    # Calculate separator line length
    separator_length = 10 + 10 + 10 + 17 + 10 + (len(valid_extra_columns) * 11) - 1  # -1 for last space
    
    # Print header
    print("\n" + header_format.format(*all_headers))
    print("-" * separator_length)
    
    # Helper function to format extra column values
    def format_extra_value(value):
        """Format extra column values based on their type"""
        if pd.isna(value):
            return "N/A"
        elif isinstance(value, (datetime.datetime, datetime.time)):
            if isinstance(value, datetime.datetime):
                return value.strftime("%H:%M:%S")
            else:
                return value.strftime("%H:%M:%S")
        elif isinstance(value, (int, pd.Int64Dtype)) or (isinstance(value, float) and value.is_integer()):
            # Check if this looks like a Unix timestamp (seconds since epoch)
            # Valid range: roughly 2000-01-01 to 2100-01-01
            int_value = int(value)
            if 946684800 <= int_value <= 4102444800:  # Year 2000 to 2100
                try:
                    # Convert Unix timestamp to datetime and extract time
                    dt = pd.to_datetime(int_value, unit='s')
                    return dt.strftime("%H:%M:%S")
                except (ValueError, OSError):
                    # If conversion fails, treat as regular integer
                    return str(int_value)
            else:
                # Regular integer, not a timestamp
                return str(int_value)
        elif isinstance(value, float):
            # Check if this looks like a Unix timestamp with fractional seconds
            if 946684800 <= value <= 4102444800:  # Year 2000 to 2100
                try:
                    # Convert Unix timestamp to datetime and extract time
                    dt = pd.to_datetime(value, unit='s')
                    return dt.strftime("%H:%M:%S")
                except (ValueError, OSError):
                    # If conversion fails, treat as regular float
                    return str(round(value, 2))
            else:
                # Regular float, not a timestamp
                return str(round(value, 2))
        else:
            # Convert to string and truncate if too long
            str_value = str(value)
            return str_value[:10] if len(str_value) > 10 else str_value
    
    # Process each flight
    for _, row in df.iterrows():
        # Extract callsign
        callsign = row['acid']
        
        # Format TOBT time
        tobt_time = row['tobt_time'] if pd.notna(row['tobt_time']) else "N/A"
        if isinstance(tobt_time, datetime.time):
            tobt_time = tobt_time.strftime("%H:%M:%S")
        
        # Format CTOT time
        ctot_time = row['ctot_time'] if pd.notna(row['ctot_time']) else "N/A"
        if isinstance(ctot_time, datetime.time):
            ctot_time = ctot_time.strftime("%H:%M:%S")
        
        # Format runway slot
        slot_start = row['rw_cur_starttime']
        slot_end = row['rw_cur_endtime']
        if isinstance(slot_start, datetime.time) and isinstance(slot_end, datetime.time):
            slot_start = slot_start.strftime("%H:%M:%S")
            slot_end = slot_end.strftime("%H:%M:%S")
            rwy_slot = f"{slot_start}-{slot_end}"
        else:
            rwy_slot = "N/A"
        
        # Calculate difference between TOBT and end slot time in minutes
        if pd.notna(row['tobt_time']) and pd.notna(row['rw_cur_endtime']):
            # Convert times to minutes since midnight
            tobt_minutes = row['tobt_time'].hour * 60 + row['tobt_time'].minute
            slot_end_minutes = row['rw_cur_endtime'].hour * 60 + row['rw_cur_endtime'].minute
            
            # Better handling of time differences across midnight
            if slot_end_minutes < tobt_minutes and (tobt_minutes - slot_end_minutes) > 1000:
                # This is likely a case where slot_end is after midnight
                slot_end_minutes += 24 * 60
            elif tobt_minutes < slot_end_minutes and (slot_end_minutes - tobt_minutes) > 1000:
                # This is likely a case where tobt is after midnight
                tobt_minutes += 24 * 60
            
            # Calculate the difference properly
            taxi_minutes = row.get('taxi_time_minutes')
            if pd.notna(taxi_minutes):
                diff_minutes = (slot_end_minutes - tobt_minutes) - float(taxi_minutes)
            else:
                diff_minutes = "N/A"
        else:
            diff_minutes = "N/A"
        
        # Prepare base row data
        row_data = [callsign, tobt_time, ctot_time, rwy_slot, diff_minutes]
        
        # Add extra column values
        for col in valid_extra_columns:
            formatted_value = format_extra_value(row[col])
            row_data.append(formatted_value)
        
        # Print row with proper alignment
        print(header_format.format(*row_data))
    
