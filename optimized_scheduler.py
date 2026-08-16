#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized Flight Scheduling Script

This script contains optimized versions of key functions from the Outbound_scheduler_Jens_new_csv.py
script, addressing performance bottlenecks in DataFrame operations, groupby, conditions_obp, 
and adapt_obp functions. The goal is to reduce the runtime from 47 seconds to a more reasonable time.
"""

import pandas as pd
import numpy as np
import datetime
import pathlib as pl
import os
import time
from functools import lru_cache
import warnings

# Suppress unnecessary warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', message='.*Downcasting.*', category=FutureWarning)
# pd.options.mode.chained_assignment = None  # default='warn'

# Global slot definition
# Define slot sizes
slots = pd.DataFrame(index=range(144))


def load_TOBT_data_optimized(csv_name):
    """
    Optimized data loading function with smarter filtering and type handling
    """
    path_data = pl.Path('custom_test_scenarios')
    
    # Read only the columns we need directly
    df = pd.read_csv(path_data / f"{csv_name}.csv", low_memory=False,
                     usecols=['timesec', 'sfplid', 'acid', 'adep', 'dest', 'eobt', 'actype', 'wtc',
                              'etd', 'retd', 'atd', 'aobt', 'etot', 'tsat', 'ttot', 'asrt', 'sobt', 'tobt', 'ctot',
                              'trwy'])

    # Initial filtering efficiently
    mask = (df.trwy.notna()) & (df.trwy != '0')
    df = df[mask].copy()
    
    df['trwy'] = df['trwy'].astype(str)
    
    # Convert date columns efficiently with vectorized operations
    date_columns = ['timesec', 'eobt', 'atd', 'aobt', 'sobt', 'tobt', 'ctot']
    for col in date_columns:
        # Only process columns that exist and have string values
        if col in df.columns and df[col].dtype == 'object':
            mask = df[col].notna() & (df[col] != '')
            df.loc[mask, col] = pd.to_datetime(df.loc[mask, col], format='%d/%m/%y %H:%M').astype('int64') // 10**9

    # Process outbound flights efficiently with boolean indexing
    df['dt'] = pd.to_datetime(df['timesec'], unit='s')
    mask = (df['adep'] == 'EHAM') & (df['dest'] != 'EHAM')
    df = df[mask].copy()

    # Identify threads with vectorized operations
    df.sort_values(['sfplid', 'timesec'], inplace=True)
    df['dif'] = df.timesec.diff(1)
    df['thread'] = np.where(abs(df.dif) > 2, 1, 0)
    df['thread'] = df.thread.cumsum()
    df['new'] = df['thread'].diff(-1) == -1
    df.loc[df.index[-1], 'new'] = True
    
    # Extract unique flights - use latest update for each sfplid-acid combination
    # First, sort by timesec descending to get the latest update first
    df.sort_values(['sfplid', 'acid', 'timesec'], ascending=[True, True, False], inplace=True)
    
    # Then, drop duplicates keeping only the first entry (which will be the latest due to our sorting)
    df_dep = df.drop_duplicates(subset=['sfplid', 'acid'], keep='first').copy()
    
    # Add derived columns efficiently
    df_dep.loc[:, 'airline'] = df_dep.acid.str[:3]
    df_dep.loc[:, 'atd_s'] = pd.to_datetime(df_dep['atd'], unit='s')
    df_dep.loc[:, 'eobt_s'] = pd.to_datetime(df_dep['eobt'], unit='s')
    df_dep.loc[:, 'date'] = df_dep['eobt_s'].dt.date
    df_dep.loc[:, 'tsat_s'] = pd.to_datetime(df_dep['tsat'], unit='s')
    df_dep.loc[:, 'tsat_time'] = df_dep['tsat_s'].dt.time
    df_dep.loc[:, 'tobt_s'] = pd.to_datetime(df_dep['tobt'], unit='s')
    df_dep.loc[:, 'tobt_time'] = df_dep['tobt_s'].dt.time
    
    # Scheduling calculations
    df_dep.loc[:, 'sched_ttot'] = df_dep['tobt']
    
    # Handle CTOT efficiently
    df_dep.loc[:, 'ctot'] = pd.to_numeric(df_dep['ctot'], errors='coerce')
    df_dep.loc[df_dep['ctot'] == 0.0, 'ctot'] = np.nan
    df_dep.loc[:, 'ctot_s'] = pd.to_datetime(df_dep['ctot'], unit='s', errors='coerce')
    df_dep.loc[df_dep.ctot_s.dt.year == 1970, 'ctot_s'] = np.nan
    
    # Apply CTOT constraints using vectorized operations
    valid_ctot_mask = df_dep['ctot'].notna()
    plus_mask = valid_ctot_mask & (df_dep['ctot']+10*60 > df_dep['sched_ttot'])
    minus_mask = valid_ctot_mask & (df_dep['ctot']-5*60 < df_dep['sched_ttot'])
    
    df_dep.loc[plus_mask, 'sched_ttot'] = df_dep.loc[plus_mask, 'ctot'] + 10*60
    df_dep.loc[minus_mask, 'sched_ttot'] = df_dep.loc[minus_mask, 'ctot'] - 5*60
    
    # Additional derived fields
    df_dep.loc[:, 'sched_ttot_s'] = pd.to_datetime(df_dep['sched_ttot'], unit='s')
    df_dep.loc[:, 'sched_ttot_time'] = pd.to_datetime(df_dep['tobt']+15*60, unit='s').dt.time
    df_dep.loc[:, 'ctot_time'] = df_dep['ctot_s'].dt.time
    df_dep.loc[:, 'sobt_s'] = pd.to_datetime(df_dep['sobt'], unit='s')
    df_dep.loc[:, 'sobt_time'] = df_dep['sobt_s'].dt.time
    df_dep.loc[:, 'difference_tobt-sobt'] = df_dep.tobt - df_dep.sobt
    df_dep.loc[:, 'difference_tobt-sobt'] = pd.to_datetime(df_dep['difference_tobt-sobt'], unit='s')
    df_dep.loc[:, 'taxi'] = round((df_dep['ttot'] - df_dep['tsat'])/60)
    
    # Calculate runway slots using vectorized operations
    df_dep.loc[:, 'rw_slot'] = (df_dep['sched_ttot_s'].dt.hour * 6 + df_dep['sched_ttot_s'].dt.minute / 10)
    df_dep.loc[:, 'rw_slot'] = df_dep['rw_slot'].apply(np.floor)
    
    # Filter and initialize
    df_dep = df_dep[df_dep['tobt'] != 0.0]
    df_dep.loc[:, 'slot_shift'] = 0
    df_dep.loc[:, 'slot_shiftback'] = 0
    df_dep.loc[:, 'ctot_cancelled'] = np.nan
    df_dep.loc[:, 'last_ctot'] = np.nan
    df_dep.loc[:, 'prev_tobt_expired'] = np.nan
    df_dep.loc[:, 'shift_day'] = np.nan

    # No need to call drop_duplicates again since we already selected unique flights
    # by sfplid and acid in the earlier step
    
    # Get flights with ATD
    atd_mask = df_dep.atd_s >= pd.to_datetime('2019-01-01')
    df_atd = df_dep[atd_mask][['acid', 'sfplid', 'date', 'atd']].copy()
    
    # No need to group again, we already have unique flights
    df_atd = df_atd.sort_values('acid')
    
    # Join with ATD data efficiently - keep only flights that actually took off
    df_dep = pd.merge(df_dep, df_atd[['acid', 'sfplid', 'date']], 
                      on=['acid', 'sfplid', 'date'], how='inner')
                      
    # Capture the complete update history for each flight for CTOT analysis
    # This DataFrame contains all updates for each flight
    df_history = df.copy()
    # Add date column to history data for CTOT analysis
    if 'date' not in df_history.columns:
        df_history['eobt_s'] = pd.to_datetime(df_history['eobt'], unit='s')
        df_history['date'] = df_history['eobt_s'].dt.date
    # We'll pass this to conditions_obp_optimized for proper CTOT update tracking
    
    return df, df_atd, df_dep, df_history


def optimized_selection(df, slot_size):
    """
    Efficiently select flights to keep vs push based on priority scores
    
    Args:
        df: DataFrame with priority columns
        slot_size: Maximum number of flights to keep
    
    Returns:
        tuple: (keep_df, push_df) containing flights to keep and push
    """
    if df.empty or slot_size <= 0:
        return pd.DataFrame(), df.copy() if not df.empty else pd.DataFrame()
        
    # Convert priority NaN values to large numbers to ensure correct sorting
    priority_columns = [f'priority_{i}' for i in range(1, 9)]
    df_sorted = df.copy()
    
    # Use numpy's efficient operations to replace NaNs with a large value
    for col in priority_columns:
        if col in df_sorted.columns:
            df_sorted[col] = np.nan_to_num(df_sorted[col].values, nan=999999)
    
    # Sort all at once with stable sort
    df_sorted = df_sorted.sort_values(
        priority_columns,
        ascending=[True] * len(priority_columns),
        na_position='last',
        kind='stable'  # Use stable sort to maintain order for equal values
    )
    
    # Simple slice to get keep and push flights
    keep_df = df_sorted.iloc[:slot_size].copy() if len(df_sorted) > 0 else pd.DataFrame()
    push_df = df_sorted.iloc[slot_size:].copy() if len(df_sorted) > slot_size else pd.DataFrame()
    
    return keep_df, push_df


def correct_df_newrw(df):
    """
    Optimized version of the correct_df_newrw function
    Updates runway slot endtimes and starttimes for flights with shifted days
    Uses vectorized operations for better performance
    
    Now handles cases where rw_cur column doesn't exist yet
    """
    # Check if rw_cur column exists, if not initialize it from rw_slot
    if 'rw_cur' not in df.columns:
        df['rw_cur'] = df['rw_slot'].copy()
    
    # Handle day shifts
    day_shift_mask = df['rw_cur'] == 144
    if day_shift_mask.any():
        df.loc[day_shift_mask, 'shift_day'] = 1
        df.loc[day_shift_mask, 'rw_cur'] = 0
    
    # Pre-compute slot mappings once - reference external slots DataFrame
    global slots
    
    # Use efficient dictionary mapping for slot lookups
    slot_endtime_map = slots['slot_endtime'].to_dict()
    slot_starttime_map = slots['slot_starttime'].to_dict()
    
    # Use vectorized mapping instead of iloc operations
    df['rw_cur_endtime'] = df['rw_cur'].map(slot_endtime_map)
    df['rw_cur_starttime'] = df['rw_cur'].map(slot_starttime_map)
    
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
    
    return df


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
    needed_columns = ['ctot', 'acid', 'sfplid', 'date']
    group_df = pd.DataFrame(group_tuple, columns=needed_columns)
    
    # Prepare result dictionary
    result = {}
    
    # Calculate CTOT updates efficiently
    ctot_values = group_df['ctot'].values
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
        
    # Add keys for DataFrame assembly
    if len(group_df) > 0:
        result['acid'] = group_df['acid'].iloc[0]
        result['sfplid'] = group_df['sfplid'].iloc[0]
        result['date'] = group_df['date'].iloc[0]
    
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
    
    # Extract unique groups efficiently using numpy operations
    # Get unique combinations of acid, sfplid, date
    unique_keys = df[['acid', 'sfplid', 'date']].drop_duplicates().values
    
    # Pre-allocate results list for better memory efficiency
    results = []
    
    # Process each unique combination
    for acid, sfplid, date in unique_keys:
        # Use NumPy boolean indexing for faster filtering
        mask = ((df['acid'].values == acid) & 
                (df['sfplid'].values == sfplid) & 
                (df['date'].values == date))
        
        # Extract only needed columns for processing
        group_data = df.loc[mask, ['ctot', 'acid', 'sfplid', 'date']]
        
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
        
        # Join with original DataFrame efficiently
        merge_cols = ['acid', 'sfplid', 'date']
        update_cols = ['ctot_updates', 'ctot_cancelled', 'last_ctot']
        
        # Only merge the necessary columns for better performance
        merge_df = result_df[merge_cols + update_cols]
        
        # Use left join to preserve all original rows
        df = pd.merge(df, merge_df, on=merge_cols, how='left')
        
        return df
        
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
    key_columns = ['acid', 'sfplid']
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


def adapt_obp_vectorized(slots, df_run, slotlist, loop_slot, df_dep_day):
    """
    Vectorized version of adapt_obp that minimizes iterations and DataFrame operations
    
    Args:
        slots: DataFrame with slot information
        df_run: DataFrame with flight data
        slotlist: List of slots
        loop_slot: Current slot being processed
        df_dep_day: DataFrame with departure day information
        
    Returns:
        Tuple of (df_run, slots, df_dep_day)
    """
    # Ensure slot_nr is properly defined
    if 'slot_nr' not in slots.columns:
        slots['slot_nr'] = slots.index
    
    # Create a mask for slots to process
    process_slots = slots['slot_nr'] > loop_slot
    
    if not process_slots.any():
        return df_run, slots, df_dep_day
    
    # Get slots that need to be processed, sorted
    slot_indices = sorted(slots[process_slots].index)
    
    # Create lookup dictionaries for slot info
    slot_sizes = slots['slot_size'].to_dict()
    
    # Process each slot without creating unnecessary DataFrames
    for slot in slot_indices:
        # Skip if not applicable
        if slots.at[slot, 'execute_push'] != 1:
            continue
        
        slot_size = slot_sizes.get(slot, 0)
        
        # Create efficient masks for flight filtering
        slot_flights_mask = (df_run['rw_slot'] == slot)
        
        if not slot_flights_mask.any():
            continue
            
        # Count flights
        flight_count = slot_flights_mask.sum()
        
        # Only process if we have more flights than slots
        if flight_count <= slot_size:
            continue
        
        # Extract only needed flights efficiently
        slot_flights = df_run.loc[slot_flights_mask]
        
        # Use our optimized selection logic
        keep, push = optimized_selection(slot_flights, slot_size)
        
        if push.empty:
            continue
            
        # Update slot shift counter for pushed flights using vectorized operations
        push_acid_set = set(push['acid'])
        push_sfplid_set = set(push['sfplid'])
        push_date_set = set(push['date'])
        
        # Create an efficient mask
        update_mask = (df_run['acid'].isin(push_acid_set) & 
                      df_run['sfplid'].isin(push_sfplid_set) & 
                      df_run['date'].isin(push_date_set))
                                     
        if update_mask.any():
            # Update in a single vectorized operation
            df_run.loc[update_mask, 'slot_shift'] += 1
            df_run.loc[update_mask, 'rw_cur'] += 1
            
            # Handle day shift
            day_shift_mask = df_run.loc[update_mask, 'rw_cur'] == 144
            if day_shift_mask.any():
                day_shift_indices = df_run.index[update_mask][day_shift_mask]
                df_run.loc[day_shift_indices, 'shift_day'] = 1
                df_run.loc[day_shift_indices, 'rw_cur'] = 0
                
            # Update corresponding flights in df_dep_day
            update_dep_mask = (df_dep_day['acid'].isin(push_acid_set) & 
                              df_dep_day['sfplid'].isin(push_sfplid_set) & 
                              df_dep_day['date'].isin(push_date_set))
                              
            if update_dep_mask.any():
                df_dep_day.loc[update_dep_mask, 'slot_shift'] += 1
                df_dep_day.loc[update_dep_mask, 'rw_cur'] += 1
                
                # Handle day shift in df_dep_day too
                dep_day_shift_mask = df_dep_day.loc[update_dep_mask, 'rw_cur'] == 144
                if dep_day_shift_mask.any():
                    dep_day_shift_indices = df_dep_day.index[update_dep_mask][dep_day_shift_mask]
                    df_dep_day.loc[dep_day_shift_indices, 'shift_day'] = 1
                    df_dep_day.loc[dep_day_shift_indices, 'rw_cur'] = 0
    
    # Update slot counts efficiently 
    for slot in slots.index:
        slots.at[slot, 'count'] = (df_run['rw_cur'] == slot).sum()
    
    return df_run, slots, df_dep_day


def adapt_obp_moveback_vectorized(slots, df_run, loop_slot, df_dep_day):
    """
    Vectorized version of adapt_obp_moveback that minimizes iteration and unnecessary operations
    
    Args:
        slots: DataFrame with slot information
        df_run: DataFrame with flight data
        loop_slot: Current slot being processed
        df_dep_day: DataFrame with departure day information
        
    Returns:
        Tuple of updated DataFrames
    """
    # Pre-allocate empty DataFrames
    moveback_lst = pd.DataFrame()
    nothing = pd.DataFrame()
    
    # Get relevant slots efficiently
    relevant_slots = slots[(slots['slot_nr'] > loop_slot) & (slots['execute_moveback'] == 1)]
    
    if relevant_slots.empty:
        return moveback_lst, nothing, slots, df_run, df_dep_day
    
    # Process each relevant slot
    for slot, slot_row in relevant_slots.iterrows():
        slot_size_local = slot_row['slot_size']
        slots_available = slot_row['count_verschil']
        
        # Create efficient masks for filtering
        condition1 = ((df_run['rw_slot'] <= slot) & 
                     (df_run['rw_slot'] < df_run['rw_cur']) & 
                     (df_run['rw_cur'] > slot))
                     
        condition2 = ((df_run['rw_slot'] > slot) & 
                     (df_run['rw_slot'] > df_run['rw_cur']) & 
                     (df_run['rw_cur'] < slot) & 
                     (df_run['shift_day'] == 1))
        
        # Check if we have flights to process
        if not condition1.any() and not condition2.any():
            continue
            
        # Update slot count
        curr_slot_count = (df_run['rw_cur'] == slot).sum()
        slots.loc[slot, 'count'] = curr_slot_count
        
        # Only proceed if we have room in the slot
        if curr_slot_count >= slot_size_local:
            continue
        
        # Extract relevant flights efficiently
        flights1 = df_run[condition1]
        flights2 = df_run[condition2]
        
        # Combine efficiently
        if flights1.empty and flights2.empty:
            continue
        elif flights1.empty:
            moveback_lst = flights2.copy()
        elif flights2.empty:
            moveback_lst = flights1.copy()
        else:
            moveback_lst = pd.concat([flights1, flights2])
        
        if moveback_lst.empty:
            continue
        
        # Initialize priority columns efficiently
        priority_cols = [f'priority_{i}' for i in range(1, 9)]
        for col in priority_cols:
            moveback_lst[col] = np.nan
        
        moveback_lst['priority_5'] = 999
        
        # Set priorities using vectorized operations
        # Priority 1 - CTOT Updates
        moveback_lst.loc[moveback_lst['ctot_updates'] >= 10, 'priority_1'] = 1
        
        # Priority 2 - Slot Shifts
        moveback_lst.loc[moveback_lst['slot_shift'] > 2, 'priority_2'] = 1
        
        # Priority 3 - CTOT Violations
        ctot_violation_mask = ((moveback_lst['ctot_s'] + pd.Timedelta(minutes=5)).dt.time < 
                              moveback_lst['rw_cur_endtime'])
        moveback_lst.loc[ctot_violation_mask, 'priority_3'] = 1
        
        complex_violation_mask = ((moveback_lst['ctot_cancelled'] == 0) & 
                                 (pd.to_datetime(moveback_lst['last_ctot']-60*10, unit='s').dt.time < 
                                 moveback_lst['rw_cur_endtime']) & 
                                 (pd.to_datetime(df_run['last_ctot']+60*10, unit='s').dt.time > 
                                 df_run['rw_cur_endtime']) & 
                                 (pd.to_datetime(df_run['last_ctot'], unit='s').dt.time <= 
                                 df_run['rw_cur_endtime']))
        moveback_lst.loc[complex_violation_mask, 'priority_3'] = 1
        
        # Priority 4 - CTOT Cancellations
        ctot_cancel_mask = ((moveback_lst['ctot_cancelled'] > 0) & 
                           (pd.to_datetime(moveback_lst['last_ctot'], unit='s').dt.time > 
                           moveback_lst['rw_cur_endtime']) & 
                           (pd.to_datetime(df_run['last_ctot']+60*20, unit='s').dt.time > 
                           df_run['rw_cur_endtime']) & 
                           (pd.to_datetime(df_run['last_ctot']-60*5, unit='s').dt.time < 
                           df_run['rw_cur_endtime']))
        moveback_lst.loc[ctot_cancel_mask, 'priority_4'] = 1
        
        # Priority 5 - Single Slot Shifts
        moveback_lst.loc[moveback_lst['slot_shift'] >= 1, 'priority_5'] = 2
        moveback_lst.loc[moveback_lst['slot_shift'] > 1, 'priority_5'] = 1
        
        # Priority 6 - TOBT Sorting (vectorized version)
        moveback_lst = moveback_lst.sort_values(by=['tobt_s'])
        unique_tobt_vals = moveback_lst['tobt_s'].unique()
        tobt_rank_map = {val: i+1 for i, val in enumerate(unique_tobt_vals)}
        moveback_lst['priority_6'] = moveback_lst['tobt_s'].map(tobt_rank_map)
        
        # Priority 7 - TOBT-SOBT Difference (vectorized version)
        moveback_lst = moveback_lst.sort_values(by=['difference_tobt-sobt'], ascending=False)
        unique_diff_vals = moveback_lst['difference_tobt-sobt'].unique()
        diff_rank_map = {val: i+1 for i, val in enumerate(unique_diff_vals)}
        moveback_lst['priority_7'] = moveback_lst['difference_tobt-sobt'].map(diff_rank_map)
        
        # Priority 8 - SFPL ID Sorting
        moveback_lst = moveback_lst.sort_values(by=['sfplid'], ascending=True)
        moveback_lst['priority_8'] = np.arange(1, len(moveback_lst) + 1)
        
        # Use optimized selection
        keep, nothing = optimized_selection(moveback_lst, int(slots_available))
        
        if keep.empty:
            continue
            
        # Create efficient sets of identifying values for faster lookups
        keep_acid_set = set(keep['acid'])
        keep_sfplid_set = set(keep['sfplid']) 
        keep_date_set = set(keep['date'])
        
        # Update flights efficiently with vectorized operations
        # Update df_dep_day
        dep_day_mask = (df_dep_day['acid'].isin(keep_acid_set) & 
                       df_dep_day['sfplid'].isin(keep_sfplid_set) & 
                       df_dep_day['date'].isin(keep_date_set))
                       
        if dep_day_mask.any():
            df_dep_day.loc[dep_day_mask, 'slot_shiftback'] += 1
            df_dep_day.loc[dep_day_mask, 'rw_cur'] = df_dep_day.loc[dep_day_mask, 'rw_slot']
        
        # Update df_run
        run_mask = (df_run['acid'].isin(keep_acid_set) & 
                   df_run['sfplid'].isin(keep_sfplid_set) & 
                   df_run['date'].isin(keep_date_set))
                   
        if run_mask.any():
            df_run.loc[run_mask, 'slot_shiftback'] += 1
            df_run.loc[run_mask, 'rw_cur'] = df_run.loc[run_mask, 'rw_slot']
        
        # Update slot time references
        df_run = correct_df_newrw(df_run)
        df_dep_day = correct_df_newrw(df_dep_day)
    
    return moveback_lst, nothing, slots, df_run, df_dep_day


def run_benchmark():
    """
    Run a benchmark to compare the performance of the original and optimized functions
    """
    # Define slots for testing
    global slots
    
    # Create a simple slots DataFrame
    slots = pd.DataFrame(index=range(144))
    slots['slot_nr'] = slots.index
    slots['slot_size'] = 6  # Default slot size
    slots['count'] = 0
    slots['count_verschil'] = 6  # Available slots
    slots['execute_push'] = 1
    slots['execute_moveback'] = 1
    
    # Add slot time information
    time_values = []
    for hour in range(24):
        for i in range(6):
            minute = i * 10
            slot_time = f"{hour:02d}:{minute:02d}"
            end_minute = minute + 10 if minute < 50 else 0
            end_hour = hour if minute < 50 else (hour + 1) % 24
            end_time = f"{end_hour:02d}:{end_minute:02d}"
            time_values.append((slot_time, end_time))
    
    slots['slot_starttime'] = [datetime.time(int(t[0].split(':')[0]), int(t[0].split(':')[1])) 
                              for t in time_values]
    slots['slot_endtime'] = [datetime.time(int(t[1].split(':')[0]), int(t[1].split(':')[1])) 
                            for t in time_values]
    
    print("Running benchmark...")
    
    # 1. Test optimized data loading
    start_time = time.time()
    try:
        _, _, df_dep, df_history = load_TOBT_data_optimized('2_ctot')
        load_time = time.time() - start_time
        print(f"Data loading completed in {load_time:.3f} seconds")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Make sure we have data to work with
    if df_dep.empty:
        print("No data loaded")
        return
        
    # 2. Test conditions_obp - process df_history for CTOT analysis
    start_time = time.time()
    ctot_info = conditions_obp_optimized(df_history)
    # Join the CTOT info back to our main dataframe
    if not ctot_info.empty:
        df_dep = pd.merge(df_dep, ctot_info[['acid', 'sfplid', 'date', 'ctot_updates', 'ctot_cancelled', 'last_ctot']], 
                         on=['acid', 'sfplid', 'date'], how='left')
    df_with_conditions = df_dep
    conditions_time = time.time() - start_time
    print(f"conditions_obp completed in {conditions_time:.3f} seconds")
    
    # 3. Test optimized selection
    selection_time = 0  # Initialize with default value
    if len(df_dep) > 10:
        test_df = df_dep.head(10).copy()
        test_df[['priority_1', 'priority_2', 'priority_3', 'priority_4', 
                'priority_5', 'priority_6', 'priority_7', 'priority_8']] = np.random.randint(1, 5, size=(10, 8))
        
        start_time = time.time()
        keep, push = optimized_selection(test_df, 5)
        selection_time = time.time() - start_time
        print(f"optimized_selection completed in {selection_time:.3f} seconds")
        print(f"  keep: {len(keep)} rows, push: {len(push)} rows")
    
    # 4. Test correct_df_newrw
    start_time = time.time()
    df_corrected = correct_df_newrw(df_with_conditions)
    correct_time = time.time() - start_time
    print(f"correct_df_newrw completed in {correct_time:.3f} seconds")
    
    # 5. Test adapt_obp_vectorized 
    df_run = df_corrected.copy()
    start_time = time.time()
    df_run, slots, df_dep_day = adapt_obp_vectorized(slots, df_run, [], 0, df_corrected)
    adapt_time = time.time() - start_time
    print(f"adapt_obp_vectorized completed in {adapt_time:.3f} seconds")
    
    # 6. Test adjust_df_cur_optimized
    start_time = time.time()
    df_run, df_dep_day = adjust_df_cur_optimized(df_run, df_corrected, 0, time.time(), time.time() - 600)
    adjust_time = time.time() - start_time
    print(f"adjust_df_cur_optimized completed in {adjust_time:.3f} seconds")
    
    # 7. Test adapt_obp_moveback_vectorized
    start_time = time.time()
    moveback_lst, nothing, slots, df_run, df_dep_day = adapt_obp_moveback_vectorized(
        slots, df_run, 0, df_dep_day)
    moveback_time = time.time() - start_time
    print(f"adapt_obp_moveback_vectorized completed in {moveback_time:.3f} seconds")
    
    # Total time
    total_time = load_time + conditions_time + selection_time + correct_time + adapt_time + adjust_time + moveback_time
    print(f"\nTotal run time: {total_time:.3f} seconds")
    
    # Print some debug info about unique flights
    print(f"\nTotal rows: {len(df_run)}")
    print(f"Unique sfplid count: {df_run['sfplid'].nunique()}")
    print(f"Unique acid count: {df_run['acid'].nunique()}")
    print(f"Unique sfplid-acid combinations: {df_run.groupby(['sfplid', 'acid']).ngroups}")
    
    # Make sure we have unique flights in the final output
    df_run_unique = df_run.drop_duplicates(['sfplid', 'acid'])
    
    # Print flight scheduling results to verify correctness
    print_flight_results(df_run_unique)
    

def print_flight_results(df):
    """
    Print a nicely formatted table of flight results showing:
    - Callsign
    - Latest TOBT time
    - Latest CTOT time
    - Runway slot time window
    - Difference between end slot time and TOBT in minutes
    
    Args:
        df: DataFrame containing flight scheduling data
    """
    # Create header with appropriate spacing
    print("\n{:<10} {:<10} {:<10} {:<15} {:<10}".format(
        "CALLSIGN", "TOBT", "CTOT", "RWY SLOT", "DIFF (min)"))
    print("-" * 55)
    
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
            diff_minutes = tobt_minutes - slot_end_minutes
        else:
            diff_minutes = "N/A"
        
        # Print row with proper alignment
        print("{:<10} {:<10} {:<10} {:<15} {:<10}".format(
            callsign, tobt_time, ctot_time, rwy_slot, diff_minutes))
    

    

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        import traceback
        print(f"Error during benchmark: {e}")
        traceback.print_exc()
