#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flight scheduling module for optimizing runway slot allocation.

This module handles the core scheduling logic for allocating flights to runway slots
based on various constraints and priorities.
"""

import pandas as pd
import numpy as np
from flight_prioritization import optimized_selection
from slot_manager import correct_df_newrw
import global_vars


def adapt_obp_vectorized(slots, df_run, slotlist, loop_slot, df_dep_day, verbose=0):
    """
    Improved vectorized version that processes all slots sequentially without endless loops.
    
    Algorithm:
    1. All flights are initially assigned based on TOBT + taxi time + CTOT constraints
    2. Process each slot sequentially from start to end
    3. For each overfilled slot, use prioritization to determine which flights to push
    4. Push excess flights to the very next slot
    5. Continue processing subsequent slots (which may now be overfilled)
    6. Single pass through all slots ensures no infinite loops
    
    Args:
        slots: DataFrame with slot information
        df_run: DataFrame with flight data (flights should already be assigned to initial slots)
        slotlist: List of slots
        loop_slot: Starting slot to process from
        df_dep_day: DataFrame with departure day information
        verbose: Verbosity level for additional output (default: 0)
        
    Returns:
        Tuple of (df_run, slots, df_dep_day)
    """
    # Ensure slot_nr is properly defined
    if 'slot_nr' not in slots.columns:
        slots['slot_nr'] = slots.index
    
    # Create a mask for slots to process
    process_slots = slots['slot_nr'] >= loop_slot
    
    if not process_slots.any():
        return df_run, slots, df_dep_day
    
    # Get slots that need to be processed, sorted
    slot_indices = sorted(slots[process_slots].index)
    
    # Create lookup dictionaries for slot info
    slot_sizes = slots['slot_size'].to_dict()
    
    if verbose > 0:
        print(f"DEBUG: Starting sequential slot processing from slot {loop_slot}")
        print(f"DEBUG: Initial slot distribution: {df_run['rw_cur'].value_counts().to_dict()}")

    has_blocks = 'is_mri_block' in df_run.columns
    if has_blocks:
        block_mask_series = pd.Series(
            df_run['is_mri_block'].to_numpy(dtype=bool, na_value=False),
            index=df_run.index,
        )
    else:
        block_mask_series = None
    
    # Precompute flight_key -> index mapping (future extensibility)
    _fk_index_map = pd.Series(df_run.index.values, index=df_run['flight_key'])  # noqa: F841

    # Precompute all priority columns once (vectorized) to avoid per-slot ranking overhead
    n = len(df_run)
    if n:
        # Ensure needed columns exist
        for i in range(1, 9):
            col = f'priority_{i}'
            if col not in df_run.columns:
                df_run[col] = np.nan
        # Priority 1: CTOT updates high -> keep earlier: invert logic so lower is better
        if 'ctot_updates' in df_run.columns:
            df_run['priority_1'] = np.where(df_run['ctot_updates'] >= 10, 0, 1)
        # Priority 2: slot_shift >2 prefers staying
        if 'slot_shift' in df_run.columns:
            df_run['priority_2'] = np.where(df_run['slot_shift'] > 2, 0, 1)
        # Priority 3: slack (latest_ok_sec - rw_cur_end_sec) lower slack -> smaller value preferred
        if {'latest_ok_sec','rw_cur_end_sec'}.issubset(df_run.columns):
            df_run['priority_3'] = (df_run['latest_ok_sec'] - df_run['rw_cur_end_sec'])
        # Priority 4: cancellations
        if {'ctot_cancelled','last_ctot','rw_cur_endtime'}.issubset(df_run.columns):
            try:
                last_ctot_time_all = pd.to_datetime(df_run['last_ctot'], unit='s').dt.time
                mask4_all = (df_run['ctot_cancelled'] > 0) & (last_ctot_time_all > df_run['rw_cur_endtime'])
                df_run['priority_4'] = np.where(mask4_all, 0, 1)
            except Exception:
                pass
        # Priority 5: rw_slot rank
        if 'rw_slot' in df_run.columns:
            df_run['priority_5'] = df_run['rw_slot'].rank(method='dense')
        # Priority 6: TOBT rank
        if 'tobt_s' in df_run.columns:
            df_run['priority_6'] = df_run['tobt_s'].rank(method='dense')
        # Priority 7: difference_tobt-sobt descending rank
        if 'difference_tobt-sobt' in df_run.columns:
            df_run['priority_7'] = df_run['difference_tobt-sobt'].rank(method='dense', ascending=False)
        # Priority 8: sfplid rank
        if 'sfplid' in df_run.columns:
            df_run['priority_8'] = df_run['sfplid'].rank(method='dense')

    # Prepare numpy arrays for composite selection: we build a weighted key to avoid lexsort overhead
    # Scale each priority to an integer rank sequence (already ranks) then combine.
    rank_cols = [f'priority_{i}' for i in range(1,9) if f'priority_{i}' in df_run.columns]
    # Create normalized ranks to small ints
    composite = None
    if rank_cols:
        comp = np.zeros(len(df_run), dtype='float64')
        for col in rank_cols:
            vals = df_run[col].to_numpy(dtype='float64', copy=False)
            # Replace NaN with large number so they deprioritize
            np_nan_mask = np.isnan(vals)
            if np_nan_mask.any():
                vals = vals.copy()
                vals[np_nan_mask] = np.nanmax(vals[~np_nan_mask]) + 1000 if (~np_nan_mask).any() else 1e9
            comp = comp * 1e6 + np.clip(vals, -1e12, 1e12)  # positional weighting
        composite = comp

    # Initialize slot counts array for incremental maintenance
    slot_counts = np.zeros(len(slots), dtype=int)
    if 'rw_cur' in df_run.columns:
        cur_vals = df_run['rw_cur'].dropna().astype(int).to_numpy()
        valid_mask = (cur_vals >= 0) & (cur_vals < len(slot_counts))
        binc = np.bincount(cur_vals[valid_mask], minlength=len(slot_counts))
        slot_counts[:len(binc)] = binc
        slots['count'] = slot_counts
        slots['count_verschil'] = slots['slot_size'] - slots['count']

    for slot in slot_indices:
        if slots.at[slot, 'execute_push'] != 1:
            continue
        slot_size = slot_sizes.get(slot, 0)
        flight_count = slot_counts[slot]
        slots.at[slot, 'count'] = flight_count
        slots.at[slot, 'count_verschil'] = slot_size - flight_count
        if flight_count <= slot_size:
            if verbose > 0:
                print(f"DEBUG: Slot {slot} within capacity ({flight_count}/{slot_size})")
            continue
        if verbose > 0:
            print(f"DEBUG: Slot {slot} overfilled by {flight_count - slot_size}")
        slot_flights_mask = (df_run['rw_cur'] == slot)
        if has_blocks:
            slot_blocks_mask = slot_flights_mask & block_mask_series
            block_count = int(slot_blocks_mask.sum())
            movable_mask_series = slot_flights_mask & (~block_mask_series)
        else:
            block_count = 0
            movable_mask_series = slot_flights_mask
        real_capacity = max(0, slot_size - block_count)
        movable_idx_array = np.nonzero(movable_mask_series.to_numpy())[0]
        if len(movable_idx_array) <= real_capacity:
            continue
        # Composite ordering using partial selection (argpartition) to avoid full sort when overfilled
        if composite is not None:
            if real_capacity <= 0:
                push = df_run.iloc[movable_idx_array]
            else:
                comp_vals = composite[movable_idx_array]
                part = np.argpartition(comp_vals, real_capacity - 1)
                push_local_idx = movable_idx_array[part[real_capacity:]]
                push = df_run.iloc[push_local_idx]
        else:
            _, push = optimized_selection(df_run.loc[movable_mask_series], real_capacity)
        if push.empty:
            continue
        next_slot = slot + 1
        if next_slot >= len(slots):
            if hasattr(global_vars, 'TOTAL_SLOTS') and next_slot >= global_vars.TOTAL_SLOTS:
                next_slot = 0
                day_shift = True
            else:
                next_slot = len(slots) - 1
                day_shift = False
        else:
            day_shift = False
        push_keys = push['flight_key'].to_numpy()
        push_mask_run = df_run['flight_key'].isin(push_keys)
        push_mask_dep = df_dep_day['flight_key'].isin(push_keys) if 'flight_key' in df_dep_day.columns else pd.Series([False]*len(df_dep_day))
        if 'earliest_ok_sec' in df_run.columns and 'slot_end_sec' in slots.columns and next_slot in slots.index:
            cand_end = slots.at[next_slot, 'slot_end_sec']
            # Vector mask of allowed pushes
            allowed_push = df_run.loc[push_mask_run, 'earliest_ok_sec'] <= cand_end
            allowed_idx = allowed_push[allowed_push].index
            allowed_mask = df_run.index.isin(allowed_idx)
            push_mask_run &= allowed_mask
            if push_mask_dep.any():
                allowed_keys = set(df_run.loc[allowed_mask, 'flight_key'])
                push_mask_dep &= df_dep_day['flight_key'].isin(allowed_keys)
        moved_indices = df_run.index[push_mask_run]
        if len(moved_indices):
            # Update shifts
            df_run.loc[moved_indices, 'slot_shift'] = df_run.loc[moved_indices, 'slot_shift'].fillna(0) + 1
            # Decrement old slot count(s)
            old_slots = df_run.loc[moved_indices, 'rw_cur'].astype(int)
            for os in old_slots:
                if 0 <= os < len(slot_counts):
                    slot_counts[os] -= 1
            df_run.loc[moved_indices, 'rw_cur'] = next_slot
            # Increment new slot count
            if 0 <= next_slot < len(slot_counts):
                slot_counts[next_slot] += len(moved_indices)
            if day_shift:
                df_run.loc[moved_indices, 'shift_day'] = 1
        if push_mask_dep.any():
            dep_idx = df_dep_day.index[push_mask_dep]
            df_dep_day.loc[dep_idx, 'slot_shift'] = df_dep_day.loc[dep_idx, 'slot_shift'].fillna(0) + 1
            df_dep_day.loc[dep_idx, 'rw_cur'] = next_slot
            if day_shift:
                df_dep_day.loc[dep_idx, 'shift_day'] = 1
        # Update counts for both involved slots (slot and next_slot)
        slots.at[slot, 'count'] = slot_counts[slot]
        slots.at[slot, 'count_verschil'] = slot_size - slot_counts[slot]
        if 0 <= next_slot < len(slot_counts):
            slots.at[next_slot, 'count'] = slot_counts[next_slot]
            slots.at[next_slot, 'count_verschil'] = slots.at[next_slot, 'slot_size'] - slot_counts[next_slot]
        if verbose > 0:
            print(f"DEBUG: Slot {slot} now count={slot_counts[slot]} after pushing {len(push)} -> slot {next_slot} (next count={slot_counts[next_slot] if next_slot < len(slot_counts) else 'NA'})")
    
    # Final update of all slot counts using numpy bincount (faster than per-slot boolean sums)
    try:
        rw_cur_int = df_run['rw_cur'].dropna().astype(int).values
        max_slot = int(slots.index.max()) if len(slots) else -1
        if max_slot >= 0:
            counts_arr = np.bincount(rw_cur_int, minlength=max_slot + 1)
            slots['count'] = slots.index.map(lambda s: counts_arr[s] if s < len(counts_arr) else 0)
            slots['count_verschil'] = slots['slot_size'] - slots['count']
    except Exception:
        # Fallback (should not normally happen)
        vc = df_run['rw_cur'].value_counts()
        slots['count'] = slots.index.map(lambda s: vc.get(s, 0))
        slots['count_verschil'] = slots['slot_size'] - slots['count']
    
    if verbose > 0:
        print(f"DEBUG: Final slot distribution: {df_run['rw_cur'].value_counts().to_dict()}")

    # Ensure df_dep_day has critical scheduling time columns (some callers expect them)
    essential_cols = [c for c in ['sched_ttot_s','rw_cur_start_sec','rw_cur_end_sec','rw_cur_starttime','rw_cur_endtime'] if c in df_run.columns]
    missing = [c for c in essential_cols if c not in df_dep_day.columns]
    if missing and 'flight_key' in df_dep_day.columns:
        merge_cols = ['flight_key'] + essential_cols
        enrich = df_run[merge_cols].drop_duplicates('flight_key')
        df_dep_day = df_dep_day.merge(enrich, on='flight_key', how='left', suffixes=('','_y'))
        # In case of collisions keep left side original, only fill NaNs
        for c in essential_cols:
            if c + '_y' in df_dep_day.columns:
                df_dep_day[c] = df_dep_day[c].fillna(df_dep_day[c + '_y'])
                df_dep_day.drop(columns=[c + '_y'], inplace=True)
    
    return df_run, slots, df_dep_day


def adapt_obp_moveback_vectorized(slots, df_run, loop_slot, df_dep_day, verbose=0):
    """
    Vectorized version of adapt_obp_moveback that minimizes iteration and unnecessary operations.
    This function handles the moveback phase of the flight scheduling process, where flights
    that were previously moved from their original slots may be moved back if conditions permit.
    
    The function strictly respects slot capacity constraints during the moveback phase:
    - Flights are only moved back to their original slots if there is sufficient capacity
    - When capacity constraints prevent moving all flights, flights are selected based on priority
    - The highest priority flights (based on multiple criteria) are selected when limited spaces are available
    
    Args:
        slots: DataFrame with slot information including slot capacity ('slot_size')
        df_run: DataFrame with flight data including current and original slot assignments
        loop_slot: Current slot being processed
        df_dep_day: DataFrame with departure day information
        verbose: Verbosity level for additional output (default: 0)
        
    Returns:
        Tuple of (moveback_lst, nothing, slots, df_run, df_dep_day) where:
            moveback_lst: DataFrame containing flights selected for moveback
            nothing: DataFrame containing flights not selected for moveback
            slots: Updated slots DataFrame
            df_run: Updated flight data
            df_dep_day: Updated departure day information
    """
    # Pre-allocate empty DataFrames
    moveback_lst = pd.DataFrame()
    nothing = pd.DataFrame()
    
    # Get relevant slots efficiently
    relevant_slots = slots[(slots['slot_nr'] > loop_slot) & (slots['execute_moveback'] == 1)]
    
    if relevant_slots.empty:
        return moveback_lst, nothing, slots, df_run, df_dep_day
    
    has_blocks = 'is_mri_block' in df_run.columns
    if has_blocks:
        block_mask_series = pd.Series(
            df_run['is_mri_block'].to_numpy(dtype=bool, na_value=False),
            index=df_run.index,
        )
    else:
        block_mask_series = None

    # Initialize slot_counts from existing slot counts or compute once
    if 'count' in slots.columns:
        slot_counts = slots['count'].to_numpy(copy=True)
    else:
        max_slot_idx = int(slots.index.max()) if len(slots) else -1
        slot_counts = np.zeros(max_slot_idx+1, dtype=int)
        vc_init = df_run['rw_cur'].dropna().astype(int).value_counts()
        for s, c in vc_init.items():
            if 0 <= s < len(slot_counts):
                slot_counts[s] = c

    # Pre-build slot start/end second maps (used for earliest_ok checks and composite evaluation)
    slot_start_sec_map = slots['slot_start_sec'].to_dict() if 'slot_start_sec' in slots.columns else {}
    slot_end_sec_map = slots['slot_end_sec'].to_dict() if 'slot_end_sec' in slots.columns else {}

    # Process each relevant slot (these are slots whose execute_moveback == 1)
    for slot, slot_row in relevant_slots.iterrows():
        slot_size_local = slot_row['slot_size']
        
        # Create efficient masks for filtering
        condition1 = ((df_run['rw_slot'] <= slot) & 
                     (df_run['rw_slot'] < df_run['rw_cur']) & 
                     (df_run['rw_cur'] > slot))
                     
        condition2 = ((df_run['rw_slot'] > slot) & 
                     (df_run['rw_slot'] > df_run['rw_cur']) & 
                     (df_run['rw_cur'] < slot) & 
                     (df_run['shift_day'] == 1))

        if has_blocks:
            condition1 &= (~block_mask_series)
            condition2 &= (~block_mask_series)
        
        # Check if we have flights to process
        if not condition1.any() and not condition2.any():
            continue
            
        # Update slot count
        curr_slot_count = (df_run['rw_cur'] == slot).sum()
        slots.loc[slot, 'count'] = curr_slot_count
        
        # Count how many flights want to move back to their original slots
        flights_to_original_slots_count = 0
        for cond in [condition1, condition2]:
            if cond.any():
                flights_to_original_slots = df_run[cond]
                original_slot_counts = flights_to_original_slots['rw_slot'].value_counts()
                for orig_slot, count in original_slot_counts.items():
                    if orig_slot == slot:
                        flights_to_original_slots_count += count

        if verbose > 0:
            print(f"DEBUG: Slot {slot} has {curr_slot_count} flights already and {flights_to_original_slots_count} flights want to move back")
        
        # Only proceed if we have room in the slot
        if curr_slot_count >= slot_size_local:
            if verbose > 0:
                print(f"DEBUG: Slot {slot} is already at capacity ({curr_slot_count}/{slot_size_local}), can't move flights back")
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
        
        # Build priority columns (lower is better). Avoid repeated sorting; use rank operations.
        moveback_lst = moveback_lst.copy()
        # Initialize priorities with neutral high values (so unset remain low influence after binary conditions)
        for i in range(1,9):
            col = f'priority_{i}'
            if col not in moveback_lst.columns:
                moveback_lst[col] = np.nan
        # Binary priorities (1 if condition, else 0)
        if 'ctot_updates' in moveback_lst.columns:
            moveback_lst['priority_1'] = (moveback_lst['ctot_updates'] >= 10).astype(int)
        if 'slot_shift' in moveback_lst.columns:
            moveback_lst['priority_2'] = (moveback_lst['slot_shift'] > 2).astype(int)
        if {'ctot_s','rw_cur_endtime'}.issubset(moveback_lst.columns):
            try:
                moveback_lst['priority_3'] = ((moveback_lst['ctot_s'] + pd.Timedelta(minutes=5)).dt.time < moveback_lst['rw_cur_endtime']).astype(int)
            except Exception:
                pass
        if {'ctot_cancelled','last_ctot','rw_cur_endtime'}.issubset(moveback_lst.columns):
            try:
                ctot_cancel_mask = ((moveback_lst['ctot_cancelled'] > 0) & (pd.to_datetime(moveback_lst['last_ctot'], unit='s').dt.time > moveback_lst['rw_cur_endtime']))
                moveback_lst['priority_4'] = ctot_cancel_mask.astype(int)
            except Exception:
                pass
        if 'slot_shift' in moveback_lst.columns:
            # Invert: more shifted gets lower value to encourage moveback
            p5 = np.full(len(moveback_lst), 3, dtype=int)
            p5[moveback_lst['slot_shift'] >= 1] = 2
            p5[moveback_lst['slot_shift'] > 1] = 1
            moveback_lst['priority_5'] = p5
        # Rank-based priorities
        if 'tobt_s' in moveback_lst.columns:
            moveback_lst['priority_6'] = moveback_lst['tobt_s'].rank(method='dense').astype(int)
        if 'difference_tobt-sobt' in moveback_lst.columns:
            moveback_lst['priority_7'] = (-moveback_lst['difference_tobt-sobt']).rank(method='dense').astype(int)
        if 'sfplid' in moveback_lst.columns:
            moveback_lst['priority_8'] = moveback_lst['sfplid'].rank(method='dense').astype(int)

        # Vector earliest_ok window feasibility for all candidates (original slot timing vs window)
        if {'earliest_ok_sec','latest_ok_sec','rw_slot'}.issubset(moveback_lst.columns) and slot_start_sec_map:
            orig_slot_int = moveback_lst['rw_slot'].astype(int)
            start_secs = orig_slot_int.map(slot_start_sec_map)
            end_secs = orig_slot_int.map(slot_end_sec_map) if slot_end_sec_map else start_secs + global_vars.SLOT_DURATION_SECONDS
            earliest = moveback_lst['earliest_ok_sec'] if 'earliest_ok_sec' in moveback_lst.columns else -np.inf
            latest = moveback_lst['latest_ok_sec'] if 'latest_ok_sec' in moveback_lst.columns else np.inf
            feasible_mask = (end_secs >= earliest) & (start_secs <= latest)
            moveback_lst = moveback_lst[feasible_mask]
            if moveback_lst.empty:
                continue

        # Group by destination slot (rw_slot) and choose flights respecting capacity per slot
        destination_slots = moveback_lst['rw_slot'].unique()
        valid_moves = []
        for dest_slot in destination_slots:
            flights_to_this_slot = moveback_lst[moveback_lst['rw_slot'] == dest_slot]
            move_count = len(flights_to_this_slot)
            # Current count from cached slot_counts (fallback to on-the-fly if out of range)
            if 0 <= dest_slot < len(slot_counts):
                current_count = slot_counts[dest_slot]
            else:
                current_count = (df_run['rw_cur'] == dest_slot).sum()
            dest_slot_size = slots.at[dest_slot, 'slot_size'] if dest_slot in slots.index else 0
            
            if verbose > 0:
                print(f"DEBUG: For destination slot {dest_slot}: {move_count} flights want to move, {current_count} already there, capacity is {dest_slot_size}")
            
            # CAPACITY DECISION LOGIC
            # Three possibilities:
            # 1. All flights can move (current + new <= capacity)
            # 2. Some flights can move (current < capacity)
            # 3. No flights can move (current >= capacity)
            if current_count + move_count <= dest_slot_size:
                # Safe to move all flights to this slot
                valid_moves.extend(flights_to_this_slot.index.tolist())
                if verbose > 0:
                    print(f"DEBUG: All {move_count} flights can move to slot {dest_slot}")
            elif dest_slot_size > current_count:
                # PARTIAL CAPACITY SCENARIO
                # We can only move some flights - need to select the highest priority ones
                available_spaces = dest_slot_size - current_count
                if available_spaces > 0:
                    # Build composite priority key (lower is better)
                    pr_cols = [f'priority_{i}' for i in range(1,9) if f'priority_{i}' in flights_to_this_slot.columns]
                    if pr_cols:
                        comp = np.zeros(len(flights_to_this_slot), dtype='float64')
                        for col in pr_cols:
                            vals = flights_to_this_slot[col].to_numpy(dtype='float64', copy=False)
                            nan_mask = np.isnan(vals)
                            if nan_mask.any():
                                vals = vals.copy()
                                vals[nan_mask] = np.nanmax(vals[~nan_mask]) + 1000 if (~nan_mask).any() else 1e9
                            comp = comp * 1e3 + np.clip(vals, -1e9, 1e9)
                        if available_spaces < len(comp):
                            part = np.argpartition(comp, available_spaces - 1)
                            chosen_pos = part[:available_spaces]
                        else:
                            chosen_pos = np.arange(len(comp))
                        chosen_indices = flights_to_this_slot.iloc[chosen_pos].index.tolist()
                        valid_moves.extend(chosen_indices)
                    else:
                        # Fallback: take first N
                        valid_moves.extend(flights_to_this_slot.index[:available_spaces].tolist())
                    if verbose > 0:
                        print(f"DEBUG: Only {available_spaces} out of {move_count} flights can move to slot {dest_slot}")
            else:
                if verbose > 0:
                    print(f"DEBUG: No flights can move to slot {dest_slot} as it's already at capacity")
        
        # FLIGHT UPDATE PHASE
        if valid_moves:
            valid_moves = list(dict.fromkeys(valid_moves))  # dedupe while preserving order
            valid_flight_keys = set(df_run.loc[valid_moves, 'flight_key'])
            # Update df_dep_day
            if 'slot_shiftback' not in df_dep_day.columns:
                df_dep_day['slot_shiftback'] = 0
            dep_mask = df_dep_day['flight_key'].isin(valid_flight_keys)
            if dep_mask.any():
                df_dep_day.loc[dep_mask, 'slot_shiftback'] = df_dep_day.loc[dep_mask, 'slot_shiftback'].fillna(0) + 1
                # Adjust slot counts: decrement current, increment destination
                orig_slots_dep = df_dep_day.loc[dep_mask, 'rw_slot'].astype(int)
                cur_slots_dep = df_dep_day.loc[dep_mask, 'rw_cur'].astype(int)
                for o, c in zip(orig_slots_dep, cur_slots_dep):
                    if 0 <= c < len(slot_counts):
                        slot_counts[c] -= 1
                    if 0 <= o < len(slot_counts):
                        slot_counts[o] += 1
                df_dep_day.loc[dep_mask, 'rw_cur'] = df_dep_day.loc[dep_mask, 'rw_slot']
            # Update df_run
            run_mask = df_run.index.isin(valid_moves)
            if 'slot_shiftback' not in df_run.columns:
                df_run['slot_shiftback'] = 0
            if run_mask.any():
                # Adjust counts via vector operations
                orig_slots = df_run.loc[run_mask, 'rw_slot'].astype(int)
                cur_slots = df_run.loc[run_mask, 'rw_cur'].astype(int)
                for o, c in zip(orig_slots, cur_slots):
                    if 0 <= c < len(slot_counts):
                        slot_counts[c] -= 1
                    if 0 <= o < len(slot_counts):
                        slot_counts[o] += 1
                df_run.loc[run_mask, 'slot_shiftback'] = df_run.loc[run_mask, 'slot_shiftback'].fillna(0) + 1
                df_run.loc[run_mask, 'rw_cur'] = df_run.loc[run_mask, 'rw_slot']
                if verbose > 0:
                    print(f"DEBUG: Moved {run_mask.sum()} flights back to their original slots")
        # Refresh slots difference for this slot (and potentially touched destination slots handled globally later)
        if 0 <= slot < len(slot_counts):
            slots.at[slot, 'count'] = slot_counts[slot]
            slots.at[slot, 'count_verschil'] = slots.at[slot, 'slot_size'] - slot_counts[slot]
        
    # Update slot time references
    df_run = correct_df_newrw(df_run, verbose)
    df_dep_day = correct_df_newrw(df_dep_day, verbose)

    # Flush back full slot counts after all movebacks
    if len(slots):
        slots['count'] = slots.index.map(lambda s: slot_counts[s] if 0 <= s < len(slot_counts) else 0)
        slots['count_verschil'] = slots['slot_size'] - slots['count']
    
    return moveback_lst, nothing, slots, df_run, df_dep_day
