#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loading module for flight scheduling system.

This module handles loading and preprocessing of flight data from CSV files.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from flight_keys import apply_flight_key_pipeline, insert_flight_key_as_second_column

# Pandas >=3.0 defaults string columns to a pyarrow-backed "str" dtype instead of
# legacy "object". This module relies on `dtype == 'object'` checks and on plain
# Python string ops, so restore the legacy behavior for consistent parsing.
pd.set_option('future.infer_string', False)

# On Databricks, set DMAN_BASE_DIR to the DBFS mount/volume root that contains
# the `data/` folder, e.g. os.environ['DMAN_BASE_DIR'] = '/dbfs/mnt/mycontainer'
_BASE_DIR = Path(os.environ.get('DMAN_BASE_DIR', str(Path(__file__).resolve().parent)))
TAXI_MATRIX_PATH = _BASE_DIR / 'data' / 'taxi_time' / 'median_taxi_matrix.pkl'

# Runway-level fallbacks (minutes) when a stand/runway pair is missing in the matrix.
RUNWAY_FALLBACK_MINUTES: Dict[str, float] = {
    '09': 12.0,
    '18C': 14.0,
    '18L': 11.0,
    '24': 9.0,
    '27': 13.0,
    '36C': 12.0,
    '36L': 17.0,
}

STAND_COLUMN_CANDIDATES: Sequence[str] = (
    'stand',
    'gate',
    'gate_pos',
    'ramp_pos',
    'gate_pos_short',
    'parking_position',
)

RUNWAY_COLUMN_CANDIDATES: Sequence[str] = (
    'trwy',
    'rwy',
    'runway',
    'dep_runway',
)

_TAXI_LOOKUP_CACHE: Optional[Dict[tuple[str, str], float]] = None


def _select_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first matching column name from candidates (case-insensitive)."""
    if not candidates:
        return None
    lower_map = {col.lower(): col for col in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _normalize_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    token = str(value).strip().upper()
    if token in {'', 'NAN', 'NONE'}:
        return None
    return token


def _normalize_runway(value: Any) -> Optional[str]:
    token = _normalize_token(value)
    if token is None:
        return None
    if token.startswith('RWY'):
        token = token[3:].strip()
    if token.endswith('.0'):
        token = token[:-2]
    return token


def _load_taxi_lookup() -> Dict[tuple[str, str], float]:
    global _TAXI_LOOKUP_CACHE
    if _TAXI_LOOKUP_CACHE is not None:
        return _TAXI_LOOKUP_CACHE

    if not TAXI_MATRIX_PATH.exists():
        raise FileNotFoundError(f"Taxi matrix not found at {TAXI_MATRIX_PATH}")

    matrix = pd.read_pickle(TAXI_MATRIX_PATH)
    if matrix.empty:
        raise ValueError(f"Taxi matrix at {TAXI_MATRIX_PATH} is empty")

    matrix.index = matrix.index.astype(str).str.strip().str.upper()
    matrix.columns = [str(c).strip().upper() for c in matrix.columns]

    lookup: Dict[tuple[str, str], float] = {}
    for stand, row in matrix.iterrows():
        for runway, value in row.items():
            if pd.notna(value):
                lookup[(stand, runway)] = float(value)

    if not lookup:
        raise ValueError(f"Taxi matrix at {TAXI_MATRIX_PATH} contains no usable values")

    _TAXI_LOOKUP_CACHE = lookup
    return lookup


def _resolve_taxi_time_minutes(df: pd.DataFrame) -> pd.Series:
    """Resolve taxi time per row based on stand/runway; fallback to runway defaults."""
    runway_col = _select_column(df, RUNWAY_COLUMN_CANDIDATES)
    if runway_col is None:
        raise ValueError("Taxi time resolution requires a runway column (trwy/rwy/runway)")

    stand_col = _select_column(df, STAND_COLUMN_CANDIDATES)
    stands = df[stand_col] if stand_col else pd.Series([None] * len(df), index=df.index)
    runways = df[runway_col]

    lookup = _load_taxi_lookup()
    runway_fallback = {k.upper(): float(v) for k, v in RUNWAY_FALLBACK_MINUTES.items()}


    unavailable_taxitime_count = 0
    resolved: list[float] = []
    for idx in df.index:
        runway_token = _normalize_runway(runways.at[idx])
        if runway_token is None:
            raise ValueError(f"Missing runway for taxi lookup at row {idx}")

        stand_value = stands.at[idx] if stand_col else None
        stand_token = _normalize_token(stand_value)
        taxi_minutes: Optional[float] = None

        if stand_token is not None:
            taxi_minutes = lookup.get((stand_token, runway_token))
            if taxi_minutes is None and '-' in stand_token:
                taxi_minutes = lookup.get((stand_token.replace('-', ''), runway_token))

        if taxi_minutes is None:
            taxi_minutes = runway_fallback.get(runway_token)

        if taxi_minutes is None:
            taxi_minutes = 10.0
            unavailable_taxitime_count += 1
            
        # if taxi_minutes is None:
        #     raise ValueError(
        #         f"Taxi time unavailable for stand '{stand_token or 'UNKNOWN'}' and runway '{runway_token}'"
        #     )

        resolved.append(float(taxi_minutes))
    
    if unavailable_taxitime_count > 0:
        print(f"Taxi time unavailable for {unavailable_taxitime_count} rows.")

    return pd.Series(resolved, index=df.index, dtype='float64')


def load_TOBT_data_optimized(csv_name):
    """
    Optimized data loading function with smarter filtering and type handling
    
    Args:
        csv_name: Name of the CSV file (without extension) to load from custom_test_scenarios folder
        
    Returns:
        Tuple of (df, df_atd, df_dep, df_history) DataFrames
    """
    # Check if the filename ends with .csv, if not add it
    if not csv_name.lower().endswith('.csv'):
        csv_name = f"{csv_name}.csv"
    

    # Read only the columns we need directly
    df = pd.read_csv(
        csv_name,
        # low_memory=False,
        engine='pyarrow',
        # usecols=['timesec', 'sfplid', 'acid', 'adep', 'dest', 'eobt', 'actype', 'wtc',
        #          'etd', 'retd', 'atd', 'aobt', 'etot', 'tsat', 'ttot', 'asrt', 'sobt', 'tobt', 'ctot', 'trwy']
    )

    # Normalize blank/empty string cells to NaN (easier downstream handling)
    # (Object columns only; numeric coercion happens later.)
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].replace({'': np.nan, ' ': np.nan})

    # Initial filtering efficiently
    mask =  (df.trwy.notna()) & \
            (df.trwy != '0') & \
            (df.tobt.notna()) & \
            (df.tobt != '')
    df = df[mask].copy()
    
    df['trwy'] = df['trwy'].astype(str)

    # Resolve taxi time per flight before further processing
    df['taxi_time_minutes'] = _resolve_taxi_time_minutes(df)

    # Apply flight key pipeline BEFORE datetime conversion to preserve original format
    df = apply_flight_key_pipeline(df)

# TODO delete duplicate entries (of miss zelfs iets aggresiever want kunnen miss milisec zijn etc.)


    # Convert date columns efficiently with vectorized operations
    # 'timesec' is included since its values are timestamps despite the name.
    date_columns = ['timesec', 'eobt', 'atd','etd','retd','etot','tsat','ttot','asrt','aobt', 'sobt', 'tobt', 'ctot']
    for col in date_columns:
        if col not in df.columns:
            continue
        if df[col].dtype == 'object':
            # Only process rows that have string values
            mask = df[col].notna() & (df[col] != '')
            # Force seconds resolution before the int64 cast: pandas may parse to
            # us/ms/ns resolution depending on input, and astype('int64') returns
            # the raw integer in whatever that resolution is (not always ns).
            parsed = pd.to_datetime(df.loc[mask, col], format='mixed', dayfirst=True)
            df.loc[mask, col] = parsed.astype('datetime64[s]').astype('int64')
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Some CSV variants get auto-parsed to datetime64 by the pyarrow engine;
            # normalize these the same way (epoch seconds, NaT -> NaN) for downstream code.
            nat_mask = df[col].isna()
            epoch_seconds = df[col].astype('datetime64[s]').astype('int64')
            df[col] = epoch_seconds.where(~nat_mask, np.nan)

    # Process outbound flights efficiently with boolean indexing
    df['dt'] = pd.to_datetime(df['timesec'], unit='s')
    mask = (df['adep'] == 'EHAM') & (df['dest'] != 'EHAM') & (df['dest'] != 'ZZZZ') # dest = ZZZZ used by police helicopter / coastguard
    df = df[mask].copy()

    # Identify threads with vectorized operations
    df.sort_values(['sfplid', 'timesec'], inplace=True)
    # 'timesec' is now normalized to epoch seconds above, so a plain numeric diff works
    df['dif'] = df.timesec.diff(1)
    df['thread'] = np.where(abs(df.dif) > 2, 1, 0)
    df['thread'] = df.thread.cumsum()
    df['new'] = df['thread'].diff(-1) == -1
    df.loc[df.index[-1], 'new'] = True
    
    # Extract unique flights - prioritize rows with ATD data, then latest update
    # For flights that departed (have ATD), we want the row with ATD data
    # For flights that didn't depart, we want the latest update
    
    # Create a priority column: rows with ATD get priority 1, others get priority 2
    df['has_atd'] = df['atd'].notna() & (df['atd'] != '')
    df['priority'] = df['has_atd'].map({True: 1, False: 2})
    
    # Sort by flight_key, then by priority (ATD rows first), then by latest timesec and sfplid
    df.sort_values(['flight_key', 'priority', 'timesec', 'sfplid'], 
                   ascending=[True, True, False, False], inplace=True)
    
    # Drop duplicates keeping the first entry (which will be ATD row if exists, else latest update)
    df_dep = df.drop_duplicates(subset=['flight_key'], keep='first').copy()

    # Clean up temporary columns
    df_dep.drop(['has_atd', 'priority'], axis=1, inplace=True, errors='ignore')

    # Capture statistics from deduplicated data BEFORE ATD filtering
    # Calculate flight plan updates: same callsign with different sfplids
    total_tobt_updates = len(df)  # All rows (TOBT updates)
    unique_flight_keys = df_dep['flight_key'].nunique()
    unique_sfplids = df['sfplid'].nunique()  # Unique flight plan IDs
    unique_callsigns = df_dep['acid'].nunique()
    
    # Flight plan updates = cases where same callsign gets different sfplid
    # This happens when a flight plan is refiled/updated with new route, fuel, etc.
    callsign_sfplid_combinations = df[['acid', 'sfplid']].drop_duplicates()
    callsigns_with_multiple_sfplids = callsign_sfplid_combinations.groupby('acid').size()
    flight_plan_updates_count = (callsigns_with_multiple_sfplids - 1).sum()  # Extra sfplids per callsign
    
    original_stats = {
        'total_tobt_updates': total_tobt_updates,  # All TOBT update rows 
        'unique_flight_keys': unique_flight_keys,  # Unique flights after deduplication
        'unique_sfplids': unique_sfplids,  # Unique flight plan IDs
        'unique_callsigns': unique_callsigns,  # Unique callsigns after deduplication
        'flight_plan_updates': flight_plan_updates_count,  # Actual flight plan changes (callsign refiled)
        'before_atd_filter': len(df_dep)  # Flights before ATD filtering
    }
    
    # Ensure proper column ordering for df_dep
    df_dep = insert_flight_key_as_second_column(df_dep)
    
    # Add derived columns efficiently
    df_dep.loc[:, 'airline'] = df_dep.acid.str[:3]
    df_dep.loc[:, 'atd_s'] = pd.to_datetime(df_dep['atd'], unit='s')
    df_dep.loc[:, 'eobt_s'] = pd.to_datetime(df_dep['eobt'], unit='s')
    df_dep.loc[:, 'date'] = df_dep['eobt_s'].dt.date
    df_dep.loc[:, 'tsat_s'] = pd.to_datetime(df_dep['tsat'], unit='s')
    df_dep.loc[:, 'tsat_time'] = df_dep['tsat_s'].dt.time
    df_dep.loc[:, 'tobt_s'] = pd.to_datetime(df_dep['tobt'], unit='s')
    df_dep.loc[:, 'tobt_time'] = df_dep['tobt_s'].dt.time
    
    # Import global configuration
    import global_vars
    
    # Scheduling calculations - add taxi time to TOBT to get scheduled TTOT
    if 'taxi_time_minutes' not in df_dep.columns:
        raise ValueError("Expected taxi_time_minutes column after taxi lookup")
    df_dep.loc[:, 'taxi_time_minutes'] = pd.to_numeric(df_dep['taxi_time_minutes'], errors='coerce')
    if df_dep['taxi_time_minutes'].isna().any():
        missing_flights = df_dep.loc[df_dep['taxi_time_minutes'].isna(), 'flight_key'].unique()[:5]
        raise ValueError(
            f"Unresolved taxi times for flights: {', '.join(map(str, missing_flights))}"
        )
    df_dep.loc[:, 'sched_ttot'] = df_dep['tobt'] + (df_dep['taxi_time_minutes'] * 60)
    
    # Handle CTOT efficiently
    df_dep.loc[:, 'ctot'] = pd.to_numeric(df_dep['ctot'], errors='coerce')
    df_dep.loc[df_dep['ctot'] == 0.0, 'ctot'] = np.nan
    df_dep.loc[:, 'ctot_s'] = pd.to_datetime(df_dep['ctot'], unit='s', errors='coerce')
    df_dep.loc[df_dep.ctot_s.dt.year == 1970, 'ctot_s'] = np.nan
    
    # Apply CTOT constraints using vectorized operations
    valid_ctot_mask = df_dep['ctot'].notna()
    
    # Use configurable CTOT margins from global vars
    ctot_min_margin = global_vars.CTOT_MIN_MARGIN
    ctot_max_margin = global_vars.CTOT_MAX_MARGIN
    
    # For flights with CTOT, the scheduled takeoff time must respect BOTH:
    # 1. TOBT + taxi time (physical constraint)
    # 2. CTOT window (regulatory constraint)
    
    # Calculate CTOT window bounds
    ctot_earliest = df_dep['ctot'] - ctot_min_margin*60  # CTOT - 5 minutes
    ctot_latest = df_dep['ctot'] + ctot_max_margin*60    # CTOT + 10 minutes
    
    # For CTOT flights, use the LATER of: (TOBT+taxi) or (CTOT-5min)
    ctot_constrained_time = pd.DataFrame({
        'tobt_plus_taxi': df_dep['sched_ttot'],  # Already calculated as TOBT + taxi
        'ctot_earliest': ctot_earliest
    }).max(axis=1)
    
    # Apply CTOT constraints
    df_dep.loc[valid_ctot_mask, 'sched_ttot'] = ctot_constrained_time[valid_ctot_mask]
    
    # Cap at CTOT + max_margin if needed
    over_limit_mask = valid_ctot_mask & (df_dep['sched_ttot'] > ctot_latest)
    df_dep.loc[over_limit_mask, 'sched_ttot'] = ctot_latest[over_limit_mask]
    
    # Additional derived fields
    df_dep.loc[:, 'sched_ttot_s'] = pd.to_datetime(df_dep['sched_ttot'], unit='s')
    # df_dep.loc[:, 'sched_ttot_time'] = pd.to_datetime(df_dep['tobt']+15*60, unit='s').dt.time # vgm doet deze line niets, en klopt niet want hier worden 15 min opeens toegevoegd
    df_dep.loc[:, 'ctot_time'] = df_dep['ctot_s'].dt.time
    df_dep.loc[:, 'sobt_s'] = pd.to_datetime(df_dep['sobt'], unit='s')
    df_dep.loc[:, 'sobt_time'] = df_dep['sobt_s'].dt.time
    df_dep['difference_tobt-sobt'] = pd.to_datetime(df_dep.tobt - df_dep.sobt, unit='s')
    df_dep.loc[:, 'taxi'] = round((pd.to_datetime(df_dep['ttot']) - pd.to_datetime(df_dep['tsat'])).dt.total_seconds()/60)

    # Calculate runway slots using vectorized operations
    # Import globals for slot configuration
    import global_vars
    
    # Second-precise slot indexing (avoid minute-based rounding bias)
    secs_midnight = (df_dep['sched_ttot_s'].dt.hour * 3600 +
                     df_dep['sched_ttot_s'].dt.minute * 60 +
                     df_dep['sched_ttot_s'].dt.second)
    df_dep.loc[:, 'rw_slot'] = (secs_midnight // global_vars.SLOT_DURATION_SECONDS).astype(int)
    
    # Filter and initialize
    df_dep = df_dep[df_dep['tobt'] != 0.0]
    df_dep.loc[:, 'slot_shift'] = 0
    df_dep.loc[:, 'slot_shiftback'] = 0
    df_dep.loc[:, 'ctot_cancelled'] = np.nan
    df_dep.loc[:, 'last_ctot'] = np.nan
    df_dep.loc[:, 'prev_tobt_expired'] = np.nan
    df_dep.loc[:, 'shift_day'] = np.nan
    
    # DISABLE ATD filtering for now since notebook analysis shows 100% of callsigns have ATD
    # The aggressive filtering was removing valid flights incorrectly
    
    # Create df_atd for compatibility but include all flights
    df_atd = df_dep[['flight_key', 'acid', 'sfplid', 'date', 'atd']].copy()
    df_atd = df_atd.sort_values('acid')
    
    # NO FILTERING - keep all flights
    # df_dep = df_dep  # Keep all flights, no ATD-based filtering
                      
    # Capture the complete update history for each flight for CTOT analysis
    # This DataFrame contains all updates for each flight
    df_history = df.copy()

    # # Multiple updates in the same minute should be consolidated
    # df_history['update_minute'] = pd.to_datetime(df_history['timesec']).dt.floor('min')
    # df_history = df_history.sort_values(['flight_key', 'timesec'])
    # df_history = df_history.drop_duplicates(subset=['flight_key', 'update_minute'], keep='last')
    # df_history = df_history.drop('update_minute', axis=1)

    # Add date column to history data for CTOT analysis
    if 'date' not in df_history.columns:
        df_history['eobt_s'] = pd.to_datetime(df_history['eobt'], unit='s')
        df_history['date'] = df_history['eobt_s'].dt.date
    # Ensure flight_key column ordering for history data as well
    df_history = insert_flight_key_as_second_column(df_history)
    # We'll pass this to conditions_obp_optimized for proper CTOT update tracking
    
    # Add CTOT window bound columns in seconds since epoch for enforcement later
    # Use -inf / +inf for flights without CTOT
    with_ctot = df_dep['ctot'].notna()
    ctot_min_margin = global_vars.CTOT_MIN_MARGIN
    ctot_max_margin = global_vars.CTOT_MAX_MARGIN
    # Epoch seconds already in 'ctot'; create earliest/latest seconds (floats)
    df_dep['earliest_ok'] = np.where(with_ctot, df_dep['ctot'] - ctot_min_margin * 60, -np.inf)
    df_dep['latest_ok'] = np.where(with_ctot, df_dep['ctot'] + ctot_max_margin * 60, np.inf)

    # Day-relative (seconds since local midnight) CTOT window for intra-day slot comparisons
    # Compute midnight epoch for sched_ttot (already epoch seconds)
    midnight_epoch = df_dep['sched_ttot'] - (df_dep['sched_ttot'] % 86400)
    # Convert earliest/latest epoch to relative seconds within same day; if crosses day, allow full range by normalizing modulo 86400
    df_dep['earliest_ok_sec'] = np.where(with_ctot,
        (df_dep['earliest_ok'] - midnight_epoch) % 86400,
        -np.inf)
    df_dep['latest_ok_sec'] = np.where(with_ctot,
        (df_dep['latest_ok'] - midnight_epoch) % 86400,
        np.inf)

    return df, df_atd, df_dep, df_history, original_stats
