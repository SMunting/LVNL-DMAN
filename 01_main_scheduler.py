# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 — Slot-Based Flight Scheduler
# MAGIC
# MAGIC This notebook is the Databricks equivalent of `main.py`.
# MAGIC
# MAGIC **Prerequisites:** Run `00_setup.py` on the cluster first.
# MAGIC
# MAGIC **Widgets (parameters):**
# MAGIC | Widget | Description | Default |
# MAGIC |--------|-------------|---------|
# MAGIC | `csv` | CSV file stem (no extension) inside `data/converted/` | `4_slot_overfill` |
# MAGIC | `slot_duration` | Slot duration in seconds | `600` |
# MAGIC | `slot_capacity` | Aircraft per slot | `6` |
# MAGIC | `ctot_min_margin` | Minutes before CTOT still allowed | `5` |
# MAGIC | `ctot_max_margin` | Minutes after CTOT still allowed | `10` |
# MAGIC | `runway` | Filter to this runway only (leave blank = all) | `` |
# MAGIC | `output` | Output CSV name (leave blank = no file saved) | `` |
# MAGIC | `show_results` | Print full results table (`true`/`false`) | `false` |
# MAGIC | `build_snapshots` | YYYY-MM-DD to build minute snapshots (leave blank = skip) | `` |
# MAGIC | `minute_stride` | Minutes between snapshots | `1` |
# MAGIC | `snapshots_out` | DBFS path for snapshot output | `/dbfs/tmp/snapshots` |
# MAGIC | `snapshots_mode` | `full` or `compact` | `compact` |
# MAGIC | `mri_day` | YYYY-MM-DD to enforce MRI runway closures (leave blank = none) | `` |
# MAGIC | `verbose` | Verbosity level (0/1/2) | `0` |

# COMMAND ----------

# ── Widget declarations ────────────────────────────────────────────────────────
dbutils.widgets.text("csv",              "4_slot_overfill",    "CSV file stem")
dbutils.widgets.text("slot_duration",   "600",                "Slot duration (seconds)")
dbutils.widgets.text("slot_capacity",   "6",                  "Aircraft per slot")
dbutils.widgets.text("ctot_min_margin", "5",                  "CTOT min margin (min)")
dbutils.widgets.text("ctot_max_margin", "10",                 "CTOT max margin (min)")
dbutils.widgets.text("runway",          "",                   "Runway filter (blank = all)")
dbutils.widgets.text("output",          "",                   "Output CSV name (blank = none)")
dbutils.widgets.dropdown("show_results","false", ["true","false"], "Print results table")
dbutils.widgets.text("build_snapshots", "",                   "Build snapshots for YYYY-MM-DD (blank = skip)")
dbutils.widgets.text("minute_stride",   "1",                  "Snapshot stride (minutes)")
dbutils.widgets.text("snapshots_out",   "/dbfs/tmp/snapshots","Snapshot output path")
dbutils.widgets.dropdown("snapshots_mode","compact",["full","compact"],"Snapshot mode")
dbutils.widgets.text("mri_day",         "",                   "MRI closure day YYYY-MM-DD (blank = none)")
dbutils.widgets.text("verbose",         "0",                  "Verbosity (0/1/2)")

# COMMAND ----------

# ── Read widget values ─────────────────────────────────────────────────────────
import sys, os
from pathlib import Path

# Ensure the databricks/ folder is on sys.path so library modules are importable
_NB_DIR = Path(os.path.abspath("")).parent if Path(os.path.abspath("")).name == "notebooks" else Path(os.path.abspath(""))
if str(_NB_DIR) not in sys.path:
    sys.path.insert(0, str(_NB_DIR))

csv_name         = dbutils.widgets.get("csv")
slot_duration    = int(dbutils.widgets.get("slot_duration"))   if dbutils.widgets.get("slot_duration")    else None
slot_capacity    = int(dbutils.widgets.get("slot_capacity"))   if dbutils.widgets.get("slot_capacity")    else None
ctot_min_margin  = int(dbutils.widgets.get("ctot_min_margin")) if dbutils.widgets.get("ctot_min_margin")  else None
ctot_max_margin  = int(dbutils.widgets.get("ctot_max_margin")) if dbutils.widgets.get("ctot_max_margin")  else None
runway           = dbutils.widgets.get("runway")   or None
output           = dbutils.widgets.get("output")   or None
show_results     = dbutils.widgets.get("show_results") == "true"
build_snapshots  = dbutils.widgets.get("build_snapshots") or None
minute_stride    = int(dbutils.widgets.get("minute_stride"))
snapshots_out    = dbutils.widgets.get("snapshots_out")
snapshots_mode   = dbutils.widgets.get("snapshots_mode")
mri_day          = dbutils.widgets.get("mri_day") or None
verbose          = int(dbutils.widgets.get("verbose"))

print(f"csv={csv_name!r}  slot_duration={slot_duration}  slot_capacity={slot_capacity}")
print(f"ctot_min={ctot_min_margin}  ctot_max={ctot_max_margin}  runway={runway!r}")
print(f"output={output!r}  show_results={show_results}  verbose={verbose}")
print(f"build_snapshots={build_snapshots!r}  mri_day={mri_day!r}")

# COMMAND ----------

# ── Imports ────────────────────────────────────────────────────────────────────
import time
import pandas as pd

from data_loader import load_TOBT_data_optimized
from ctot_analyzer import conditions_obp_optimized, check_ctot_violations, print_ctot_violations
from slot_manager import initialize_slots, correct_df_newrw, adjust_df_cur_optimized
from flight_scheduler import adapt_obp_vectorized, adapt_obp_moveback_vectorized
from utils import print_flight_results, convert_epoch_columns_to_datetime
from data.runway.runway_mri import (
    rwy_mri,
    build_mri_minute_maps,
    build_takeoff_capacity_blocks,
    merge_slot_capacity_summary,
)
import global_vars

# COMMAND ----------

# ── Helper functions (identical to main.py) ────────────────────────────────────

def check_multi_day_data(df):
    if df.empty:
        return
    if 'date' in df.columns:
        unique_dates = df['date'].dropna().unique()
        if len(unique_dates) > 1:
            print(f"\n WARNING: Data contains flights from {len(unique_dates)} different dates:")
            for date in sorted(unique_dates):
                flight_count = len(df[df['date'] == date])
                print(f"   - {date}: {flight_count} flights")
            print("   Please verify scheduled off-block times are correct for multi-day operations.")
    if 'flight_key' in df.columns and 'date' in df.columns:
        flight_key_dates = df.groupby('flight_key')['date'].nunique()
        multi_date_flight_keys = flight_key_dates[flight_key_dates > 1]
        if len(multi_date_flight_keys) > 0:
            print(f"\n WARNING: {len(multi_date_flight_keys)} flight keys appear on multiple dates:")
            for flight_key, date_count in multi_date_flight_keys.head(10).items():
                dates = df[df['flight_key'] == flight_key]['date'].unique()
                print(f"   - {flight_key}: appears on {date_count} dates ({', '.join(map(str, sorted(dates)))})")
            if len(multi_date_flight_keys) > 10:
                print(f"   ... and {len(multi_date_flight_keys) - 10} more flight keys")
            print("   This may indicate data spanning multiple days or duplicate flight keys.")


def filter_flights_by_actual_runway(df, target_runway, verbose=0):
    if df.empty or 'flight_key' not in df.columns or 'atd' not in df.columns or 'trwy' not in df.columns:
        return df
    initial_count = len(df)
    initial_flight_keys = df['flight_key'].nunique()
    flight_keys_to_keep = set()
    multi_runway_warnings = []
    target_runway_str = str(target_runway)
    has_atd_mask = df['atd'].notna() & (df['atd'] != '')
    atd_df = df[has_atd_mask][['flight_key', 'trwy']].dropna()
    if not atd_df.empty:
        runway_by_flight_key = atd_df.groupby('flight_key')['trwy'].apply(lambda x: x.dropna().unique())
        for flight_key, runways in runway_by_flight_key.items():
            if len(runways) == 0:
                continue
            elif len(runways) > 1:
                multi_runway_warnings.append((flight_key, list(runways)))
                actual_runway = str(runways[0])
                if actual_runway == target_runway_str:
                    flight_keys_to_keep.add(flight_key)
            else:
                actual_runway = str(runways[0])
                if actual_runway == target_runway_str:
                    flight_keys_to_keep.add(flight_key)
    all_flight_keys = set(df['flight_key'].unique())
    flight_keys_with_atd = set(runway_by_flight_key.index) if not atd_df.empty else set()
    flight_keys_no_atd = all_flight_keys - flight_keys_with_atd
    if multi_runway_warnings:
        print(f"\n WARNING: Found {len(multi_runway_warnings)} flight keys with multiple runways in ATD rows:")
        for flight_key, runways in multi_runway_warnings[:5]:
            print(f"   - {flight_key}: found runways {runways} (using {runways[0]})")
        if len(multi_runway_warnings) > 5:
            print(f"   ... and {len(multi_runway_warnings) - 5} more flight keys")
    filtered_df = df[df['flight_key'].isin(flight_keys_to_keep)].copy()
    final_count = len(filtered_df)
    print(f"Runway filtering for runway {target_runway}:")
    print(f"  - Flight keys analyzed: {initial_flight_keys}")
    print(f"  - Flight keys using runway {target_runway}: {len(flight_keys_to_keep)}")
    print(f"  - Flight keys without ATD (never departed): {len(flight_keys_no_atd)}")
    print(f"  - Flight keys using other runways: {initial_flight_keys - len(flight_keys_to_keep) - len(flight_keys_no_atd)}")
    print(f"  - Rows before filtering: {initial_count}")
    print(f"  - Rows after filtering: {final_count} (removed {initial_count - final_count})")
    if verbose > 0 and flight_keys_no_atd:
        print(f"  - Flight keys removed (no ATD): {sorted(list(flight_keys_no_atd))}")
    return filtered_df


def calculate_tsat(df, slots=None):
    import datetime
    df = df.copy()
    if 'taxi_time_minutes' not in df.columns:
        raise ValueError("DataFrame must include taxi_time_minutes for TSAT calculation")
    df['taxi_time_minutes'] = pd.to_numeric(df['taxi_time_minutes'], errors='coerce')
    if df['taxi_time_minutes'].isna().any():
        missing = df.loc[df['taxi_time_minutes'].isna(), 'flight_key'].unique()[:5]
        raise ValueError("Missing taxi_time_minutes for flights: " + ', '.join(map(str, missing)))
    if slots is not None:
        slot_map = {}
        for idx in slots.index:
            slot_map[float(idx)] = {
                'start': slots.at[idx, 'slot_starttime'],
                'end': slots.at[idx, 'slot_endtime']
            }
        for idx, row in df.iterrows():
            slot_num = row['rw_cur']
            if pd.notna(slot_num) and slot_num in slot_map:
                df.at[idx, 'rw_cur_starttime'] = slot_map[slot_num]['start']
                df.at[idx, 'rw_cur_endtime'] = slot_map[slot_num]['end']
    df['tsat'] = pd.NaT
    valid_slots_mask = pd.notna(df['rw_cur_starttime']) & pd.notna(df['rw_cur_endtime'])
    if valid_slots_mask.any():
        if 'sched_ttot_s' in df.columns:
            base_dates = df.loc[valid_slots_mask, 'sched_ttot_s'].dt.date
        else:
            base_dates = pd.Series([datetime.date(2024, 1, 1)] * valid_slots_mask.sum(),
                                   index=df.loc[valid_slots_mask].index)
        start_times = pd.to_datetime(base_dates.astype(str) + ' ' +
                                     df.loc[valid_slots_mask, 'rw_cur_starttime'].astype(str))
        scheduled_takeoff_time = start_times
        taxi_durations = pd.to_timedelta(df.loc[valid_slots_mask, 'taxi_time_minutes'], unit='m')
        tsat_times = scheduled_takeoff_time - taxi_durations
        df.loc[valid_slots_mask, 'tsat'] = tsat_times
    return df


def run_flight_scheduling(csv_name, slot_duration_seconds=None, aircraft_per_slot=None,
                          ctot_min_margin=None, ctot_max_margin=None,
                          output=None, runway=None, verbose=0, closure_context=None):
    print(f"Running flight scheduling for {csv_name}...")
    duration_display = f"{slot_duration_seconds} seconds" if slot_duration_seconds else "600 seconds (10 minutes)"
    capacity_display = aircraft_per_slot if aircraft_per_slot else "6"
    print(f"Slot configuration: Duration = {duration_display}, Capacity = {capacity_display} aircraft per slot")
    print("Taxi time: dynamic per-flight matrix (stand/runway)")
    timing = {}
    slots = initialize_slots(
        slot_duration_seconds=slot_duration_seconds,
        aircraft_per_slot=aircraft_per_slot,
        ctot_min_margin=ctot_min_margin,
        ctot_max_margin=ctot_max_margin
    )
    start_time = time.time()
    try:
        _, _, df_dep, df_history, original_stats = load_TOBT_data_optimized(csv_name)
        timing['load_time'] = time.time() - start_time
        print(f"Data loading completed in {timing['load_time']:.3f} seconds")
        check_multi_day_data(df_dep)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None
    if df_dep.empty:
        print("No data loaded")
        return timing, None, None, None
    start_time = time.time()
    df_history_with_ctot = conditions_obp_optimized(df_history)
    if not df_history_with_ctot.empty and 'ctot_updates' in df_history_with_ctot.columns:
        ctot_cols = ['flight_key', 'ctot_updates', 'ctot_cancelled', 'last_ctot']
        available_cols = [col for col in ctot_cols if col in df_history_with_ctot.columns]
        if len(available_cols) >= 2:
            ctot_info = df_history_with_ctot[available_cols].drop_duplicates(['flight_key'])
            df_dep = pd.merge(df_dep, ctot_info, on=['flight_key'], how='left')
        else:
            print(f"Warning: Missing CTOT columns. Available: {available_cols}")
    else:
        print("Warning: No CTOT analysis results or empty data")
    df_with_conditions = df_dep
    timing['conditions_time'] = time.time() - start_time
    print(f"CTOT analysis completed in {timing['conditions_time']:.3f} seconds")
    if runway:
        df_with_conditions = filter_flights_by_actual_runway(df_with_conditions, runway, verbose)
        if df_with_conditions.empty:
            print(f"No flights found for runway {runway} after filtering")
            return timing, None, None, original_stats
    if closure_context:
        active_runways = []
        if 'trwy' in df_with_conditions.columns:
            active_runways = sorted({str(r).strip() for r in df_with_conditions['trwy'].dropna().unique() if str(r).strip()})
        requested_runways = closure_context.get('requested_runways') or []
        requested_runways = [str(r).strip() for r in requested_runways if r]
        map_runways = {str(r).strip() for r in closure_context.get('runways_from_map', []) if r}
        candidate_runways = requested_runways or active_runways
        if not candidate_runways:
            candidate_runways = list(map_runways)
        candidate_runways = [r for r in candidate_runways if r]
        if map_runways and candidate_runways:
            missing = set(candidate_runways) - map_runways
            if missing:
                raise ValueError(f"MRI availability missing takeoff map entries for runways: {sorted(missing)}")
        if not candidate_runways:
            raise ValueError("Unable to determine runways for MRI closure enforcement")
        closure_context['active_runways'] = sorted(candidate_runways)
        blocks_df, summary_df = build_takeoff_capacity_blocks(
            closure_context['maps'],
            closure_context['minutes'],
            slots,
            day=closure_context['day'],
            slot_duration_seconds=global_vars.SLOT_DURATION_SECONDS,
            runway_filter=closure_context['active_runways'],
        )
        if not summary_df.empty:
            slots = merge_slot_capacity_summary(slots, summary_df)
        if not blocks_df.empty:
            df_with_conditions = pd.concat([df_with_conditions, blocks_df], ignore_index=True, sort=False)
            if 'is_mri_block' in df_with_conditions.columns:
                df_with_conditions['is_mri_block'] = df_with_conditions['is_mri_block'].fillna(False)
    start_time = time.time()
    df_corrected = correct_df_newrw(df_with_conditions, verbose)
    timing['correct_time'] = time.time() - start_time
    print(f"Runway slot calculation completed in {timing['correct_time']:.3f} seconds")
    df_run = df_corrected.copy()
    start_time = time.time()
    try:
        df_run, slots, df_dep_day = adapt_obp_vectorized(slots, df_run, [], 0, df_corrected, verbose)
        timing['adapt_time'] = time.time() - start_time
        print(f"Forward scheduling completed in {timing['adapt_time']:.3f} seconds")
    except Exception as e:
        print(f"Error during forward scheduling: {e}")
        return timing, df_corrected, None, original_stats
    start_time = time.time()
    df_run, df_dep_day = adjust_df_cur_optimized(df_run, df_corrected, 0, time.time(), time.time() - 600)
    timing['adjust_time'] = time.time() - start_time
    print(f"Slot adjustment completed in {timing['adjust_time']:.3f} seconds")
    if verbose > 0:
        print("DEBUG: Slot distribution before moveback:")
        slot_distribution = df_run['rw_cur'].value_counts().to_dict()
        for slot, count in sorted(slot_distribution.items()):
            print(f"DEBUG: Slot {int(slot)} has {count} flights")
    start_time = time.time()
    moveback_lst, nothing, slots, df_run, df_dep_day = adapt_obp_moveback_vectorized(
        slots, df_run, 0, df_dep_day, verbose)
    timing['moveback_time'] = time.time() - start_time
    print(f"Backward scheduling completed in {timing['moveback_time']:.3f} seconds")
    if verbose > 0:
        print("DEBUG: Slot distribution after moveback:")
        slot_distribution = df_run['rw_cur'].value_counts().to_dict()
        for slot, count in sorted(slot_distribution.items()):
            print(f"DEBUG: Slot {int(slot)} has {count} flights")
    timing['total_time'] = sum(timing.values())
    print(f"\nTotal processing time: {timing['total_time']:.3f} seconds")
    original_count = len(df_run)
    if verbose > 0:
        print("DEBUG: Slot distribution before deduplication:")
        slot_distribution = df_run['rw_cur'].value_counts().to_dict()
        for slot, count in sorted(slot_distribution.items()):
            print(f"DEBUG: Slot {int(slot)} has {count} flights")
    df_run_sorted = df_run.sort_values(['flight_key', 'slot_shift'], ascending=[True, False])
    df_run_unique = df_run_sorted.drop_duplicates(['flight_key'], keep='first')
    if 'taxi_time_minutes' not in df_run_unique.columns and 'taxi_time_minutes' in df_with_conditions.columns:
        taxi_lookup = df_with_conditions[['flight_key', 'taxi_time_minutes']].drop_duplicates('flight_key')
        df_run_unique = df_run_unique.merge(taxi_lookup, on='flight_key', how='left')
    if verbose > 0:
        print("DEBUG: Slot distribution after deduplication:")
        slot_distribution = df_run_unique['rw_cur'].value_counts().to_dict()
        for slot, count in sorted(slot_distribution.items()):
            print(f"DEBUG: Slot {int(slot)} has {count} flights")
    final_count = len(df_run_unique)
    if final_count < original_count:
        print(f"Note: Consolidated {original_count} flight assignments into {final_count} unique flights")
        print("      Each flight now has its final slot assignment")
    df_run_unique = calculate_tsat(df_run_unique, slots)
    if len(df_run_unique) > 0 and 'tsat' in df_run_unique.columns:
        valid_tsat_count = df_run_unique['tsat'].notna().sum()
        print(f"Calculated TSAT for {valid_tsat_count} flights")
    if {'earliest_ok_sec','rw_cur'}.issubset(df_run_unique.columns) and 'slot_end_sec' in slots.columns:
        end_map = slots['slot_end_sec'].to_dict()
        capacity = getattr(global_vars, 'AIRCRAFT_PER_SLOT', 6)
        slot_counts = df_run_unique['rw_cur'].value_counts().to_dict()
        adjusted = 0
        for idx, row in df_run_unique[df_run_unique['ctot'].notna()].iterrows():
            assigned_slot = int(row['rw_cur'])
            slot_end = end_map.get(assigned_slot)
            earliest_ok_sec = row.get('earliest_ok_sec')
            if pd.isna(earliest_ok_sec) or slot_end is None:
                continue
            if slot_end < earliest_ok_sec - 1e-6:
                target_slot = assigned_slot
                while True:
                    target_slot += 1
                    if target_slot not in end_map:
                        break
                    if end_map[target_slot] < earliest_ok_sec - 1e-6:
                        continue
                    if slot_counts.get(target_slot, 0) < capacity:
                        slot_counts[assigned_slot] = slot_counts.get(assigned_slot, 1) - 1
                        slot_counts[target_slot] = slot_counts.get(target_slot, 0) + 1
                        df_run_unique.at[idx, 'rw_cur'] = target_slot
                        if 'slot_shift' in df_run_unique.columns and 'rw_slot' in df_run_unique.columns:
                            orig = row.get('rw_slot') if pd.notna(row.get('rw_slot')) else assigned_slot
                            df_run_unique.at[idx, 'slot_shift'] = int(target_slot) - int(orig)
                        adjusted += 1
                    break
        if adjusted > 0:
            print(f"Post-processing: minimally moved {adjusted} CTOT flight(s) forward (interval enforcement).")
    if 'earliest_ok_sec' in df_run_unique.columns and 'rw_cur' in df_run_unique.columns and 'slot_center_sec' in slots.columns:
        center_map = slots['slot_center_sec'].to_dict()
        df_run_unique['assigned_center_sec'] = df_run_unique['rw_cur'].astype(int).map(center_map)
        early_mask = (df_run_unique['ctot'].notna()) & (df_run_unique['assigned_center_sec'] < (df_run_unique['earliest_ok_sec'] - 1e-6))
        early_count = early_mask.sum()
        if early_count > 0:
            print(f"WARNING: {early_count} CTOT flights assigned before earliest_ok (should be 0). Listing first 5:")
            print(df_run_unique.loc[early_mask, ['acid','rw_cur','assigned_center_sec','earliest_ok_sec','ctot']].head())
        else:
            print("Sanity check passed: no CTOT flights scheduled before earliest_ok center.")
    return timing, df_run_unique, slots, original_stats

# COMMAND ----------

# MAGIC %md ## Build Snapshots (optional)

# COMMAND ----------

closure_context = None
_mri_day = mri_day or build_snapshots
if _mri_day:
    maps, minutes, runway_list = build_mri_minute_maps(
        rwy_mri, _mri_day, runways=[runway] if runway else None
    )
    closure_context = {
        'day': pd.Timestamp(f"{_mri_day} 00:00:00"),
        'maps': maps,
        'minutes': minutes,
        'runways_from_map': runway_list,
    }
    if runway:
        closure_context['requested_runways'] = [str(runway)]
        closure_context['active_runways'] = [str(runway)]
    print(f"MRI availability loaded for runway(s): {closure_context.get('active_runways', runway_list)}")

if build_snapshots:
    from snapshot_generator_bins import generate_day_snapshots, HistoricalCSVDataSource, FULL_PIPELINE
    import snapshot_generator_bins as _KR
    _KR.ENFORCE_VACATED_CAPACITY_ON_FULL = True
    _KR.DISALLOW_PAST_SLOT_ASSIGNMENT = False
    day_str = build_snapshots
    day_start = pd.Timestamp(f"{day_str} 00:00:00")
    day_end = day_start + pd.Timedelta(hours=23, minutes=59)
    print(f"Building snapshots for {day_str} from {day_start} to {day_end} stride={minute_stride} min")
    initialize_slots(
        slot_duration_seconds=slot_duration,
        aircraft_per_slot=slot_capacity,
        ctot_min_margin=ctot_min_margin,
        ctot_max_margin=ctot_max_margin,
    )
    _, _, _, df_history, _ = load_TOBT_data_optimized(csv_name)
    try:
        df_ctot_meta = conditions_obp_optimized(df_history)
        if not df_ctot_meta.empty:
            keep_cols = [c for c in ['flight_key','ctot_updates','ctot_cancelled','last_ctot'] if c in df_ctot_meta.columns]
            if 'flight_key' in keep_cols:
                ctot_unique = df_ctot_meta[keep_cols].drop_duplicates('flight_key')
                df_history = df_history.merge(ctot_unique, on='flight_key', how='left')
    except Exception as _e:
        print(f"WARNING: CTOT metadata merge failed for snapshots: {_e}")
    if runway:
        print(f"Applying runway filter for snapshots (runway={runway}) using actual ATD runway logic...")
        filtered = filter_flights_by_actual_runway(df_history, runway, verbose=verbose)
        kept_flights = filtered['flight_key'].nunique()
        total_flights = df_history['flight_key'].nunique() if not df_history.empty else 0
        print(f"Snapshot runway filter kept {kept_flights}/{total_flights} flight_keys")
        df_history = filtered
    if 'timesec' in df_history.columns:
        if pd.api.types.is_datetime64_any_dtype(df_history['timesec']):
            df_history['timesec'] = (df_history['timesec'].astype('int64') // 10**9).astype('int64')
        df_history['timesec'] = pd.to_numeric(df_history['timesec'], errors='coerce').fillna(0).astype('int64')
    data_source = HistoricalCSVDataSource(df_history)
    if closure_context:
        active_snapshot_runways = sorted({str(r).strip() for r in df_history['trwy'].dropna().unique() if r})
        if active_snapshot_runways:
            closure_context['active_runways'] = active_snapshot_runways
    store = generate_day_snapshots(day_start, day_end, FULL_PIPELINE, data_source,
                                   minute_stride=minute_stride, verbose=verbose > 0,
                                   closure_context=closure_context)
    out_dir = Path(snapshots_out) / day_str
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(runway) if runway else "all"
    store.to_disk(out_dir, mode=snapshots_mode, include_hashes=False, suffix=suffix)
    print(f"Snapshots written to {out_dir} (base file: snapshot_{suffix}.parquet)")
    dbutils.notebook.exit(f"Snapshots written to {out_dir}")

# COMMAND ----------

# MAGIC %md ## Run Scheduler

# COMMAND ----------

timing, df_final, slots, original_stats = run_flight_scheduling(
    csv_name,
    slot_duration_seconds=slot_duration,
    aircraft_per_slot=slot_capacity,
    ctot_min_margin=ctot_min_margin,
    ctot_max_margin=ctot_max_margin,
    output=None,            # handle output below
    runway=runway,
    verbose=verbose,
    closure_context=closure_context,
)

# COMMAND ----------

# MAGIC %md ## Results

# COMMAND ----------

if df_final is not None and original_stats is not None:
    print(f"\nScheduled {len(df_final)} flights:")
    print(f"  - Unique flight keys:   {original_stats['unique_flight_keys']}")
    print(f"  - Unique sfplids:       {original_stats['unique_sfplids']}")
    print(f"  - Unique callsigns:     {original_stats['unique_callsigns']}")
    print(f"  - Total rows (updates): {original_stats['total_tobt_updates']}")
    print(f"  - Before processing:    {original_stats['before_atd_filter']}")
    print(f"  - After processing:     {len(df_final)}")
    flight_plan_changes = original_stats['flight_plan_updates']
    if flight_plan_changes > 0:
        print(f"  - Flight plan updates:  {flight_plan_changes}")
    else:
        print("  - Flight plan updates:  0 (no callsigns refiled)")
    processed_diff = original_stats['before_atd_filter'] - len(df_final)
    if processed_diff > 0:
        print(f"  - Processing removed: {processed_diff} flights (invalid data/missing components)")
    elif processed_diff < 0:
        print(f"  - Processing added: {abs(processed_diff)} flights (consolidated data)")

    if output:
        output_path = output if output.endswith('.csv') else output + '.csv'
        if not output_path.startswith('/dbfs') and not output_path.startswith('/Volumes'):
            output_path = f"/dbfs/tmp/{output_path}"
        from flight_keys import insert_flight_key_as_second_column
        df_to_save = insert_flight_key_as_second_column(df_final)
        df_to_save = convert_epoch_columns_to_datetime(df_to_save)
        df_to_save.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    if show_results:
        print_flight_results(df_final, slots, verbose=verbose)

    violations = check_ctot_violations(df_final, slots, verbose=verbose)
    if violations:
        print_ctot_violations(violations)

    if timing:
        print("\nDetailed timing information:")
        for step, duration in timing.items():
            print(f"  {step}: {duration:.3f} seconds")
else:
    print("Flight scheduling failed. Check errors above.")

# COMMAND ----------

# Display the result as a Spark DataFrame for easy exploration in Databricks
if df_final is not None:
    display(spark.createDataFrame(convert_epoch_columns_to_datetime(df_final).astype(str)))
