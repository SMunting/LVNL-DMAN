# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 — Pairwise (Continuous) Flight Scheduler
# MAGIC
# MAGIC This notebook is the Databricks equivalent of `main_pairwise.py`.
# MAGIC
# MAGIC **Prerequisites:** Run `00_setup.py` on the cluster first.
# MAGIC
# MAGIC **Widgets (parameters):**
# MAGIC | Widget | Description | Default |
# MAGIC |--------|-------------|---------|
# MAGIC | `csv` | CSV file stem (no extension) inside `data/converted/` | `4_slot_overfill` |
# MAGIC | `ctot_min_margin` | Minutes before CTOT still allowed | `5` |
# MAGIC | `ctot_max_margin` | Minutes after CTOT still allowed | `10` |
# MAGIC | `runway` | Filter to this runway only (leave blank = all) | `` |
# MAGIC | `output` | Full output path for result CSV (blank = no file saved) | `` |
# MAGIC | `show_results` | Print full results table (`true`/`false`) | `false` |
# MAGIC | `build_snapshots` | YYYY-MM-DD to build minute snapshots (blank = skip) | `` |
# MAGIC | `minute_stride` | Minutes between snapshots | `1` |
# MAGIC | `snapshots_out` | DBFS path for snapshot output | `/dbfs/tmp/snapshots` |
# MAGIC | `snapshots_mode` | `full` or `compact` | `compact` |
# MAGIC | `mri_day` | YYYY-MM-DD to enforce MRI runway closures (blank = none) | `` |
# MAGIC | `verbose` | Verbosity level (0/1/2) | `0` |

# COMMAND ----------

# ── Widget declarations ────────────────────────────────────────────────────────
dbutils.widgets.text("csv",              "4_slot_overfill",    "CSV file stem")
dbutils.widgets.text("ctot_min_margin", "5",                  "CTOT min margin (min)")
dbutils.widgets.text("ctot_max_margin", "10",                 "CTOT max margin (min)")
dbutils.widgets.text("runway",          "",                   "Runway filter (blank = all)")
dbutils.widgets.text("output",          "",                   "Output path (blank = none)")
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

_NB_DIR = Path(os.path.abspath("")).parent if Path(os.path.abspath("")).name == "notebooks" else Path(os.path.abspath(""))
if str(_NB_DIR) not in sys.path:
    sys.path.insert(0, str(_NB_DIR))

csv_name        = dbutils.widgets.get("csv")
ctot_min_margin = int(dbutils.widgets.get("ctot_min_margin")) if dbutils.widgets.get("ctot_min_margin") else None
ctot_max_margin = int(dbutils.widgets.get("ctot_max_margin")) if dbutils.widgets.get("ctot_max_margin") else None
runway          = dbutils.widgets.get("runway")   or None
output          = dbutils.widgets.get("output")   or None
show_results    = dbutils.widgets.get("show_results") == "true"
build_snapshots = dbutils.widgets.get("build_snapshots") or None
minute_stride   = int(dbutils.widgets.get("minute_stride"))
snapshots_out   = dbutils.widgets.get("snapshots_out")
snapshots_mode  = dbutils.widgets.get("snapshots_mode")
mri_day         = dbutils.widgets.get("mri_day") or None
verbose         = int(dbutils.widgets.get("verbose"))

print(f"csv={csv_name!r}  ctot_min={ctot_min_margin}  ctot_max={ctot_max_margin}")
print(f"runway={runway!r}  output={output!r}  show_results={show_results}  verbose={verbose}")
print(f"build_snapshots={build_snapshots!r}  mri_day={mri_day!r}")

# COMMAND ----------

# ── Imports ────────────────────────────────────────────────────────────────────
import time
import pandas as pd

from data_loader import load_TOBT_data_optimized
from ctot_analyzer import conditions_obp_optimized
from pairwise_scheduler import run_pairwise_scheduler
from utils import print_flight_results, convert_epoch_columns_to_datetime
from data.runway.runway_mri import (
    rwy_mri,
    build_mri_minute_maps,
    build_takeoff_capacity_blocks,
)
import global_vars

# COMMAND ----------

# ── Helper functions (identical to main_pairwise.py) ───────────────────────────

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
        runway_by_flight_key = atd_df.groupby('flight_key')['trwy'].apply(lambda x: x.unique())
        for fk, runways in runway_by_flight_key.items():
            if len(runways) > 1:
                multi_runway_warnings.append((fk, runways))
                actual_runway = str(runways[0])
            else:
                actual_runway = str(runways[0])
            if actual_runway == target_runway_str:
                flight_keys_to_keep.add(fk)
    all_flight_keys = set(df['flight_key'].unique())
    flight_keys_with_atd = set(runway_by_flight_key.index) if not atd_df.empty else set()
    flight_keys_no_atd = all_flight_keys - flight_keys_with_atd
    if multi_runway_warnings:
        print(f"WARNING: {len(multi_runway_warnings)} flight(s) have ATD values for multiple runways:")
        for fk, runways in multi_runway_warnings[:5]:
            print(f"  - Flight {fk}: runways {list(runways)}")
        if len(multi_runway_warnings) > 5:
            print(f"  ... and {len(multi_runway_warnings) - 5} more")
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
        print(f"DEBUG: Flight keys without ATD (sample): {list(flight_keys_no_atd)[:5]}")
    return filtered_df


def run_pairwise_flight_scheduling(csv_name, ctot_min_margin=None, ctot_max_margin=None,
                                    output=None, runway=None, verbose=0, closure_context=None):
    print(f"Running continuous pairwise scheduling for {csv_name}...")
    print("Mode: Continuous pairwise sequencing (no fixed slots)")
    print("Taxi time: dynamic per-flight matrix (stand/runway)")
    timing = {}
    if ctot_min_margin is not None:
        global_vars.CTOT_MIN_MARGIN = ctot_min_margin
    if ctot_max_margin is not None:
        global_vars.CTOT_MAX_MARGIN = ctot_max_margin
    start_time = time.time()
    try:
        _, _, df_dep, df_history, original_stats = load_TOBT_data_optimized(csv_name)
        timing['load_time'] = time.time() - start_time
        print(f"Data loading completed in {timing['load_time']:.3f} seconds")
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
    capacity_blocks_df = None
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
        from slot_manager import initialize_slots
        slots_template = initialize_slots()
        capacity_blocks_df, _ = build_takeoff_capacity_blocks(
            closure_context['maps'],
            closure_context['minutes'],
            slots_template,
            day=closure_context['day'],
            slot_duration_seconds=getattr(global_vars, 'SLOT_DURATION_SECONDS', 600),
            runway_filter=closure_context['active_runways'],
        )
    from snapshot_generator import _build_state_dataframe
    df_state = _build_state_dataframe(df_with_conditions)
    start_time = time.time()
    scheduled_dicts, vacated_map, tsat_map = run_pairwise_scheduler(
        df_state,
        frozen_assignments={},
        now_ts=None,
        capacity_blocks=capacity_blocks_df,
        existing_vacated=None,
        prev_slot_map=None,
    )
    timing['scheduling_time'] = time.time() - start_time
    print(f"Pairwise scheduling completed in {timing['scheduling_time']:.3f} seconds")
    if scheduled_dicts:
        df_final = pd.DataFrame(scheduled_dicts)
        if 'ttot' in df_final.columns:
            df_final['rw_cur'] = df_final['rw_seq']
            df_final['rw_cur_starttime'] = df_final['ttot']
            df_final['rw_cur_endtime'] = df_final['ttot']
    else:
        df_final = pd.DataFrame()
    timing['total_time'] = sum(timing.values())
    print(f"\nTotal processing time: {timing['total_time']:.3f} seconds")
    if not df_final.empty:
        print(f"\nScheduled {len(df_final)} flights")
        if vacated_map:
            total_vacated = sum(len(v) for v in vacated_map.values())
            print(f"Vacated (missed TSAT) slots: {total_vacated}")
    if not df_final.empty and not df_with_conditions.empty:
        df_orig_dedup = df_with_conditions.sort_values(['flight_key', 'timesec'], ascending=[True, False]) if 'timesec' in df_with_conditions.columns else df_with_conditions
        df_orig_dedup = df_orig_dedup.drop_duplicates(['flight_key'], keep='first')
        merge_cols = [c for c in df_orig_dedup.columns if c not in df_final.columns or c == 'flight_key']
        df_final = pd.merge(df_final, df_orig_dedup[merge_cols], on='flight_key', how='left')
    if output and not df_final.empty:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_epoch_columns_to_datetime(df_final).to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    return timing, df_final, None, original_stats

# COMMAND ----------

# MAGIC %md ## Build Snapshots (optional)

# COMMAND ----------

closure_context = None
_mri_day = mri_day or build_snapshots
if _mri_day:
    maps, minutes, runway_list = build_mri_minute_maps(
        rwy_mri, _mri_day, runways=runway
    )
    closure_context = {
        'day': pd.Timestamp(f"{_mri_day} 00:00:00"),
        'maps': maps,
        'minutes': minutes,
        'runways_from_map': runway_list,
    }
    if runway:
        closure_context['requested_runways'] = [str(runway)]

if build_snapshots:
    from snapshot_generator import (
        generate_day_snapshots, FULL_PIPELINE,
        HistoricalCSVDataSource,
    )
    day = pd.to_datetime(build_snapshots).normalize()
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(hours=23, minutes=59)
    print(f"Building snapshots for {day.date()} (stride={minute_stride} min)")
    _, _, df_dep, df_history, _ = load_TOBT_data_optimized(csv_name)
    if df_history.empty:
        raise RuntimeError("No history data available for snapshot generation")
    if runway:
        print(f"Applying runway filter for snapshots (runway={runway}) using actual ATD runway logic...")
        filtered = filter_flights_by_actual_runway(df_history, runway, verbose=verbose)
        kept_flights = filtered['flight_key'].nunique()
        total_flights = df_history['flight_key'].nunique() if not df_history.empty else 0
        print(f"Snapshot runway filter kept {kept_flights}/{total_flights} flight_keys")
        df_history = filtered
    data_source = HistoricalCSVDataSource(df_history)
    store = generate_day_snapshots(
        day_start, day_end, FULL_PIPELINE, data_source,
        minute_stride=minute_stride, verbose=verbose > 0,
        event_driven=False, closure_context=closure_context,
    )
    out_folder = Path(snapshots_out) / day.strftime('%Y-%m-%d')
    out_folder.mkdir(parents=True, exist_ok=True)
    suffix = str(runway) if runway else "all"
    store.to_disk(out_folder, mode=snapshots_mode, include_hashes=False, suffix=suffix)
    print(f"Snapshots written to {out_folder}")
    print(f"Total snapshots: {len(store.times())}")
    dbutils.notebook.exit(f"Snapshots written to {out_folder}")

# COMMAND ----------

# MAGIC %md ## Run Pairwise Scheduler

# COMMAND ----------

timing, df_final, _, original_stats = run_pairwise_flight_scheduling(
    csv_name,
    ctot_min_margin=ctot_min_margin,
    ctot_max_margin=ctot_max_margin,
    output=output,
    runway=runway,
    verbose=verbose,
    closure_context=closure_context,
)

# COMMAND ----------

# MAGIC %md ## Results

# COMMAND ----------

if df_final is not None and not df_final.empty:
    if show_results:
        extra_columns = []
        print_flight_results(df_final, None, verbose=verbose, extra_columns=extra_columns)
    print("\nScheduling Summary:")
    print(f"  Total flights processed: {len(df_final)}")
    if 'rwy' in df_final.columns:
        by_runway = df_final['rwy'].value_counts()
        print("  Flights by runway:")
        for rwy, count in by_runway.items():
            print(f"    {rwy}: {count}")
    if 'status' in df_final.columns:
        by_status = df_final['status'].value_counts()
        print("  Flights by status:")
        for status, count in by_status.items():
            print(f"    {status}: {count}")
    if timing:
        print("\nDetailed timing information:")
        for step, duration in timing.items():
            print(f"  {step}: {duration:.3f} seconds")
else:
    print("\nNo results to display")

# COMMAND ----------

# Display the result as a Spark DataFrame for easy exploration in Databricks
if df_final is not None and not df_final.empty:
    display(spark.createDataFrame(convert_epoch_columns_to_datetime(df_final).astype(str)))
