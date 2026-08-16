# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 — Flight Scheduler Analysis
# MAGIC
# MAGIC This notebook is the Databricks equivalent of `analysis.py`.
# MAGIC
# MAGIC Load the CSV output produced by **01_main_scheduler** and analyse ATD
# MAGIC (Actual Departure Time) compliance against assigned runway slots.
# MAGIC
# MAGIC **Prerequisites:** Run `00_setup.py` on the cluster first.
# MAGIC
# MAGIC **Widgets:**
# MAGIC | Widget | Description |
# MAGIC |--------|-------------|
# MAGIC | `csv_file` | DBFS path to the scheduler output CSV |
# MAGIC | `output` | DBFS path to save the analysis result CSV (blank = none) |
# MAGIC | `verbose` | Verbosity level (0/1/2) |

# COMMAND ----------

# ── Widget declarations ────────────────────────────────────────────────────────
dbutils.widgets.text("csv_file", "/dbfs/tmp/scheduler_output.csv", "Input CSV path (DBFS)")
dbutils.widgets.text("output",   "",                               "Output CSV path (blank = none)")
dbutils.widgets.text("verbose",  "0",                              "Verbosity (0/1/2)")

# COMMAND ----------

import sys, os
from pathlib import Path

_NB_DIR = Path(os.path.abspath("")).parent if Path(os.path.abspath("")).name == "notebooks" else Path(os.path.abspath(""))
if str(_NB_DIR) not in sys.path:
    sys.path.insert(0, str(_NB_DIR))

import pandas as pd
import numpy as np
from datetime import time

csv_file = dbutils.widgets.get("csv_file")
output   = dbutils.widgets.get("output") or None
verbose  = int(dbutils.widgets.get("verbose"))

print(f"csv_file={csv_file!r}  output={output!r}  verbose={verbose}")

# COMMAND ----------

# ── Analysis functions (identical to analysis.py) ──────────────────────────────

def convert_timestamp_to_time(timestamp_value):
    if pd.isna(timestamp_value):
        return pd.NA
    try:
        if isinstance(timestamp_value, (int, float)):
            if 946684800 <= timestamp_value <= 4102444800:
                dt = pd.to_datetime(timestamp_value, unit='s')
                return dt.strftime('%H:%M:%S')
            else:
                return pd.NA
        elif isinstance(timestamp_value, str):
            dt = pd.to_datetime(timestamp_value)
            return dt.strftime('%H:%M:%S')
        else:
            dt = pd.to_datetime(timestamp_value)
            return dt.strftime('%H:%M:%S')
    except Exception:
        return pd.NA


def analyze_atd_compliance(df, verbose=0):
    results = []
    for _, flight in df.iterrows():
        result = {
            'callsign': flight['acid'],
            'sfplid': flight['sfplid'],
            'assigned_slot_start': flight.get('rw_cur_starttime', None),
            'assigned_slot_end': flight.get('rw_cur_endtime', None),
            'latest_tobt': convert_timestamp_to_time(flight.get('tobt', pd.NA)),
            'latest_ctot': convert_timestamp_to_time(flight.get('ctot', pd.NA)),
            'latest_asrt': convert_timestamp_to_time(flight.get('asrt', pd.NA)),
            'atd': None,
            'within_slot': None,
            'deviation_minutes': None,
            'deviation_type': None
        }
        atd_value = flight.get('atd', np.nan)
        if pd.notna(atd_value):
            try:
                if isinstance(atd_value, (int, float)):
                    if 946684800 <= atd_value <= 4102444800:
                        atd_dt = pd.to_datetime(atd_value, unit='s')
                        result['atd'] = atd_dt.strftime('%H:%M:%S')
                        atd_time_obj = atd_dt.time()
                    else:
                        if verbose > 0:
                            print(f"DEBUG: Invalid timestamp for {flight['acid']}: {atd_value}")
                        continue
                elif isinstance(atd_value, str):
                    atd_dt = pd.to_datetime(atd_value)
                    result['atd'] = atd_dt.strftime('%H:%M:%S')
                    atd_time_obj = atd_dt.time()
                else:
                    atd_dt = pd.to_datetime(atd_value)
                    result['atd'] = atd_dt.strftime('%H:%M:%S')
                    atd_time_obj = atd_dt.time()
                if (pd.notna(flight.get('rw_cur_starttime')) and
                        pd.notna(flight.get('rw_cur_endtime'))):
                    slot_start = flight['rw_cur_starttime']
                    slot_end = flight['rw_cur_endtime']
                    def time_to_minutes(t):
                        if isinstance(t, time):
                            return t.hour * 60 + t.minute
                        elif isinstance(t, str):
                            t_obj = pd.to_datetime(t).time()
                            return t_obj.hour * 60 + t_obj.minute
                        return None
                    atd_minutes = time_to_minutes(atd_time_obj)
                    slot_start_minutes = time_to_minutes(slot_start)
                    slot_end_minutes = time_to_minutes(slot_end)
                    if all(x is not None for x in [atd_minutes, slot_start_minutes, slot_end_minutes]):
                        result['within_slot'] = slot_start_minutes <= atd_minutes <= slot_end_minutes
                        if not result['within_slot']:
                            if atd_minutes < slot_start_minutes:
                                result['deviation_minutes'] = slot_start_minutes - atd_minutes
                                result['deviation_type'] = 'EARLY'
                            else:
                                result['deviation_minutes'] = atd_minutes - slot_end_minutes
                                result['deviation_type'] = 'LATE'
                        else:
                            result['deviation_minutes'] = 0
                            result['deviation_type'] = 'ON_TIME'
            except Exception as e:
                if verbose > 0:
                    print(f"DEBUG: Error processing ATD for {flight.get('acid', 'UNKNOWN')}: {e}")
                continue
        results.append(result)
    return pd.DataFrame(results)


def print_analysis_summary(analysis_df):
    valid_flights = analysis_df[analysis_df['atd'].notna()]
    if valid_flights.empty:
        print("No flights with ATD data found.")
        return
    total_flights = len(valid_flights)
    on_time = len(valid_flights[valid_flights['within_slot']])
    early = len(valid_flights[valid_flights['deviation_type'] == 'EARLY'])
    late = len(valid_flights[valid_flights['deviation_type'] == 'LATE'])
    compliance_rate = (on_time / total_flights) * 100 if total_flights > 0 else 0
    print("\n" + "="*60)
    print(" ATD SLOT COMPLIANCE ANALYSIS")
    print("="*60)
    print(f"Total flights with ATD: {total_flights}")
    print(f"Within assigned slot: {on_time} ({compliance_rate:.1f}%)")
    print(f"Early departures: {early}")
    print(f"Late departures: {late}")
    non_compliant = valid_flights[~valid_flights['within_slot']]
    if not non_compliant.empty:
        print(f"\nNon-compliant flights ({len(non_compliant)}):")
        print("CALLSIGN   ATD        ASSIGNED SLOT      DEVIATION")
        print("-" * 55)
        for _, flight in non_compliant.iterrows():
            slot_str = f"{flight['assigned_slot_start']}-{flight['assigned_slot_end']}"
            if pd.notna(flight['deviation_minutes']):
                dev_str = f"{flight['deviation_type']} {int(flight['deviation_minutes'])}min"
            else:
                dev_str = "N/A"
            atd_display = flight['atd'] if pd.notna(flight['atd']) else "N/A"
            print(f"{flight['callsign']:<10} {atd_display:<10} {slot_str:<18} {dev_str}")
    print("="*60)

# COMMAND ----------

# MAGIC %md ## Load & Analyse

# COMMAND ----------

print(f"Loading data from {csv_file}...")
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} flights")

analysis_df = analyze_atd_compliance(df, verbose=verbose)
print_analysis_summary(analysis_df)

if output:
    analysis_df.to_csv(output, index=False)
    print(f"\nDetailed analysis saved to: {output}")

# COMMAND ----------

# Display interactive table
display(spark.createDataFrame(analysis_df.astype(str)))
