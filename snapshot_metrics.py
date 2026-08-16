#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot Metrics CLI

Compute evaluation metrics directly from snapshot stores produced by
main.py --build-snapshots. The loader is shared with snapshot_explorer.py
to ensure changes remain in one place.

Metrics covered (see thesis evaluation section):
- Predictive accuracy (requires history CSV for AOBT/ATOT)
- Schedule stability (from slot assignment changes per flight)
- Throughput utilisation (overall and per-hour)
- Timeliness of stable schedule
- Equity (requires history CSV to provide categories)

Usage examples:
  python snapshot_metrics.py --store output/snapshots/2024-08-06/snapshot_24.parquet
  python snapshot_metrics.py --store output/snapshots/2024-08-06 --history-csv data/converted/VEMMIS_cdm_outbound_2024_aug_6-8_converted.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import math
from typing import Dict, List, Any, Optional, Iterable
import numpy as np
import pandas as pd

import global_vars
from snapshot_explorer import load_snapshot_dataframe, SnapshotExplorer, _resolve_store_path
from flight_metrics import (
    predictive_accuracy,
    schedule_stability,
    throughput_utilisation,
    timeliness_of_stable_schedule,
    equity_measures,
)


def _slot_minutes(slot_idx: int) -> float:
    return (int(slot_idx) * global_vars.SLOT_DURATION_SECONDS) / 60.0


def _build_assignments(df_snap: pd.DataFrame) -> Dict[str, List[tuple[float, float]]]:
    """Return per-flight assignment change sequence.

    For each flight_key, produce a list of (timestamp_minutes, slot_minutes)
    entries ordered by snapshot_time, only when rw_cur changes.
    """
    out: Dict[str, List[tuple[float, float]]] = {}
    grp = df_snap.sort_values(['flight_key', 'snapshot_time']).groupby('flight_key', sort=False)
    for fk, g in grp:
        g = g[g['rw_cur'].notna()].copy()
        if g.empty:
            continue
        last = None
        seq: List[tuple[float, float]] = []
        for _, r in g.iterrows():
            cur = int(r.rw_cur)
            if last is None or cur != last:
                tmin = r.snapshot_time.value / 60_000_000_000  # ns -> minutes
                seq.append((tmin, _slot_minutes(cur)))
                last = cur
        if seq:
            out[str(fk)] = seq
    return out


def _freeze_time_minutes(history: pd.DataFrame) -> Optional[float]:
    """First snapshot_time (minutes) where is_frozen is True; None if never frozen."""
    if 'is_frozen' in history.columns:
        fr = history[history['is_frozen'] == True]  # noqa: E712
        if not fr.empty:
            ts = pd.to_datetime(fr['snapshot_time'].iloc[0])
            return ts.value / 60_000_000_000
    # Fallback: first time status becomes FROZEN/TAKEN_OFF
    if 'status' in history.columns:
        fr2 = history[history['status'].isin(['FROZEN', 'TAKEN_OFF'])]
        if not fr2.empty:
            ts = pd.to_datetime(fr2['snapshot_time'].iloc[0])
            return ts.value / 60_000_000_000
    return None


def _build_timeliness_inputs(df_snap: pd.DataFrame, aobt_minutes_map: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    flights: List[Dict[str, Any]] = []
    for fk, g in df_snap.sort_values('snapshot_time').groupby('flight_key', sort=False):
        seq = []
        last = None
        g2 = g[g['rw_cur'].notna()].copy()
        for _, r in g2.iterrows():
            cur = int(r.rw_cur)
            if last is None or cur != last:
                tmin = r.snapshot_time.value / 60_000_000_000
                seq.append((tmin, _slot_minutes(cur)))
                last = cur
        if not seq:
            continue
        # Prefer AOBT from history map when available
        aobt_min = None
        if aobt_minutes_map and str(fk) in aobt_minutes_map:
            aobt_min = aobt_minutes_map[str(fk)]
        if aobt_min is None:
            aobt_min = _freeze_time_minutes(g)
        if aobt_min is None:
            # Not frozen within snapshot horizon; skip for timeliness
            continue
        flights.append({
            'flight_key': str(fk),
            'aobt': aobt_min,
            'assignments': seq,
        })
    return flights


def _throughput_from_final(final_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute throughput with awareness of VACATED slots.

    - used_real: flights excluding synthetic VACATED__ entries
    - vacated: count of VACATED entries (blocked capacity) per hour
    - used_including_vacated: used_real + vacated (should not exceed capacity under consistent config)
    """
    cap_per_hour = int(3600 / global_vars.SLOT_DURATION_SECONDS) * getattr(global_vars, 'AIRCRAFT_PER_SLOT', 6)
    per_hour: List[Dict[str, Any]] = []
    note = None
    if 'ttot' in final_df.columns and final_df['ttot'].notna().any():
        dfh = final_df.copy()
        dfh['ttot_dt'] = pd.to_datetime(dfh['ttot'], utc=True, errors='coerce')
        dfh['hour'] = dfh['ttot_dt'].dt.floor('h')
        s = dfh['status'].astype(str) if 'status' in dfh.columns else pd.Series(False, index=dfh.index)
        k = dfh['flight_key'].astype(str) if 'flight_key' in dfh.columns else pd.Series('', index=dfh.index)
        is_vac = s.eq('VACATED') | k.str.startswith('VACATED__')
        used_real_counts = dfh.loc[~is_vac, 'hour'].value_counts().sort_index()
        vac_counts = dfh.loc[is_vac, 'hour'].value_counts().sort_index()
        all_hours = sorted(set(used_real_counts.index).union(set(vac_counts.index)))
        for hour in all_hours:
            used_real = int(used_real_counts.get(hour, 0))
            vacated = int(vac_counts.get(hour, 0))
            used_incl = used_real + vacated
            util_real = throughput_utilisation(used_real, cap_per_hour)
            util_incl = throughput_utilisation(used_incl, cap_per_hour)
            effective_cap_after_vacated = max(0, cap_per_hour - vacated)
            util_real_vs_effective = 0.0 if effective_cap_after_vacated == 0 else throughput_utilisation(used_real, effective_cap_after_vacated)
            if used_incl > cap_per_hour:
                note = 'Observed scheduled entries (incl. vacated) exceed declared capacity. This may indicate a config mismatch (slot size/capacity) between snapshot generation and current globals) or multi-runway aggregation.'
            per_hour.append({
                'hour_start': pd.Timestamp(hour).isoformat(),
                'used_slots_real': used_real,
                'vacated_slots': vacated,
                'used_slots_including_vacated': used_incl,
                'declared_capacity': int(cap_per_hour),
                'effective_capacity_after_vacated': int(effective_cap_after_vacated),
                'utilisation_pct_real': float(util_real),
                'utilisation_pct_including_vacated': float(util_incl),
                'utilisation_pct_real_vs_effective': float(util_real_vs_effective),
            })
        overall_used_real = int((~is_vac).sum())
        overall_vacated = int(is_vac.sum())
        overall_used_incl = overall_used_real + overall_vacated
        # overall capacity = span across first/last TTOT rounded to hours
        tmin = dfh['ttot_dt'].min()
        tmax = dfh['ttot_dt'].max()
        span_seconds = (tmax.ceil('h') - tmin.floor('h')).total_seconds() if pd.notna(tmin) and pd.notna(tmax) else 0
        span_hours = max(1, math.ceil(span_seconds / 3600.0)) if span_seconds > 0 else 1
        overall_cap = span_hours * cap_per_hour
        overall = {
            'used_slots_real': overall_used_real,
            'vacated_slots': overall_vacated,
            'used_slots_including_vacated': overall_used_incl,
            'declared_capacity': int(overall_cap),
            'capacity_per_hour': int(cap_per_hour),
            'utilisation_pct_real': float(throughput_utilisation(overall_used_real, overall_cap)),
            'utilisation_pct_including_vacated': float(throughput_utilisation(overall_used_incl, overall_cap)),
        }
    else:
        per_hour = []
        overall = {
            'used_slots_real': 0,
            'vacated_slots': 0,
            'used_slots_including_vacated': 0,
            'declared_capacity': 0,
            'capacity_per_hour': int(cap_per_hour),
            'utilisation_pct_real': 0.0,
            'utilisation_pct_including_vacated': 0.0,
        }
        note = 'No TTOT values present; overall declared_capacity set to 0.'
    return {
        'overall': overall,
        'per_hour': per_hour,
        'note': note,
    }


def _summ_stats(series: pd.Series) -> Dict[str, float]:
    if series is None or series.empty:
        return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'max': 0.0}
    s = series.dropna().astype(float)
    if s.empty:
        return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'max': 0.0}
    return {
        'count': int(len(s)),
        'mean': float(s.mean()),
        'std': float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        'min': float(s.min()),
        'p25': float(s.quantile(0.25)),
        'p50': float(s.quantile(0.50)),
        'p75': float(s.quantile(0.75)),
        'max': float(s.max()),
    }


def _stability_distributions(assignments: Dict[str, List[tuple[float, float]]], out_dir: Path, file_suffix: Optional[str] = None) -> Dict[str, Any]:
    """Compute per-flight resequences and total change (minutes), persist CSV, and return summaries.

    - resequences = number of slot changes = max(0, len(seq) - 1)
    - change_minutes = sum of absolute differences in slot_minutes between consecutive assignments
    """
    rows = []
    for fk, seq in assignments.items():
        if not seq or len(seq) < 2:
            rows.append({'flight_key': fk, 'resequences': 0, 'change_minutes': 0.0})
            continue
        changes = 0
        total_change = 0.0
        prev_slot = seq[0][1]
        for i in range(1, len(seq)):
            slot_min = seq[i][1]
            if slot_min != prev_slot:
                changes += 1
                total_change += abs(slot_min - prev_slot)
                prev_slot = slot_min
        rows.append({'flight_key': fk, 'resequences': int(changes), 'change_minutes': float(total_change)})
    df_out = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = f"_{file_suffix}" if file_suffix else ""
    fpath = out_dir / f'metrics_stability_changes{suf}.csv'
    df_out.to_csv(fpath, index=False)
    return {
        'resequences': _summ_stats(df_out['resequences']),
        'change_minutes': _summ_stats(df_out['change_minutes']),
        'file': str(fpath.resolve()),
    }


def _differences_from_history(final_df: pd.DataFrame, history_csv: str, out_dir: Path, file_suffix: Optional[str] = None) -> Dict[str, Any]:
    """Compute and persist per-flight differences for:
    - ATD vs TTOT (actual take-off vs target take-off)
    - AOBT vs TOBT (actual off-block vs target off-block)
    - ASRT vs TSAT (airport slot ref time vs target start-up)
    Writes three CSVs with deltas in minutes.
    """
    from data_loader import load_TOBT_data_optimized
    df, _atd, df_dep, _hist, _stats = load_TOBT_data_optimized(history_csv)
    if df_dep.empty:
        return {}
    dep = df_dep.copy()
    keep_cols = ['flight_key','acid','dest','asrt','aobt','atd']
    dep = dep[keep_cols].drop_duplicates('flight_key', keep='first')
    fin = final_df[['flight_key','ttot','tobt','tsat','rwy','status']].copy()
    fin['ttot_dt'] = pd.to_datetime(fin['ttot'], utc=True, errors='coerce')
    fin['tobt_dt'] = pd.to_datetime(fin['tobt'], utc=True, errors='coerce')
    fin['tsat_dt'] = pd.to_datetime(fin['tsat'], utc=True, errors='coerce')
    merged = fin.merge(dep, on='flight_key', how='left')
    # Convert actuals
    for col in ['atd','aobt','asrt']:
        merged[f'{col}_dt'] = pd.to_datetime(merged[col], unit='s', utc=True, errors='coerce')
    # Delta helpers
    def mins(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return float((a - b).total_seconds() / 60.0)
    # Compute deltas
    merged['delta_atd_vs_ttot_min'] = [mins(a, b) for a, b in zip(merged['atd_dt'], merged['ttot_dt'])]
    merged['delta_aobt_vs_tobt_min'] = [mins(a, b) for a, b in zip(merged['aobt_dt'], merged['tobt_dt'])]
    merged['delta_asrt_vs_tsat_min'] = [mins(a, b) for a, b in zip(merged['asrt_dt'], merged['tsat_dt'])]
    # Persist distributions
    out_dir.mkdir(parents=True, exist_ok=True)
    cols_base = ['flight_key','acid','dest','rwy','status']
    d1 = merged[cols_base + ['ttot_dt','atd_dt','delta_atd_vs_ttot_min']].copy()
    d2 = merged[cols_base + ['tobt_dt','aobt_dt','delta_aobt_vs_tobt_min']].copy()
    d3 = merged[cols_base + ['tsat_dt','asrt_dt','delta_asrt_vs_tsat_min']].copy()
    # Name files with optional date+runway suffix
    suf = f"_{file_suffix}" if file_suffix else ""
    f1 = out_dir / f'metrics_diff_atd_vs_ttot{suf}.csv'
    f2 = out_dir / f'metrics_diff_aobt_vs_tobt{suf}.csv'
    f3 = out_dir / f'metrics_diff_asrt_vs_tsat{suf}.csv'
    d1.to_csv(f1, index=False)
    d2.to_csv(f2, index=False)
    d3.to_csv(f3, index=False)
    # Summaries
    return {
        'atd_vs_ttot': _summ_stats(d1['delta_atd_vs_ttot_min']),
        'aobt_vs_tobt': _summ_stats(d2['delta_aobt_vs_tobt_min']),
        'asrt_vs_tsat': _summ_stats(d3['delta_asrt_vs_tsat_min']),
        'files': {
            'atd_vs_ttot': str(f1.resolve()),
            'aobt_vs_tobt': str(f2.resolve()),
            'asrt_vs_tsat': str(f3.resolve()),
        }
    }


def _predictive_accuracy_and_equity(final_df: pd.DataFrame, history_csv: str, equity_key: str) -> Dict[str, Any]:
    # Lazy import to avoid heavy loader unless requested
    from data_loader import load_TOBT_data_optimized

    # load history; accept full path
    csv_path = history_csv
    df, _atd, df_dep, _hist, _stats = load_TOBT_data_optimized(csv_path)
    if df_dep.empty:
        return {'predictive_accuracy': None, 'equity': None}
    # map actuals by flight_key
    dep = df_dep.copy()
    dep = dep[['flight_key', 'aobt', 'atd', 'tobt', 'acid', 'dest', 'actype', 'wtc']].drop_duplicates('flight_key', keep='first')
    dep['airline'] = dep['acid'].astype(str).str[:3]

    fin = final_df[['flight_key', 'ttot']].copy()
    fin['ttot_dt'] = pd.to_datetime(fin['ttot'], utc=True, errors='coerce')
    fin['scheduled_tot'] = fin['ttot_dt'].astype('int64') / 60_000_000_000  # minutes

    merged = fin.merge(dep, on='flight_key', how='left')
    # Build flights list
    flights_pa: List[Dict[str, Any]] = []
    for _, r in merged.iterrows():
        try:
            tobt_pred_min = pd.to_datetime(r['tobt'], unit='s', utc=True).value / 60_000_000_000 if pd.notna(r['tobt']) else None
            aobt_min = pd.to_datetime(r['aobt'], unit='s', utc=True).value / 60_000_000_000 if pd.notna(r['aobt']) else None
            atot_min = pd.to_datetime(r['atd'], unit='s', utc=True).value / 60_000_000_000 if pd.notna(r['atd']) else None
        except Exception:
            tobt_pred_min = aobt_min = atot_min = None
        entry = {
            'tobt_pred': tobt_pred_min,
            'aobt': aobt_min,
            'scheduled_tot': r.get('scheduled_tot'),
            'atot': atot_min,
            'airline': r.get('airline'),
            'dest': r.get('dest'),
            'actype': r.get('actype'),
            'wtc': r.get('wtc'),
        }
        flights_pa.append({k: v for k, v in entry.items() if v is not None})

    # filter valid entries for PA
    flights_pa_valid = [f for f in flights_pa if all(k in f for k in ('tobt_pred', 'aobt', 'scheduled_tot', 'atot'))]
    pa = predictive_accuracy(flights_pa_valid) if flights_pa_valid else None

    # equity
    if equity_key not in ('airline', 'dest', 'actype', 'wtc'):
        equity_key = 'airline'
    flights_eq = [
        {
            'scheduled_tot': f['scheduled_tot'],
            'atot': f['atot'],
            equity_key: f.get(equity_key)
        }
        for f in flights_pa_valid if f.get(equity_key) is not None
    ]
    eq = equity_measures(flights_eq, equity_key) if flights_eq else None
    return {'predictive_accuracy': pa, 'equity': eq}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compute metrics from snapshot store')
    p.add_argument('--store', required=True, help='Path to snapshot folder or snapshot_<suffix>.parquet')
    p.add_argument('--history-csv', help='Optional CSV with history to compute predictive accuracy and equity')
    p.add_argument('--equity-key', default='airline', help='Category for equity: airline|dest|actype|wtc')
    p.add_argument('--thresholds', default='60,45,30,20,15,10,5,1', help='Comma-separated lead times (minutes)')
    p.add_argument('--out', help='Optional output directory (creates date subfolder and places all outputs there)')
    p.add_argument('--wide', action='store_true', help='Print full JSON without truncation')
    p.add_argument('--runway', type=str, help='Force filtering to a single runway (e.g., 24, 18L, 36L)')
    return p.parse_args(list(argv) if argv is not None else None)


# def main(argv: Optional[Iterable[str]] = None) -> int:
#     args = parse_args(argv)
#     try:
#         df = load_snapshot_dataframe(args.store)
#     except Exception as e:
#         print(f"Error loading snapshot store: {e}")
#         return 1

#     explorer = SnapshotExplorer(df)
#     final_state = explorer.final_schedule()
#     final_df = final_state.schedule.copy()

def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        df_all = load_snapshot_dataframe(args.store)
    except Exception as e:
        print(f"Error loading snapshot store: {e}")
        return 1

    # --- optional single-runway filtering ---
    def _std(v):
        return None if pd.isna(v) else str(v).strip()

    df = df_all
    requested_runway = str(args.runway).strip() if getattr(args, 'runway', None) else None
    if requested_runway:
        # Fast path: filter by a runway column if present
        candidate_cols = [c for c in ['rwy', 'runway', 'runway_id'] if c in df_all.columns]
        if candidate_cols:
            col = candidate_cols[0]
            df = df_all[df_all[col].map(_std) == requested_runway].copy()
        else:
            # Fallback: derive allowed flights from final schedule, then filter snapshots
            explorer_tmp = SnapshotExplorer(df_all)
            final_df_tmp = explorer_tmp.final_schedule().schedule.copy()
            if 'rwy' in final_df_tmp.columns:
                allowed_fks = set(
                    final_df_tmp.loc[final_df_tmp['rwy'].map(_std) == requested_runway, 'flight_key']
                    .astype(str).tolist()
                )
                if allowed_fks:
                    df = df_all[df_all['flight_key'].astype(str).isin(allowed_fks)].copy()
        if df.empty:
            print(f"No snapshot rows for runway={requested_runway}; exiting.")
            return 0

    explorer = SnapshotExplorer(df)
    final_state = explorer.final_schedule()
    final_df = final_state.schedule.copy()


    # Derive day and runway labels
    day_str = pd.Timestamp(explorer.first_snapshot_time).date().isoformat()
    try:
        src_path = _resolve_store_path(args.store)
        base_name = src_path.name
    except Exception:
        base_name = ''
    runway = None
    if base_name.startswith('snapshot_'):
        suffix = base_name.split('.', 1)[0].split('_', 1)[1]
        runway = suffix
    elif base_name in ('snapshots_all.parquet', 'snapshots_all.csv'):
        runway = 'all'
    # Fallback to data if not from filename
    if not runway:
        if 'rwy' in final_df.columns and final_df['rwy'].notna().any():
            uniq = sorted(set(str(x) for x in final_df['rwy'].dropna().unique()))
            runway = uniq[0] if len(uniq) == 1 else 'mixed'
        else:
            runway = 'unknown'

    if requested_runway:
        runway = requested_runway

    # Sanitize runway for filenames
    runway_label = ''.join(ch for ch in str(runway) if ch.isalnum()) or 'unknown'
    file_suffix = f"{day_str}_rwy{runway_label}"

    # Set up output directory structure
    if args.out:
        base_out_dir = Path(args.out)
    else:
        base_out_dir = Path('output') / 'metrics'
    
    # Create date-specific subdirectory
    day_out_dir = base_out_dir / day_str
    day_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the same directory for distributions
    dist_dir = day_out_dir / 'distributions'

    # Stability
    assignments = _build_assignments(df)
    flights_stab = []
    for fk, seq in assignments.items():
        flights_stab.append({'flight_key': fk, 'assignments': seq})
    stab = schedule_stability(flights_stab)
    # Stability distributions (per-flight)
    stab_dists = _stability_distributions(assignments, dist_dir, file_suffix)

    # Timeliness
    thresholds = [int(x.strip()) for x in str(args.thresholds).split(',') if x.strip()]
    # Build optional AOBT map for timeliness when history CSV is provided
    aobt_map: Optional[Dict[str, float]] = None
    if args.history_csv:
        try:
            from data_loader import load_TOBT_data_optimized
            _df, _atd, df_dep_tm, _hist, _stats = load_TOBT_data_optimized(args.history_csv)
            if not df_dep_tm.empty:
                df_dep_tm = df_dep_tm[['flight_key','aobt']].drop_duplicates('flight_key', keep='first')
                df_dep_tm['aobt_dt'] = pd.to_datetime(df_dep_tm['aobt'], unit='s', utc=True, errors='coerce')
                aobt_map = {
                    str(r.flight_key): (r.aobt_dt.value / 60_000_000_000)
                    for _, r in df_dep_tm.iterrows() if pd.notna(r.aobt_dt)
                }
        except Exception:
            aobt_map = None
    flights_time = _build_timeliness_inputs(df, aobt_map)
    timely = timeliness_of_stable_schedule(flights_time, thresholds)

    # Throughput
    thr = _throughput_from_final(final_df)

    # Predictive accuracy + Equity (optional)
    pa_eq = {'predictive_accuracy': None, 'equity': None}
    diffs_summary = None
    if args.history_csv:
        pa_eq = _predictive_accuracy_and_equity(final_df, args.history_csv, args.equity_key)
        # Detailed differences and distributions (day-scoped directory)
        diffs_summary = _differences_from_history(final_df, args.history_csv, dist_dir, file_suffix)

    result = {
        'source_store': str(args.store),
        'period': {
            'first_snapshot': explorer.first_snapshot_time.isoformat(),
            'last_snapshot': explorer.last_snapshot_time.isoformat(),
        },
        'context': {
            'day': day_str,
            'runway': runway
        },
        'units': {
            'differences': 'minutes',
            'throughput': 'percentage',
        },
        'counts': {
            'flights_final': int(len(final_df)),
            'snapshot_rows': int(len(df)),
        },
        'stability': {
            **stab,
            'distributions': stab_dists,
        },
        'timeliness': timely,
        'throughput': thr,
        'predictive_accuracy': pa_eq.get('predictive_accuracy'),
        'equity': pa_eq.get('equity'),
        'differences': diffs_summary,
    }

    # Output JSON file in the date directory
    json_filename = f'snapshot_metrics_{file_suffix}.json'
    out_path = day_out_dir / json_filename

    try:
        with out_path.open('w') as f:
            json.dump(result, f, indent=2)
        pretty = result if args.wide else {
            **result,
            'throughput': {
                **result['throughput'],
                'per_hour': result['throughput']['per_hour'][:6]  # trim
            }
        }
        print(json.dumps(pretty, indent=2))
        if not args.wide:
            print("Note: per_hour truncated in console (showing first 6). Full list is in the JSON file.")
        print(f"Wrote metrics -> {out_path}")
    except Exception as e:
        print(json.dumps(result, indent=2))
        print(f"Warning: could not write to {out_path}: {e}")
    return 0


if __name__ == '__main__': 
    raise SystemExit(main())
