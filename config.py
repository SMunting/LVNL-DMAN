#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KPI configuration constants.

These values control default KPI horizon buckets, throughput windowing and
stability thresholds. They can be tuned without touching KPI logic.
"""

# A slot prediction shift larger than this (in minutes) counts as "substantial"
SUBSTANTIAL_SHIFT_THRESHOLD_MINUTES = 10

# Horizon buckets (minutes) – upper edges for categorisation of prediction horizon
# Example buckets produced: 0-15, 15-30, 30-60, 60-120, 120-180, >180
# Horizon buckets shortened for fine-grained near-term evaluation
HORIZONS_MINUTES = [5, 10, 15, 20, 25, 30]

# Minimum sample count to consider a horizon bucket statistically reportable
HORIZON_BUCKET_MIN_COUNT = 5

# Whether to drop buckets below minimum count (False keeps them with flag column)
HORIZON_BUCKET_DROP_UNDER_MIN = False

# Bootstrap settings for confidence intervals
BOOTSTRAP_ITERATIONS = 500  # increase for publication runs
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_RANDOM_SEED = 42
BOOTSTRAP_MIN_SAMPLE = 3  # skip CI if fewer samples

# Throughput rolling window configuration
THROUGHPUT_WINDOW_MINUTES = 60   # window size
THROUGHPUT_STEP_MINUTES = 15     # slide step

# Pilot annoyance heuristic: last substantial shift within this many minutes
# before (final) planned take‑off flags the flight (combined with >=2 substantial shifts)
PILOT_ANNOYANCE_WINDOW_MINUTES = 60

# Near adherence definition (slot indices)
NEAR_ADHERENCE_SLOT_TOLERANCE = 1  # within +/- 1 slot counts as near-adherent

# Output folder for plots (will be created under ./output if missing)
PLOTS_SUBDIR = "plots"
