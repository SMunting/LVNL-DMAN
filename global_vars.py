#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global variables for the flight scheduling system.
Contains shared data structures and configuration settings used across multiple modules.
"""

import pandas as pd
import datetime
import math

# Default slot configuration
SLOT_DURATION_SECONDS = 600  # Default: 10 minutes (600 seconds)
SLOT_SIZE = 6  # Default: 6 aircraft per slot
# Deprecated: dynamic taxi times are resolved per flight; retain attribute for legacy access.
TAXI_TIME_MINUTES = None
CTOT_MIN_MARGIN = 5  # Default: 5 minutes before CTOT
CTOT_MAX_MARGIN = 10  # Default: 10 minutes after CTOT
SLOTS_PER_HOUR = int(round(3600 / SLOT_DURATION_SECONDS))  # informational only
# Use ceiling so the final partial interval (if any) is represented
TOTAL_SLOTS = math.ceil(86400 / SLOT_DURATION_SECONDS)

# Global slot definition
slots = pd.DataFrame(index=range(TOTAL_SLOTS))

def configure_slots(duration_seconds=None, aircraft_per_slot=None, taxi_time_minutes=None,
                 ctot_min_margin=None, ctot_max_margin=None):
    """
    Configure global slot settings with custom parameters
    
    Args:
        duration_seconds: Duration of each slot in seconds (default: 600 seconds/10 minutes)
        aircraft_per_slot: Maximum number of aircraft allowed per slot (default: 6)
        taxi_time_minutes: Deprecated; retained for backward compatibility but ignored.
        ctot_min_margin: Minutes before CTOT that are still allowed (default: 5)
        ctot_max_margin: Minutes after CTOT that are still allowed (default: 10)
        
    Returns:
        Updated slots DataFrame
    """
    global SLOT_DURATION_SECONDS, SLOT_SIZE, TAXI_TIME_MINUTES, CTOT_MIN_MARGIN, CTOT_MAX_MARGIN
    global SLOTS_PER_HOUR, TOTAL_SLOTS, slots
    
    # Update slot duration if specified
    if duration_seconds is not None:
        SLOT_DURATION_SECONDS = duration_seconds
        SLOTS_PER_HOUR = int(round(3600 / SLOT_DURATION_SECONDS))
        TOTAL_SLOTS = math.ceil(86400 / SLOT_DURATION_SECONDS)
        
    # Update aircraft per slot if specified
    if aircraft_per_slot is not None:
        SLOT_SIZE = aircraft_per_slot
    
    # Update taxi time if specified
    if taxi_time_minutes is not None:
        TAXI_TIME_MINUTES = None
    
    # Update CTOT margins if specified
    if ctot_min_margin is not None:
        CTOT_MIN_MARGIN = ctot_min_margin
    
    if ctot_max_margin is not None:
        CTOT_MAX_MARGIN = ctot_max_margin
        
    # Create new slots DataFrame with updated settings (seconds-based cumulative stepping)
    slots = pd.DataFrame(index=range(TOTAL_SLOTS))

    slots['slot_nr'] = slots.index
    slots['slot_size'] = SLOT_SIZE
    slots['count'] = 0
    slots['count_verschil'] = SLOT_SIZE
    slots['execute_push'] = 1
    slots['execute_moveback'] = 1

    start_secs = []
    end_secs = []
    center_secs = []
    for i in range(TOTAL_SLOTS):
        start_sec = (i * SLOT_DURATION_SECONDS) % 86400
        end_sec = (start_sec + SLOT_DURATION_SECONDS) % 86400
        center_sec = (start_sec + SLOT_DURATION_SECONDS / 2) % 86400
        start_secs.append(start_sec)
        end_secs.append(end_sec)
        center_secs.append(center_sec)

    def sec_to_time(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return datetime.time(h, m, s)

    slots['slot_start_sec'] = start_secs
    slots['slot_end_sec'] = end_secs
    slots['slot_center_sec'] = center_secs
    slots['slot_starttime'] = [sec_to_time(s) for s in start_secs]
    slots['slot_endtime'] = [sec_to_time(s) for s in end_secs]
    
    return slots
