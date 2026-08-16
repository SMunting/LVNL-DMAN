#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flight key generation module for deterministic flight identification.

This module provides functions to generate unique flight keys based on normalized
callsign, destination, and SOBT time, replacing the use of sfplid/acid combinations
throughout the flight scheduling system.
"""

from __future__ import annotations
import re
from typing import Any, Optional
import pandas as pd

# Regex constants for normalization
RE_SPACES = re.compile(r"\s+")
RE_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def normalize_callsign(raw: Any) -> Optional[str]:
    """
    Normalize callsign for flight key generation.
    
    Process: str(raw) → strip → collapse whitespace → uppercase → remove non-alphanumerics
    
    Args:
        raw: Raw callsign value (any type)
        
    Returns:
        Normalized callsign string or None if invalid
    """
    if pd.isna(raw):
        return None
    text = str(raw).strip().upper()
    text = RE_SPACES.sub(" ", text)
    text = RE_NON_ALNUM.sub("", text)
    return text or None


def normalize_dest(raw: Any) -> Optional[str]:
    """
    Normalize destination for flight key generation.
    
    Process: str(raw) → strip → collapse whitespace → uppercase → remove non-alphanumerics
    Keep letters/digits only (ICAO/IATA codes are alphanumeric)
    
    Args:
        raw: Raw destination value (any type)
        
    Returns:
        Normalized destination string or None if invalid
    """
    if pd.isna(raw):
        return None
    text = str(raw).strip().upper()
    text = RE_SPACES.sub(" ", text)
    text = RE_NON_ALNUM.sub("", text)
    return text or None


def extract_sobt_hms(raw: Any) -> Optional[str]:
    """
    Extract time component from SOBT for flight key generation.
    
    Handle both datetime strings and Unix timestamps.
    Parse to datetime with errors='coerce' and no timezone conversion.
    Extract time component only as HH:MM:SS format.
    
    Args:
        raw: Raw SOBT value (any type) - can be datetime string or Unix timestamp
        
    Returns:
        Time string in HH:MM:SS format or None if unparsable
    """
    if pd.isna(raw):
        return None
    
    # Try to parse as datetime first (for original string data)
    dt = pd.to_datetime(raw, errors="coerce")
    
    # If that fails and it looks like a Unix timestamp, try that approach
    if pd.isna(dt) and isinstance(raw, (int, float)) and raw > 1000000000:  # Unix timestamp range
        dt = pd.to_datetime(raw, unit='s', errors="coerce")
    
    if pd.isna(dt):
        return None
    
    return dt.strftime("%H:%M:%S")


def build_flight_key_columns(
    df: pd.DataFrame,
    acid_col: str = "acid",
    dest_col: str = "dest",
    sobt_col: str = "sobt",
) -> pd.DataFrame:
    """
    Build flight key columns for the given DataFrame.
    
    Adds normalized columns and flight_key column to the DataFrame.
    Does not reorder columns - use insert_flight_key_as_second_column() for that.
    
    Flight key formula: <normalized_acid>_<normalized_dest>_<sobt_time_hms>
    This creates the same key for different flight plans of the same flight.
    
    Args:
        df: DataFrame with flight data
        acid_col: Column name containing callsign/aircraft ID
        dest_col: Column name containing destination
        sobt_col: Column name containing scheduled off-block time
        
    Returns:
        DataFrame with added columns: normalized_callsign, normalized_dest, sobt_time_hms, flight_key
    """
    df = df.copy()

    # Build normalized components
    normalized_callsign = df[acid_col].map(normalize_callsign)
    normalized_dest = df[dest_col].map(normalize_dest)
    sobt_time_hms = df[sobt_col].map(extract_sobt_hms)

    # Add normalized columns to DataFrame
    df["normalized_callsign"] = normalized_callsign
    df["normalized_dest"] = normalized_dest
    df["sobt_time_hms"] = sobt_time_hms

    # Build flight key: if any component is missing → key becomes <NA>
    # Formula: callsign_dest_sobt_time_hms
    flight_key = (
        pd.Series(pd.NA, index=df.index)
        .mask(
            normalized_callsign.notna()
            & normalized_dest.notna()
            & sobt_time_hms.notna(),
            normalized_callsign.str.cat([normalized_dest, sobt_time_hms], sep="_"),
        )
    )

    df["flight_key"] = flight_key
    return df


def insert_flight_key_as_second_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder DataFrame to have flight_key as the second column.
    
    Args:
        df: DataFrame that may contain flight_key column
        
    Returns:
        DataFrame with flight_key as column index 1 (second column) if present
    """
    if "flight_key" not in df.columns:
        return df
    cols = list(df.columns)
    cols.remove("flight_key")
    # Insert after the first column
    cols = [cols[0], "flight_key"] + cols[1:]
    return df[cols]


def validate_flight_key_uniqueness(
    df: pd.DataFrame,
    key_col: str = "flight_key",
) -> None:
    """
    Validate flight key uniqueness.
    
    This function was previously used to print duplicate flight key warnings,
    but multiple entries with the same flight_key are normal (fuel updates, gate assignments, etc.).
    Flight plan updates are properly tracked via statistics instead.
    
    Args:
        df: DataFrame to check
        key_col: Column name containing flight keys
    """
    # Duplicate flight keys are normal - same flight can have multiple updates
    # Flight plan changes (different sfplids for same flight) are tracked in statistics
    pass


def log_missing_components(df: pd.DataFrame) -> None:
    """
    Log missing flight key components.
    
    Prints one line per row with pd.NA flight_key, indicating which components were missing.
    
    Args:
        df: DataFrame to check for missing components
    """
    if "flight_key" not in df.columns:
        return
        
    missing_mask = df["flight_key"].isna()
    if not missing_mask.any():
        return
    subset = df.loc[missing_mask, ["normalized_callsign", "normalized_dest", "sobt_time_hms"]]
    for idx, row in subset.iterrows():
        missing = []
        if pd.isna(row["normalized_callsign"]):
            missing.append("acid")
        if pd.isna(row["normalized_dest"]):
            missing.append("dest")
        if pd.isna(row["sobt_time_hms"]):
            missing.append("sobt_time_hms")
        print(f"[flight_key] MISSING components at index={idx}: {', '.join(missing)}")


def apply_flight_key_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the complete flight key pipeline to a DataFrame.
    
    This is a convenience function that applies all flight key operations:
    1. Build flight key columns
    2. Insert flight_key as second column
    3. Validate uniqueness
    4. Log missing components
    
    Args:
        df: DataFrame with flight data
        
    Returns:
        DataFrame with flight keys applied and positioned correctly
    """
    df = build_flight_key_columns(df)
    df = insert_flight_key_as_second_column(df)
    validate_flight_key_uniqueness(df)
    log_missing_components(df)
    return df
