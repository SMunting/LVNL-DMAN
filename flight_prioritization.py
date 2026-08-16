#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flight prioritization and selection module.

This module handles the selection of flights based on priority scoring
for slot allocation optimization.
"""

import pandas as pd
import numpy as np


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
    
    # Initialize priority columns if they don't exist
    for col in priority_columns:
        if col not in df_sorted.columns:
            df_sorted[col] = 999999  # Default high value for non-existent priority columns
        else:
            df_sorted[col] = np.nan_to_num(df_sorted[col].values, nan=999999)
    
    # Filter to only include priority columns that exist in the DataFrame
    existing_priority_columns = [col for col in priority_columns if col in df_sorted.columns]
    
    # Sort all at once with stable sort
    df_sorted = df_sorted.sort_values(
        existing_priority_columns,
        ascending=[True] * len(existing_priority_columns),
        na_position='last',
        kind='stable'  # Use stable sort to maintain order for equal values
    )
    
    # Simple slice to get keep and push flights
    keep_df = df_sorted.iloc[:slot_size].copy() if len(df_sorted) > 0 else pd.DataFrame()
    push_df = df_sorted.iloc[slot_size:].copy() if len(df_sorted) > slot_size else pd.DataFrame()
    
    return keep_df, push_df
