"""
Utility functions for fraud detection project.

Helper functions used across multiple modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath, drop_index_cols=True):
    """
    Load CSV file and optionally drop index columns.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    drop_index_cols : bool
        Whether to drop unnamed index columns

    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    df = pd.read_csv(filepath)

    if drop_index_cols:
        index_cols = [col for col in df.columns if 'Unnamed' in col]
        if index_cols:
            df = df.drop(columns=index_cols)

    return df


def print_data_summary(df, name="Dataset"):
    """
    Print summary statistics for a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to summarize
    name : str
        Name for the dataset
    """
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    if 'class' in df.columns:
        print(f"\nFraud Distribution:")
        print(df['class'].value_counts())
        print(f"\nFraud Rate: {df['class'].mean():.4%}")

    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("No missing values")


def ensure_dir(directory):
    """
    Ensure directory exists, create if it doesn't.

    Parameters
    ----------
    directory : str or Path
        Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def format_metrics(metrics, precision=4):
    """
    Format metrics dictionary for display.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to values
    precision : int
        Number of decimal places

    Returns
    -------
    str
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"{key:15s}: {value:.{precision}f}")
        else:
            lines.append(f"{key:15s}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Utility Functions")
    print("\nAvailable functions:")
    print("- load_data(): Load CSV with automatic index column removal")
    print("- print_data_summary(): Print dataset statistics")
    print("- ensure_dir(): Create directory if it doesn't exist")
    print("- format_metrics(): Format metrics for display")
