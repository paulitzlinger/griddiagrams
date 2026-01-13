"""
Data loading utilities for knot information.

This module provides functions to load and query knot data from the KnotInfo database.
The data file (knotinfo.csv) can be obtained from:
https://knotinfo.math.indiana.edu/
"""

import pandas as pd
import ast
from pathlib import Path
from typing import List

# Module-level data storage
_data = None
_data_path = None

def load_knot_data() -> None:
    """
    Load and clean knot data from ./data/knotinfo.csv
    """
    global _data, _data_path

    _data_path = Path("data/knotinfo.csv")

    if not _data_path.exists():
        raise FileNotFoundError(
            "Could not find knotinfo.csv at ./data/knotinfo.csv"
        )

    try:
        _data = pd.read_csv(_data_path)
        _data["Grid Notation"] = _data["Grid Notation"].apply(_clean_grid_notation)
        print(f"Loaded {len(_data)} knots from {_data_path}")
    except Exception as e:
        raise ValueError(f"Error loading data from {_data_path}: {e}")


def _clean_grid_notation(entry: str) -> List[List[int]]:
    """Convert grid notation string to Python list."""

    if pd.isna(entry):
        return []
    
    # Replace semicolons with commas
    entry = entry.replace(';', ',')
    
    try:
        return ast.literal_eval(entry)
    except:
        return []


def get_grid_notation(knot_name: str) -> List[List[int]]:
    """
    Get grid notation for a knot, the format in which grid diagrams are specified in the source database.
    
    Parameters
    ----------
    knot_name : str
        Name of the knot (e.g., '3_1', '4_1').
    
    Returns
    -------
    List[List[int]]
        Grid notation as list of coordinate pairs.
        
    Raises
    ------
    RuntimeError
        If data hasn't been loaded yet.
    ValueError
        If knot not found.
    """

    if _data is None:
        raise RuntimeError("No data loaded. Call load_knot_data() first.")
    
    result = _data.loc[_data['Name'] == knot_name, 'Grid Notation']
    
    if result.empty:
        raise ValueError(f"Knot '{knot_name}' not found")
    
    return result.iloc[0]


def get_genus(knot_name: str) -> int:
    """
    Get 3-genus for a knot.
    
    Parameters
    ----------
    knot_name : str
        Name of the knot.
    
    Returns
    -------
    int
        3-genus of the knot.
    """

    if _data is None:
        raise RuntimeError("No data loaded. Call load_knot_data() first.")
    
    result = _data.loc[_data['Name'] == knot_name, 'Genus-3D']
    
    if result.empty:
        raise ValueError(f"Knot '{knot_name}' not found")
    
    # Handle string values like "0" or missing data
    value = result.iloc[0]
    if pd.isna(value) or value == '':
        return 0
    
    return int(value)


def get_arc_index(knot_name: str) -> int:
    """
    Get arc index for a knot.
    
    Parameters
    ----------
    knot_name : str
        Name of the knot.
    
    Returns
    -------
    int
        Arc index of the knot.
    """

    if _data is None:
        raise RuntimeError("No data loaded. Call load_knot_data() first.")
    
    result = _data.loc[_data['Name'] == knot_name, 'Arc Index']
    
    if result.empty:
        raise ValueError(f"Knot '{knot_name}' not found")
    
    return int(result.iloc[0])


def filter_by_arc_index(min_arc: int, max_arc: int) -> List[str]:
    """
    Get all fibered knots with arc index in given range.
    
    Parameters
    ----------
    min_arc : int
        Minimum arc index (inclusive).
    max_arc : int  
        Maximum arc index (inclusive).
    
    Returns
    -------
    List[str]
        List of knot names.
    """

    if _data is None:
        raise RuntimeError("No data loaded. Call load_knot_data() first.")
    
    filtered = _data[
        (_data['Arc Index'] >= min_arc) & 
        (_data['Arc Index'] <= max_arc)
    ]
    
    return filtered['Name'].tolist()


def get_all_knot_names() -> List[str]:
    """
    Get list of all knot names in the dataset.
    
    Returns
    -------
    List[str]
        List of all knot names.
    """

    if _data is None:
        raise RuntimeError("No data loaded. Call load_knot_data() first.")
    
    return _data['Name'].tolist()