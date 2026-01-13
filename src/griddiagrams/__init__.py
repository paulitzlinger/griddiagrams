"""
knot-gridstate: A Python package for computing grid states and analyzing knot invariants.

This package provides tools for:
- Computing winding matrices and Alexander gradings for knot grid diagrams
- Finding unique perfect grid states through commutation moves
- Stabilization operations on grid diagrams
- Visualization of grid diagrams and winding matrices
- Parallel processing for large-scale computations

Basic usage:
    >>> import knot_gridstate as kg
    >>> kg.data.load_knot_data('path/to/knotinfo.csv')
    >>> 
    >>> # Get knot data
    >>> grid_notation = kg.data.get_grid_notation('3_1')
    >>> grid_list = kg.gridnotation_to_gridlist(grid_notation)
    >>> vert_list = kg.vlist(grid_list)
    >>> 
    >>> # Find grid state
    >>> result = kg.gridstate_finder_commute(vert_list, n=5)
    >>> if result:
    ...     print(f"Alexander grading: {result['alexander-grading']}")
"""

__version__ = "0.1.0"
__author__ = "Paul Leon Itzlinger"
__email__ = "paul.itzlinger@gmail.com"

# Import submodules
from . import core
from . import data
from . import plotting

# Import commonly used functions for convenience
from .core import (
    gridnotation_to_gridlist,
    vlist,
    hlist,
    w_matrix,
    knot_commute,
    vlist_to_XO,
    gridstate_finder_commute,
    gridstate_finder_stab,
)

from .plotting import plot_grid_diagram

# Define what is imported with "from knot_gridstate import *"
__all__ = [
    # Submodules
    "core",
    "data", 
    "plotting",
    # Core functions
    "gridnotation_to_gridlist",
    "vlist",
    "hlist",
    "w_matrix",
    "knot_commute",
    "vlist_to_XO",
    "gridstate_finder_commute",
    "gridstate_finder_stab",
    # Plotting function
    "plot_grid_diagram",
]


def check_data_loaded():
    """Check if knot data has been loaded."""
    if data._data is None:
        raise RuntimeError(
            "No knot data loaded. Please load data first:\n"
            "  knot_gridstate.data.load_knot_data('path/to/knotinfo.csv')\n"
            "Download the file from: https://knotinfo.math.indiana.edu/"
        )


def get_version():
    """Return the version string."""
    return __version__