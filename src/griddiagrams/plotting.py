"""
Visualization utilities for knot grid diagrams.

This module provides a function to plot grid diagrams of knots.
"""

import matplotlib.pyplot as plt

def plot_grid_diagram(X, O, matrix=None, P=None, knot_name=None):
    """
    Plot a grid diagram of a knot specified by two permutations X and O.
    
    Optionally displays a matrix of numbers at the bottom left of each grid square,
    and plots additional points specified by permutation P at grid intersections.
    
    For the permutation P, a marker is placed at the intersection of the j-th row 
    and i-th column where P[j] = i.

    Parameters
    ----------
    X : List[int]
        List of integers specifying the positions of X's in each column.
    O : List[int]
        List of integers specifying the positions of O's in each column.
    matrix : List[List[int]] or np.ndarray, optional
        A 2D array representing a matrix of numbers to be plotted at the grid squares.
    P : List[int], optional
        List of integers specifying a permutation; markers are placed where P[j] = i.
    knot_name : str, optional
        Name of the knot to be displayed at the top-left of the plot.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    """
    n = len(X)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Adjust axis limits to add equal margins on all sides
    margin = 0.5
    ax.set_xlim(-margin, n - 1 + margin)
    ax.set_ylim(-margin, n - 1 + margin)

    # Set ticks at each grid line
    ax.set_xticks(range(n+1))
    ax.set_yticks(range(n+1))

    # Remove tick labels but keep ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Enable grid lines
    ax.grid(True)

    # Hide all spines to remove black lines around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Place the knot name if provided
    if knot_name is not None:
        # Place knot name at top-left corner (just above the top-left corner of the grid)
        ax.text(-margin + 0.05, n - 1 + margin - 0.05, knot_name, 
                ha='left', va='top', fontsize=14, color='black')

    # Place X's and O's
    for i in range(n):
        x_pos = i + 0.5
        x_y = X[i] + 0.5
        o_y = O[i] + 0.5
        ax.text(x_pos, x_y, 'X', ha='center', va='center', fontsize=16, color='red')
        ax.text(x_pos, o_y, 'O', ha='center', va='center', fontsize=16, color='blue')

    # Draw vertical segments from O to X in each column
    for i in range(n):
        x = i + 0.5
        y0 = min(O[i], X[i]) + 0.5
        y1 = max(O[i], X[i]) + 0.5
        ax.plot([x, x], [y0, y1], color='black', linewidth=2)

    # For each row, find horizontal segments
    for row in range(n):
        cols_with_O = [i for i in range(n) if O[i] == row]
        cols_with_X = [i for i in range(n) if X[i] == row]
        for start_col in cols_with_O:
            # Find the next X in this row, wrapping around
            sorted_cols = sorted(cols_with_X + [start_col])
            idx = sorted_cols.index(start_col)
            end_col = sorted_cols[(idx + 1) % len(sorted_cols)]
            x0 = min(start_col, end_col) + 0.5
            x1 = max(start_col, end_col) + 0.5
            y = row + 0.5
            ax.plot([x0, x1], [y, y], color='black', linewidth=2)

    # Plot matrix entries at the bottom left of each grid square
    if matrix is not None:
        for i in range(n):
            for j in range(n):
                x_pos = j
                y_pos = n - 1 - i  # Invert y-index to match plotting coordinate system
                value = matrix[i][j]
                # Adjust positions slightly to place text at the bottom left
                ax.text(x_pos + 0.05, y_pos + 0.05, str(value), ha='left', va='bottom',
                        fontsize=10, color='grey')  # Make the matrix numbers grey

    # Plot additional points at grid intersections based on permutation P
    if P is not None:
        for j in range(n):
            i = P[j]
            x = i
            y = n - 1 - j  # Invert y-coordinate to match plotting
            # Plot a point at the grid intersection (x, y)
            ax.plot(x, y, marker='o', color='purple', markersize=8, clip_on=False)

    # Remove axes to prevent any lines around the plot
    ax.axis('off')

    plt.tight_layout()
    return fig