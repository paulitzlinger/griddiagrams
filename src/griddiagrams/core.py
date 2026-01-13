"""
This module provides the fundamental algorithms for working with grid diagrams,
including conversions between representations, commutation moves, winding matrices,
and grid state finding algorithms.
"""

from typing import List, Tuple, Optional, Dict, Set, Union
import numpy as np
from numpy.typing import NDArray


# Type aliases
GridNotation = List[List[int]]
GridList = List[int]
VertList = List[Tuple[int, int]]
HorzList = List[Tuple[int, int]]
Permutation = List[int]
WindingMatrix = NDArray[np.int_]


def gridnotation_to_gridlist(gridnotation: GridNotation) -> GridList:
    """
    Convert gridnotation to grid list representation.
    
    Parameters
    ----------
    gridnotation : List[List[int]]
        Grid notation specifying a grid diagram
        (Grid notation is the representation of grid diagrams used in the source database from knotinfo)

    Returns
    -------
    List[int]
        Grid list representation, an intermediate format
    """

    if not gridnotation:
        raise ValueError("Grid notation cannot be empty")
    
    temp = [gridnotation[0][1]]
    current_tuple = gridnotation[0]

    while len(temp) < len(gridnotation):
        if len(temp) % 2 == 1:
            # Look for matching first coordinate
            for segment in gridnotation:
                if segment[1] == temp[-1] and segment[0] != current_tuple[0]:
                    temp.append(segment[0])
                    current_tuple = segment
                    break
            else:
                raise ValueError("Invalid grid notation: no matching segment found")
        else:
            # Look for matching second coordinate
            for segment in gridnotation:
                if segment[0] == temp[-1] and segment[1] != current_tuple[1]:
                    temp.append(segment[1])
                    current_tuple = segment
                    break
            else:
                raise ValueError("Invalid grid notation: no matching segment found")

    # Convert to 0-indexed
    return [x - 1 for x in temp]


def hlist(gridlist: GridList) -> HorzList:
    """
    Convert grid list to horizontal segment list.
    
    Parameters
    ----------
    gridlist : List[int]
        Grid list representation.
    
    Returns
    -------
    List[Tuple[int, int]]
        List of tuples representing oriented horizontal segments.    
    """

    extended_grid = gridlist.copy()
    extended_grid.extend([gridlist[0], gridlist[1]])
    
    n = len(extended_grid)
    x = n + 1
    hsegments: List[Optional[Tuple[int, int]]] = [None] * (2 * n + 1)
    hsegments[x] = (extended_grid[1], extended_grid[3])

    for i in range(3, len(extended_grid) - 2, 2):
        x = x + extended_grid[i + 1] - extended_grid[i - 1]
        if 0 <= x < len(hsegments):
            hsegments[x] = (extended_grid[i], extended_grid[i + 2])
        else:
            raise IndexError("Calculated index is out of bounds!")

    # Filter out None values and return
    return [seg for seg in hsegments if seg is not None]


def vlist(gridlist: GridList) -> VertList:
    """
    Convert grid list to vertical segment list.
    
    Parameters
    ----------
    gridlist : List[int]
        Grid list representation.
    
    Returns
    -------
    List[Tuple[int, int]]
        List of tuples representing oriented vertical segments.
    """
    
    extended_grid = gridlist.copy()
    extended_grid.append(gridlist[0])
    
    n = len(extended_grid)
    x = n + 1
    vsegments: List[Optional[Tuple[int, int]]] = [None] * (2 * n + 1)
    vsegments[x] = (extended_grid[0], extended_grid[2])
    
    for i in range(2, n - 2, 2):
        x = x + extended_grid[i + 1] - extended_grid[i - 1]
        if 0 <= x < len(vsegments):
            vsegments[x] = (extended_grid[i], extended_grid[i + 2])
        else:
            raise IndexError("Calculated index is out of bounds!")
        
    return [seg for seg in vsegments if seg is not None]


def v_to_h(vertlist: VertList) -> HorzList:
    """
    Convert vertical segment list to horizontal segment list.
    
    Parameters
    ----------
    vertlist : List[Tuple[int, int]]
        Vertical segment list.
    
    Returns
    -------
    List[Tuple[int, int]]
        Horizontal segment list.
    """

    n = len(vertlist)
    horzlist = []
    
    for i in range(n):
        segment_indices = []
        for j in range(n):
            if vertlist[j][0] == i:
                segment_indices.append(j)
            elif vertlist[j][1] == i:
                segment_indices.insert(0, j)
        
        if len(segment_indices) != 2:
            raise ValueError(f"Invalid vertical list: row {i} has {len(segment_indices)} segments")
        
        horzlist.append(tuple(segment_indices))
    
    return horzlist


def h_to_v(horzlist: HorzList) -> VertList:
    """
    Convert horizontal segment list to vertical segment list.
    
    Parameters
    ----------
    horzlist : List[Tuple[int, int]]
        Horizontal segment list.
    
    Returns
    -------
    List[Tuple[int, int]]
        Vertical segment list.
    """

    n = len(horzlist)
    vertlist = []
    
    for i in range(n):
        segment_indices = []
        for j in range(n):
            if horzlist[j][0] == i:
                segment_indices.append(j)
            elif horzlist[j][1] == i:
                segment_indices.insert(0, j)
        
        if len(segment_indices) != 2:
            raise ValueError(f"Invalid horizontal list: column {i} has {len(segment_indices)} segments")
            
        vertlist.append(tuple(segment_indices))
    
    return vertlist


def can_commute(t1: Tuple[int, int], t2: Tuple[int, int]) -> bool:
    """
    Check if two segments can be commuted.
    
    Parameters
    ----------
    t1, t2 : Tuple[int, int]
        Two segments to check.
    
    Returns
    -------
    bool
        True if segments can be commuted, False otherwise.
    """

    a, b = t1
    c, d = t2
    max1, min1 = max(a, b), min(a, b)
    max2, min2 = max(c, d), min(c, d)
    
    return (
        (max1 <= min2) or 
        (min1 >= max2) or 
        (max1 >= max2 and min1 <= min2) or 
        (max2 >= max1 and min2 <= min1)
    )


def c_move(input_list: Union[VertList, HorzList]) -> List[Union[VertList, HorzList]]:
    """
    If input is vertical segment list, generates all column commutation moves.
    If input is horizontal segment list, generates all row commutation moves.
    
    Parameters
    ----------
    input_list : List[Tuple[int, int]]
        Vertical or horizontal segment list.
    
    Returns
    -------
    List[List[Tuple[int, int]]]
        List of all possible configurations after one commutation.
    """

    result = []
    seen: Set[Tuple[Tuple[int, int], ...]] = set()
    n = len(input_list)

    def add_to_result(lst: List[Tuple[int, int]]) -> None:
        """Add configuration to result if not seen before."""
        tpl = tuple(lst)
        if tpl not in seen:
            seen.add(tpl)
            result.append(lst)

    # Try adjacent commutations
    for i in range(n - 1):
        if can_commute(input_list[i], input_list[i + 1]):
            swapped_list = input_list.copy()
            swapped_list[i], swapped_list[i + 1] = swapped_list[i + 1], swapped_list[i]
            add_to_result(swapped_list)

    # Try wrap-around commutation
    if can_commute(input_list[0], input_list[-1]):
        swapped_list = input_list.copy()
        swapped_list[0], swapped_list[-1] = swapped_list[-1], swapped_list[0]
        add_to_result(swapped_list)

    return result


def knot_commute(vertlist: VertList) -> List[VertList]:
    """
    Generate all possible knot diagrams obtainable by one commutation.
    
    Parameters
    ----------
    vertlist : List[Tuple[int, int]]
        Vertical segment list.
    
    Returns
    -------
    List[List[Tuple[int, int]]]
        List of all possible vertical lists after one commutation.
        
    Notes
    -----
    This considers both vertical and horizontal commutations.
    """

    # Get vertical commutations
    v_commutations = c_move(vertlist)
    
    # Get horizontal commutations and convert back to vertical
    h_list = v_to_h(vertlist)
    h_commutations = c_move(h_list)
    h_to_v_commutations = [h_to_v(horzlist) for horzlist in h_commutations]
    
    # Combine and remove duplicates
    c_set = set(tuple(lst) for lst in v_commutations + h_to_v_commutations)
    c_list = [list(tpl) for tpl in c_set]
    
    return c_list


def w_matrix(vertlist: VertList) -> WindingMatrix:
    """
    Calculate the winding matrix for a knot grid diagram.
    
    Parameters
    ----------
    vertlist : List[Tuple[int, int]]
        Vertical segment list.
    
    Returns
    -------
    np.ndarray
        Winding matrix of grid diagram represneted by Vertical segment list.
        
    Notes
    -----
    The winding number increases by 1 when crossing an upward segment
    and decreases by 1 when crossing a downward segment.
    """

    size = len(vertlist)
    result = []

    for i in range(size):
        row = [0]  # Start with winding number 0 for first column

        for j in range(size - 1):
            tail, head = vertlist[j]
            
            if tail <= i < head:  # Upward segment
                row.append(row[-1] + 1)
            elif head <= i < tail:  # Downward segment
                row.append(row[-1] - 1)
            else:  # No crossing
                row.append(row[-1])

        result.append(row)

    return np.array(result, dtype=np.int32)


def h_type_0_permutation(matrix: WindingMatrix) -> Union[Permutation, str]:
    """
    Find a unique horizontal type-0 permutation (i.e. a unique row-perfect grid state) if it exists.
    
    Parameters
    ----------
    matrix : np.ndarray
        Winding matrix.
    
    Returns
    -------
    List[int] or str
        Permutation if unique one exists, error message otherwise.
    """

    n = len(matrix)
    
    # Find column indices of minimum values in each row
    min_indices = [
        set(i for i, val in enumerate(row) if val == min(row))
        for row in matrix
    ]
    
    result: List[Optional[int]] = [None] * n
    
    while any(s is not None for s in min_indices):
        # Find a singleton set
        singleton_index = next(
            (j for j, s in enumerate(min_indices) if s is not None and len(s) == 1),
            None
        )
        
        if singleton_index is None:
            return "No unique h-type-0 permutation exists."
        
        # Extract the single element
        singleton_set = min_indices[singleton_index]
        x = next(iter(singleton_set))
        result[singleton_index] = x
        min_indices[singleton_index] = None
        
        # Remove x from all other sets
        for s in min_indices:
            if s is not None:
                s.discard(x)
        
        # Check for empty sets
        if any(s is not None and len(s) == 0 for s in min_indices):
            return "No unique h-type-0 permutation exists."
    
    # Ensure all positions are filled
    if None in result:
        return "This should never happen, check code if it does."
    
    return [x for x in result if x is not None]


def v_type_0_permutation(matrix: WindingMatrix) -> Union[Permutation, str]:
    """
    Find a unique vertical type-0 permutation (i.e. a unique column-perfect grid state) if it exists.
    
    Parameters
    ----------
    matrix : np.ndarray
        Winding matrix.
    
    Returns
    -------
    List[int] or str
        Permutation if unique one exists, error message otherwise.
    """

    n = len(matrix)
    
    # Find row indices of minimum values in each column
    min_indices = [
        set(i for i in range(n) if matrix[i][col] == min(matrix[j][col] for j in range(n)))
        for col in range(n)
    ]
    
    result: List[Optional[int]] = [None] * n
    
    while any(s is not None for s in min_indices):
        # Find a singleton set
        singleton_index = next(
            (j for j, s in enumerate(min_indices) if s is not None and len(s) == 1),
            None
        )
        
        if singleton_index is None:
            return "No unique v-type-0 permutation exists."
        
        # Extract the single element
        singleton_set = min_indices[singleton_index]
        x = next(iter(singleton_set))
        result[singleton_index] = x
        min_indices[singleton_index] = None
        
        # Remove x from all other sets
        for s in min_indices:
            if s is not None:
                s.discard(x)
        
        # Check for empty sets
        if any(s is not None and len(s) == 0 for s in min_indices):
            return "No unique v-type-0 permutation exists."
    
    # Ensure all positions are filled
    if None in result:
        return "This should never happen, check code if it does."
    
    return [x for x in result if x is not None]


def rev(input_list: Union[VertList, HorzList]) -> Union[VertList, HorzList]:
    """
    Reverse the orientation of a knot diagram.
    
    Parameters
    ----------
    input_list : List[Tuple[int, int]]
        Vertical or horizontal segment list.
    
    Returns
    -------
    List[Tuple[int, int]]
        Segment list with reversed orientation.
    """

    return [(seg[1], seg[0]) for seg in input_list]


def a_grading(vertlist: VertList, matrix: WindingMatrix, permutation: Permutation) -> int:
    """
    Calculate the Alexander grading for a grid state.
    
    Parameters
    ----------
    vertlist : List[Tuple[int, int]]
        Vertical segment list.
    matrix : np.ndarray
        Winding matrix.
    permutation : List[int]
        Permutation representing the grid state.
    
    Returns
    -------
    int
        Alexander grading of the grid state.
    """

    n = len(vertlist)
    m = (n - 1) / 2
    
    a_sum = 0
    
    for tpl in vertlist:
        col = vertlist.index(tpl)
        upper_row = min(tpl)
        lower_row = max(tpl)

        if upper_row == 0 and col != vertlist.index(vertlist[-1]):
            upper_sum = matrix[0][col] + matrix[0][col + 1]
            
        elif upper_row != 0 and col != vertlist.index(vertlist[-1]):
            upper_sum = matrix[upper_row-1][col] + matrix[upper_row-1][col+1] + matrix[upper_row][col] + matrix[upper_row][col + 1]
        
        elif upper_row == 0 and col == vertlist.index(vertlist[-1]):
            upper_sum = matrix[0][col]
            
        elif upper_row != 0 and col == vertlist.index(vertlist[-1]):
            upper_sum = matrix[upper_row-1][col] + matrix[upper_row][col]
            
        
        if col != vertlist.index(vertlist[-1]):
            lower_sum = matrix[lower_row-1][col] + matrix[lower_row-1][col+1] + matrix[lower_row][col] + matrix[lower_row][col + 1]
        
        elif col == vertlist.index(vertlist[-1]):
            lower_sum = matrix[lower_row-1][col] + matrix[lower_row][col]
            
        a_sum = a_sum + upper_sum + lower_sum
        
    
    a_sum = a_sum/8
    w_sum = 0

    for i in permutation:
        j = permutation.index(i)
        w_sum = w_sum + matrix[j][i]
            
    return int(-w_sum + a_sum - m)


def vperm_to_hperm(vperm: Permutation) -> Permutation:
    """
    Convert vertical permutation to horizontal permutation.
    
    Parameters
    ----------
    vperm : List[int]
        Vertical permutation.

    Note
    -----
    This is only used to have all found grid states in the same format
    
    Returns
    -------
    List[int]
        Horizontal permutation.
    """

    # Create (value, index) pairs and sort by value
    indexed_perm = [(vperm[i], i) for i in range(len(vperm))]
    sorted_perm = sorted(indexed_perm, key=lambda x: x[0])
    
    return [index for _, index in sorted_perm]


def try_permutations(vertlist: VertList) -> Optional[Tuple[VertList, str, Permutation, WindingMatrix, int]]:
    """
    Try to find a unique perfect grid state for a diagram.
    
    Parameters
    ----------
    vertlist : List[Tuple[int, int]]
        Vertical segment list.
    
    Returns
    -------
    Optional[Tuple]
        If found: (vertlist, type, permutation, matrix, alexander_grading)
        If not found: None
        
    Notes
    -----
    This function tries both the original and reversed orientation,
    checking for both horizontal and vertical perfect grid states.
    """

    # Try original orientation
    matrix = w_matrix(vertlist)
    rowsum = np.sum(np.min(matrix, axis=1))
    colsum = np.sum(np.min(matrix, axis=0))
    
    # Check based on row/column sums
    if rowsum > colsum:
        # Only horizontal perfect grid state possible
        h_perm = h_type_0_permutation(matrix)
        if not isinstance(h_perm, str):
            return (vertlist, "h_type_0", h_perm, matrix, a_grading(vertlist, matrix, h_perm))
    
    elif colsum > rowsum:
        # Only vertical perfect grid state possible
        v_perm = v_type_0_permutation(matrix)
        if not isinstance(v_perm, str):
            h_perm = vperm_to_hperm(v_perm)
            return (vertlist, "v_type_0", h_perm, matrix, a_grading(vertlist, matrix, h_perm))
    
    else:  # rowsum == colsum
        # Try both types
        h_perm = h_type_0_permutation(matrix)
        if not isinstance(h_perm, str):
            return (vertlist, "h_type_0", h_perm, matrix, a_grading(vertlist, matrix, h_perm))
        
        v_perm = v_type_0_permutation(matrix)
        if not isinstance(v_perm, str):
            h_perm = vperm_to_hperm(v_perm)
            return (vertlist, "v_type_0", h_perm, matrix, a_grading(vertlist, matrix, h_perm))
    
    # Try reversed orientation
    vertlist_rev = rev(vertlist)
    matrix_rev = w_matrix(vertlist_rev)
    rowsum_rev = np.sum(np.min(matrix_rev, axis=1))
    colsum_rev = np.sum(np.min(matrix_rev, axis=0))
    
    if rowsum_rev > colsum_rev:
        h_perm = h_type_0_permutation(matrix_rev)
        if not isinstance(h_perm, str):
            return (vertlist_rev, "h_type_0", h_perm, matrix_rev, 
                   a_grading(vertlist_rev, matrix_rev, h_perm))
    
    elif colsum_rev > rowsum_rev:
        v_perm = v_type_0_permutation(matrix_rev)
        if not isinstance(v_perm, str):
            h_perm = vperm_to_hperm(v_perm)
            return (vertlist_rev, "v_type_0_rev", h_perm, matrix_rev,
                   a_grading(vertlist_rev, matrix_rev, h_perm))
    
    else:  # rowsum_rev == colsum_rev
        h_perm = h_type_0_permutation(matrix_rev)
        if not isinstance(h_perm, str):
            return (vertlist_rev, "h_type_0", h_perm, matrix_rev,
                   a_grading(vertlist_rev, matrix_rev, h_perm))
        
        v_perm = v_type_0_permutation(matrix_rev)
        if not isinstance(v_perm, str):
            h_perm = vperm_to_hperm(v_perm)
            return (vertlist_rev, "v_type_0_rev", h_perm, matrix_rev,
                   a_grading(vertlist_rev, matrix_rev, h_perm))
    
    return None


def vlist_to_XO(vertlist: VertList) -> Tuple[List[int], List[int]]:
    """
    Convert vertical list to X and O permutations.

    Note
    ----
    This is only used for plotting
    
    Parameters
    ----------
    vertlist : List[Tuple[int, int]]
        Vertical segment list.
    
    Returns
    -------
    Tuple[List[int], List[int]]
        X and O permutations.
    """

    n = len(vertlist) - 1
    tempx = [tpl[0] for tpl in vertlist]
    tempo = [tpl[1] for tpl in vertlist]
    
    # Invert for plotting coordinates
    X = [n - x for x in tempx]
    O = [n - x for x in tempo]
    
    return X, O


def gridstate_finder_commute(vertlist: VertList, n: int) -> Optional[Dict]:
    """
    Find unique perfect grid states through commutation moves.
    
    Parameters
    ----------
    vertlist : List[Tuple[int, int]]
        Initial vertical segment list.
    n : int
        Maximum number of commutation iterations.
    
    Returns
    -------
    Optional[Dict]
        Dictionary containing grid state information if found, None otherwise.
        
    Notes
    -----
    We use a breadth-first search approach to explore commutation space.
    """

    # Check initial state
    perm_result = try_permutations(vertlist)
    if perm_result:
        vlist_out, perm_type, perm, matrix, alex = perm_result
        return {
            "vlist": vlist_out,
            "matrix": matrix.tolist(),  # Convert to list for JSON serialization
            "type": perm_type,
            "gridstate": perm,
            "alexander-grading": alex
        }
    
    # BFS through commutation space
    current_states = {tuple(vertlist)}
    visited_states = set(current_states)
    
    for iteration in range(n):
        new_states = set()
        
        for state in current_states:
            commuted_states = knot_commute(list(state))
            
            for commuted in commuted_states:
                commuted_tuple = tuple(commuted)
                
                if commuted_tuple not in visited_states:
                    visited_states.add(commuted_tuple)
                    
                    # Try to find perfect grid state
                    perm_result = try_permutations(commuted)
                    if perm_result:
                        vlist_out, perm_type, perm, matrix, alex = perm_result
                        return {
                            "vlist": vlist_out,
                            "matrix": matrix.tolist(),
                            "type": perm_type,
                            "gridstate": perm,
                            "alexander-grading": alex
                        }
                    
                    new_states.add(commuted_tuple)
        
        current_states = new_states
        
        if not current_states:
            break  # No new states to explore
    
    return None


def x_nw(vertlist: VertList, loc: Tuple[int, int]) -> VertList:
    """
    Perform an X-NW stabilization at specified segment.
    
    Parameters
    ----------
    vertlist : List[Tuple[int, int]]
        Vertical segment list.
    loc : Tuple[int, int]
        Segment where to perform stabilization.
    
    Returns
    -------
    List[Tuple[int, int]]
        Stabilized vertical list.
    """

    if loc not in vertlist:
        raise ValueError(f"Segment {loc} not in vertical list")
    
    k = vertlist.index(loc)
    temp = [list(tpl) for tpl in vertlist]
    
    # Shift indices greater than loc[0]
    for segment in temp:
        for j in range(len(segment)):
            if segment[j] > loc[0]:
                segment[j] += 1
    
    # Insert new segment and modify existing one
    temp.insert(k + 1, [loc[0], loc[0] + 1])
    temp[k][0] = loc[0] + 1
    
    return [tuple(segment) for segment in temp]


def gridstate_finder_stab(vertlist: VertList, n: int) -> Optional[Dict]:
    """
    Find unique perfect grid states by trying stabilizations.
    
    Maintains a global visited set across all stabilization attempts
    to avoid recomputing states.
    
    Parameters
    ----------
    vertlist : List[Tuple[int, int]]
        Initial vertical segment list.
    n : int
        Maximum number of commutation iterations after stabilization.
    
    Returns
    -------
    Optional[Dict]
        Dictionary containing grid state information if found, None otherwise.
    """

    # Global visited set for all stabilization attempts
    global_visited = {tuple(vertlist)}
    
    # Try each stabilization
    for segment in vertlist:
        try:
            stab_vertlist = x_nw(vertlist, segment)
            stab_tuple = tuple(stab_vertlist)
            
            # Skip if we've seen this stabilized state before
            if stab_tuple in global_visited:
                continue
                
            # Custom BFS that uses the global visited set
            result = _gridstate_finder_commute_with_visited(
                stab_vertlist, n, global_visited
            )
            
            if result:
                # Add stabilization info
                result["stabilizations"] = 1
                return result
                
        except Exception:
            continue
    
    return None


def _gridstate_finder_commute_with_visited(
    vertlist: VertList, 
    n: int, 
    global_visited: Set[Tuple[Tuple[int, int], ...]]
) -> Optional[Dict]:
    """
    Helper function: gridstate_finder_commute that respects a global visited set.
    """
    
    # Check initial state
    perm_result = try_permutations(vertlist)
    if perm_result:
        vlist_out, perm_type, perm, matrix, alex = perm_result
        return {
            "vlist": vlist_out,
            "matrix": matrix.tolist(),
            "type": perm_type,
            "gridstate": perm,
            "alexander-grading": alex
        }
    
    # BFS with shared visited set
    current_states = {tuple(vertlist)}
    
    for iteration in range(n):
        new_states = set()
        
        for state in current_states:
            commuted_states = knot_commute(list(state))
            
            for commuted in commuted_states:
                commuted_tuple = tuple(commuted)
                
                if commuted_tuple not in global_visited:
                    global_visited.add(commuted_tuple)
                    
                    perm_result = try_permutations(commuted)
                    if perm_result:
                        vlist_out, perm_type, perm, matrix, alex = perm_result
                        return {
                            "vlist": vlist_out,
                            "matrix": matrix.tolist(),
                            "type": perm_type,
                            "gridstate": perm,
                            "alexander-grading": alex
                        }
                    
                    new_states.add(commuted_tuple)
        
        current_states = new_states
        
        if not current_states:
            break
    
    return None