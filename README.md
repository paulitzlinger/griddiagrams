# Griddiagrams

*Griddiagrams* is a Python package for working with grid diagrams of (fibered) knots, designed to find grid diagrams that admit a unique grid state whose Alexander grading is maximal and unique (more precisely, the Alexander grading even reaches an upper bound as explained HERE (link to be added soon)).

## Data Setup

This package uses the database from [knotinfo](https://knotinfo.org) which contains fibered prime knots up to crossing number 13.

### Manual download

1. Visit [knotinfo](https://knotinfo.org)
2. Use the advanced search with these settings:
   - Fibered: Y (Yes)
   - Select these columns to display:
     - Name
     - Fibered
     - Crossing Number
     - Genus-3D
     - Arc Index
     - Grid Notation
3. Download as CSV
4. Save as `knotinfo.csv` in the data directory `./data/knotinfo.csv`.

## Example Workflow

### Importing functions and loading data

```python
from griddiagrams.data import load_knot_data, get_grid_notation, get_genus
from griddiagrams.core import (
    gridnotation_to_gridlist,
    vlist,
    vlist_to_XO,
    gridstate_finder_commute,
)
from griddiagrams.plotting import plot_grid_diagram

load_knot_data()
```

### Selecting a knot and inspecting it

```python
knot = "12n_403"

genus = get_genus(knot)
grid_notation = get_grid_notation(knot)

print(f"Grid notation of knot {knot}:\n", grid_notation)
```

### Converting representation of grid diagram

```python
intermediate_format = gridnotation_to_gridlist(grid_notation)
vertlist_format = vlist(intermediate_format)

print(f"Vertlist format of knot {knot}:\n", vertlist_format)
```

### Plotting the original grid diagram

```python
X, O = vlist_to_XO(vertlist_format)

fig = plot_grid_diagram(X=X, O=O, knot_name=knot)
fig.savefig(f"original_diagram_{knot}.png", dpi=300, bbox_inches="tight")
```

### Searching for a *nice* grid diagram

```python
n = 50 # depth of search
result = gridstate_finder_commute(vertlist_format, n)
```

#### If succesful:

```python
if result:
    nice_grid = result["vlist"]
    winding_matrix = result["matrix"]
    perfect_state = result["gridstate"]
    alex = result["alexander-grading"]
```

Quick sanity check:

```python
if alex == genus:
    print("Sanity check passed: Alexander grading equals Seifert genus.")
else:
    print("ERROR: This should never happen! Check code for mistakes...")
```

##### Plotting the nice grid diagram

```python
X2, O2 = vlist_to_XO(nice_grid)

fig2 = plot_grid_diagram(
    X=X2,
    O=O2,
    matrix=winding_matrix,
    P=perfect_state,
    knot_name=knot,
)

fig2.savefig(f"nice_diagram_{knot}.png", dpi=300, bbox_inches="tight")
```

#### If unsuccesful:

```python
else:
    print(
        f"No nice grid diagram found at depth {n} using only commutations. "
        "Try increasing the depth or including stabilizations."
    )
```
