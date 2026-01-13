from griddiagrams.data import load_knot_data, get_grid_notation, get_genus
from griddiagrams.core import gridnotation_to_gridlist, vlist, vlist_to_XO, gridstate_finder_commute
from griddiagrams.plotting import plot_grid_diagram

# load data
load_knot_data()

# choose fibred knot from data
knot = '12n_403'

# load Seifert genus of chosen knot
genus = get_genus(knot)

# original format of grid diagram of chosen knot
grid_notation_format = get_grid_notation(knot)
print(f"Grid notation of knot {knot}:\n", grid_notation_format)

# grid diagram in vertlist notation
intermediate_format = gridnotation_to_gridlist(grid_notation_format) # this is an intermediate format that is not needed elsewhere. I should combine the functions gridnotation_to_gridlist and vlist into one...
vertlist_format = vlist(intermediate_format)
print(f"Vertlist format of knot {knot}:\n", vertlist_format)

# plotting original grid diagram
X, O = vlist_to_XO(vertlist_format)
fig = plot_grid_diagram(X = X,
                        O = O, 
                        knot_name = knot)
figname = f"original_diagram_{knot}"
fig.savefig(f"{figname}.png", dpi = 300, bbox_inches = "tight")
print(f"Plot of original minimal grid diagram saved as {figname}.")

# finding nice grid diagram and corresponding perfect grid state
n = 50 # fix depth of search
result = gridstate_finder_commute(vertlist_format, n)

if result: # if search is succesful, extract grid diagram, winding matrix, grid state and Alexander grading of grid state
    nice_grid_diagram = result["vlist"]
    winding_matrix = result["matrix"]
    perfect_grid_state = result["gridstate"]
    Alex_grading = result["alexander-grading"]

    # quick sanity check if results are logical
    if Alex_grading == genus:
        print(f"Sanity check passed, Alexander grading of perfect grid state coincides with Seifert genus of knot {knot}!")
    else: 
        print("ERROR! This should never happen! Check code for mistakes!")

    # plotting nice grid diagram together with its perfect grid state and winding matrix
    X_2, O_2 = vlist_to_XO(nice_grid_diagram)
    fig_2 = plot_grid_diagram(X = X_2, 
                            O = O_2, 
                            matrix = winding_matrix, 
                            P = perfect_grid_state, 
                            knot_name = knot)
    figname_2 = f"nice_diagram_{knot}"
    fig_2.savefig(f"{figname_2}.png", dpi = 300, bbox_inches = "tight")
    print(f"Plot of perfect grid diagram saved as {figname_2}.")

else:
    print(f"No nice grid diagram for knot {knot} found with depth {n} using only commutations.\n Try higher depth or include stabilizations!")