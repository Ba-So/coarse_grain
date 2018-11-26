# all data manipulations necessary for computations
# Here we'll define the hex_areas and find neighbors
# MISSING:
#   * Routine to assign hex area neighbors to central hexes
#   * Routine to cast hexes into objects.
#       -> parallelization through using as many objects as there are
#       processors? with one global dict, giving information on which
#       processor specific
#       hexes are to allow messages to be sent between them.
#       -> read on parallelization?

'''
    This Routine contains operations on the data fields.
    Contains:
        get_members
        get_members_idx
'''

# adapted the whole thing for use of globals
import numpy as np
import global_vars as gv


# changed according to new array structure - check
def get_members(where, i, variables):
    '''gets members of a hex_area'''
    # functional and used.
    a_nei_idx = gv.globals_dict['grid_nfo']['area_neighbor_idx'][i, :]
    out = {}
    idxcs = a_nei_idx[np.where(a_nei_idx > -1)[0],]

    out = get_members_idx(where, idxcs, variables)

    return out

# changed according to new array structure - check
# has become superflous
def get_members_idx(where, idxcs, variables):
    '''gets members of a hex_area'''
    # functional and used.
    out = {}

    idxcs = idxcs[np.where(idxcs > -1)[0],].astype(int)

    for var in variables:
        out[var] = gv.globals_dict[where][var][idxcs,]

    return out
