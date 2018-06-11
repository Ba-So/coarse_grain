# all data manipulations necessary for computations
# Here we'll define the hex_areas and find neighbors
# MISSING:
#   * Routine to assign hex area neighbors to central hexes
#   * Routine to cast hexes into objects.
#       -> parallelization through using as many objects as there are
#       processors? with one global dict, giving information on which processor specific
#       hexes are to allow messages to be sent between them.
#       -> read on parallelization?
import xarray as xr
import numpy as np
import math_op as mo



# changed according to new array structure - check
def get_members(grid_nfo, data, i, variables):
  '''gets members of a hex_area'''
  # functional and used.
  num_hex   = grid_nfo['area_num_hex'][i]
  a_nei_idx = grid_nfo['area_neighbor_idx'][i,:]
  out       = {}
  idxcs     = a_nei_idx[np.where(a_nei_idx[:num_hex]>-1)[0],]

  out = get_members_idx(data, idxcs, variables)

  return out

# changed according to new array structure - check
def get_members_idx(data, idxcs, variables):
    '''gets members of a hex_area'''
    # functional and used.
    out       = {}

    idxcs = idxcs[np.where(idxcs>-1)[0],].astype(int)

    for var in variables:
        out[var] = data[var][idxcs,]
    # why this?
    #   out[var] = np.moveaxis(out[var], -1, 0)

  #  for var in variables:
  #      if data[var].ndim == 3:
  #          out[var] = data[var][:, :, np.where(idxcs>-1)[0]]
  #      #    out[var] = np.array([data[var][:, :, j] for j in idxcs])
  #      if data[var].ndim == 2:
  #          out[var] = data[var][:, np.where(idxcs>-1)[0]]
  #      #    out[var] = np.array([data[var][:, j] for j in idxcs])
  #      if data[var].ndim == 1:
  #          out[var] = data[var][np.where(idxcs>-1)[0]]
  #      #    out[var] = np.array([data[var][j] for j in idxcs])
  #      out[var] = np.moveaxis(out[var], -1, 0)

    return out

