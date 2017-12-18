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
import math_op as mop

def account_for_pentagons(grid):
  '''accounts for superflous neighbor indices, due to pentagons'''
  
  num_edges = np.array(
            [6 for i in range(0,grid.dims['ncells'])]
            )
  cni   = grid['cell_neighbor_idx'].values

  zeroes= np.argwhere(cni == 0)


  for i in zeroes:
    num_edges[i[1]] = 5
    if i[0] != 5:
      cni[i[0],i[1]] = cni[6,i[1]]
    else:
      cni[5,i[1]] = cni[4,i[1]]

  cni -= 1

  num_edges = xr.DataArray(
            num_edges,
            dims = ['ncells']
            )
  grid  = grid.assign(num_edges = num_edges)
  grid['cell_neighbor_idx'].values = cni
  
  return grid

def define_hex_area(grid, num_rings):
  '''finds hex tiles in num_rings rings around hex tile'''

  num_hex   = 1+6*num_rings*(num_rings+1)/2

  a_nei_idx = np.array(
                [[-1 for i in range(0,grid.dims['ncells'])]
                for j in range(0, num_hex)]
                )
  num_hex   = np.array([num_hex for i in range(0,grid.dims['ncells'])])
  num_edg   = grid['num_edges'].values
  c_nei_idx = grid['cell_neighbor_idx'].values

  for idx in range(0, grid.dims['ncells']):

    jh      = 0 
    jh_c    = 1
    a_nei_idx[0,idx] = idx

    while jh_c < num_hex[idx]:
      idx_n  =  a_nei_idx[jh, idx]

      if (num_edg[idx_n] == 5):
        num_hex[idx] -= 1
        if jh_c >= num_hex[idx]:
          break
        
      for jn in range(0, num_edg[idx_n]):
        idx_c   = c_nei_idx[jn, idx_n]

        if idx_c in a_nei_idx[:,idx]:
          pass
        else:
          a_nei_idx[jh_c, idx] = idx_c
          jh_c  += 1

      jh   += 1
  
  #stuff it into grid grid DataSet

  kwargs    = {}
  area_num_hex = xr.DataArray(
            num_hex,
            dims = ['ncells']
            )
  
  kwargs['area_num_hex'] = area_num_hex

  area_neighbor_idx = xr.DataArray(
            a_nei_idx,
            dims = ['num_hex', 'ncells']
            )

  kwargs['area_neighbor_idx'] = area_neighbor_idx

  grid  = grid.assign(**kwargs)

  return grid

#This is where the rest goes
def get_members(grid, data, i, var):
  '''gets members of a hex_area'''
  num_hex   = grid['area_num_hex'].values[i]
  a_nei_idx = grid['area_neighbor_idx'].values[:,i]
  out       = np.array([])
  for var in variables:
    out       = np.vstack((out,
                  np.array(
                    [data[var].values[j] for j in a_nei_idx]
                    ))
                  )
  return out[:,:num_hex] 


def compute_uv_dyads(grid, data):
  data =    mop.area_avg_vec('hat', grid, data, ['U''V']
  data =    mop.area_avg_vec('bar', grid, data, ['U''V']
  
          




  

