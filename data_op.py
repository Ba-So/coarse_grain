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
from itertools import compress
import math_op as mo

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
      cni[i[0],i[1]] = cni[5,i[1]]
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
        elif jh_c < num_hex[idx]:
          a_nei_idx[jh_c, idx] = idx_c
          jh_c  += 1
        else:
          break
          print 'define_hex_area: error jh_c to large'

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

def get_members(grid_nfo, data, i, variables):
  '''gets members of a hex_area'''
  # functional and used.
  num_hex   = grid_nfo['area_num_hex'][i]
  a_nei_idx = grid_nfo['area_neighbor_idx'][:,i]
  out       = {}

  for var in variables:
      if data[var].ndim == 3:
          out[var] = data[var][:, :, a_nei_idx[:num_hex]]
         # out[var] = np.array([data[var][:,:,j] for j in a_nei_idx[:num_hex]])
      if data[var].ndim == 2:
          out[var] = data[var][:, a_nei_idx[:num_hex]]
        #  out[var] = np.array([data[var][:,j] for j in a_nei_idx[:num_hex]])
      if data[var].ndim == 1:
          out[var] = data[var][a_nei_idx[:num_hex]]
        #  out[var] = np.array([data[var][j] for j in a_nei_idx[:num_hex]])

  return out

def get_members_idx(data, idxcs, variables):
    '''gets members of a hex_area'''
    # functional and used.
    out       = {}

    for var in variables:
        if data[var].ndim == 3:
            out[var] = data[var][:, :, idxcs]
        #    out[var] = np.array([data[var][:, :, j] for j in idxcs])
        if data[var].ndim == 2:
            out[var] = data[var][:, idxcs]
        #    out[var] = np.array([data[var][:, j] for j in idxcs])
        if data[var].ndim == 1:
            out[var] = data[var][idxcs]
        #    out[var] = np.array([data[var][j] for j in idxcs])

    return out

def get_gradient_nfo(grid):
    '''computes the coordinates of neighbors for gradient computations
    gradient_nfo:
        coords: contains the neighbor point coordinates
            numpy([ncells, corners, [lon, lat])
        members_idxcs: contains the indices of members
            [ncells, numpy(idxcs)]
        members_rad: contains the relative distance of members to center
            [ncells, numpy(radii)]
    '''
    ncells = grid.dims['ncells']
    coarse_area = grid['coarse_area'].values
    cell_area = grid['cell_area'].values
    lon = grid['lon'].values
    lat = grid['lat'].values

    max_members = 200

    coords = np.empty([ncells, 4, 2])
    coords.fill(0)
    # compute the coordinates of the four corners for gradient
    print(' --------------')
    print(' computing corner coordinates')
    for i in range(ncells):
        lonlat  = [lon[i], lat[i]]
        area    = coarse_area[i]
        coords[i, :, :] = mo.gradient_coordinates(lonlat, area)

    # compute radii for finding members
    print(' --------------')
    print(' computing bounding radii')
    check_rad = np.empty([ncells], dtype = float)
    check_rad.fill(0)
    for i in range(ncells):
        check_rad[i] = 2 * mo.radius(cell_area[i])

    test_rad = np.equal(check_rad, 0)
    if True in test_rad:
        print("bad stuff just happened")
        return None
    # get bounding box to find members
    print(' --------------')
    print(' computing bounding boxes')
    bounds = np.empty([ncells, 4, 2, 2, 2])
    bounds.fill(0)
    for i in range(ncells):
        for j in range(4):
            lonlat = coords[i, j, :]
            bounds[i, j, :, :, :] = mo.max_min_bounds(lonlat, check_rad[i])

    print(' --------------')
    print(' finding members for gradient approximation')
    candidates = np.empty([ncells, 4, max_members], dtype = int)
    candidates.fill(-1)
    print('   --------------')
    print('   checking bounds')

    for i in range(ncells):
        for j in range(4):
            # using numpy class for highest possible optimization!
            test_lat_1 = np.all([
                np.greater_equal(lat, bounds[i, j, 0, 0, 0]),
                np.less_equal(lat, bounds[i, j, 0, 1, 0])
            ], 0)

            test_lat_2 = np.all([
                np.greater_equal(lat, bounds[i, j, 1, 0, 0]),
                np.less_equal(lat, bounds[i, j, 1, 1, 0])
            ], 0)

            test_lat = np.any([test_lat_2, test_lat_1], 0)

            test_lon_1 = np.all([
                np.greater_equal(lon, bounds[i, j, 0, 0, 1]),
                np.less_equal(lon, bounds[i, j, 0, 1, 1])
            ], 0)

            test_lon_2 = np.all([
                np.greater_equal(lon, bounds[i, j, 1, 0, 1]),
                np.less_equal(lon, bounds[i, j, 1, 1, 1])
            ], 0)

            test_lon = np.any([test_lon_2, test_lon_1], 0)

            test = np.all([test_lat, test_lon], 0)

            helper = list(compress(range(ncells), test))
            candidates[i, j, :len(helper)] = helper

    print('   --------------')
    print('   checking candidates')
    member_idx = np.empty([ncells, 4, max_members], dtype = int)
    member_idx.fill(-1)
    member_rad = np.empty([ncells, 4, max_members], dtype = float)
    member_rad.fill(-1)
    for i in range(ncells):
        for j in range(4):
            check = np.array([x for x in candidates[i, j] if x != -1])
            for k, idx in enumerate(check):
                r = mo.arc_len(
                        coords[i, j, :],
                        [lon[idx], lat[idx]])
                if r <= check_rad[i]:
                    member_idx[i, j, k] = idx
                    member_rad[i, j, k] = r

    #incorporate
    coords = xr.DataArray(
        coords,
        dims = ['ncells','num_four','num_two'])
    member_idx = xr.DataArray(
        member_idx,
        dims= ['ncells','num_four','max_members'])
    member_rad = xr.DataArray(
        member_rad,
        dims= ['ncells','num_four','max_members'])

    kwargs = {
        'coords' : coords,
        'member_idx' : member_idx,
        'member_rad' : member_rad
        }

    grid  = grid.assign(**kwargs)

    return grid


