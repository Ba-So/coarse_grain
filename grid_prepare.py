#!/usr/bin/env python
# coding=utf-8

import argparse
from itertools import compress
import numpy as np
import xarray as xr
import custom_io as cio
import math_op as mo

'''
    Module containg the routines for preparing the grid_files from icon.
    '''

def prepare_grid(path, num_rings):
    ''' subcontractor for reading ICON grid files '''
    # What:
    #     reassign
    #     rename
    #     sort pentagon stuff

    grid = cio.read_netcdfs(path)

    variables = ['vertex_index', 'vertices_of_vertex', 'dual_area_p']
    grid = cio.extract_variables(grid, variables)
    del variables
    print(grid)

    new_names = {
        'vertex': 'ncells',
        'vlon': 'lon',
        'vlat': 'lat',
        'vertex_index': 'cell_idx',
        'vertices_of_vertex': 'cell_neighbor_idx',
        'dual_area_p': 'cell_area'
    }
    grid = cio.rename_dims_vars(grid, new_names)
    del new_names

    new_attr_names = {
        'cell_idx': 'cell index',
        'cell_neighbor_idx': 'cell neighbor index',
        'cell_area': 'cell area'
    }
    grid = cio.rename_attr(grid, 'long_name', new_attr_names)
    del new_attr_names

    print '--------------'
    print 'accounting for pentagons'
    grid = account_for_pentagons(grid)
    print '--------------'
    print 'defining hex area members'
    grid = define_hex_area(grid, num_rings)
    print '--------------'
    print 'computing total hex area'
    grid = coarse_area(grid)
    print '--------------'
    print 'computing gradient_nfo'
    grid = get_gradient_nfo(grid)
    print '--------------'
    print 'writing file as {}_refined_{}.nc'.format(path[:-3],num_rings)
    new_path = path[:-3] + '_refined_{}.nc'.format(num_rings)
    cio.write_netcdf(new_path, grid)

    return None

def account_for_pentagons(grid):
    '''accounts for superflous neighbor indices, due to pentagons'''

    num_edges = np.array(
        [6 for i in range(0,grid.dims['ncells'])]
    )
    cni = grid['cell_neighbor_idx'].values

    zeroes = np.argwhere(cni == 0)

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
    '''
        finds hex tiles in num_rings rings around hex tiles
        input:
            grid: xr.data_set
            num_rings: number of rings in hex_area
        output:
            grid: modified with info about hex_area members attached
                new variable: 'area_member_idx' with -1 as mask
    '''

    # number of hexagons in an area of num_rings of hexagons
    num_hex   = mo.num_hex_from_rings(num_rings)


    # create array containing member information
    a_nei_idx = np.empty([num_hex, grid.dims['ncells']])
    a_nei_idx.fill(-1) # for masking

    num_edg   = grid['num_edges'].values
    c_nei_idx = grid['cell_neighbor_idx'].values

    # look at each and every gird cell and build neighbors...

    for idx in range(grid.dims['ncells']):

        jh      = 0
        jh_c    = 1
        a_nei_idx[0,idx] = idx
        check_num_hex = num_hex
        while jh_c < check_num_hex:
            idx_n  =  int(a_nei_idx[jh, idx])

            if (num_edg[idx_n] == 5):
                check_num_hex -= 1
                if jh_c >= check_num_hex:
                    break

            for jn in range(0, num_edg[idx_n]):
                idx_c   = c_nei_idx[jn, idx_n]

                if idx_c in a_nei_idx[:,idx]:
                    pass
                elif jh_c < check_num_hex:
                    a_nei_idx[jh_c, idx] = idx_c
                    jh_c  += 1
                else:
                    break
                    print 'define_hex_area: error jh_c to large'

            jh   += 1

    #stuff it into grid grid DataSet

    area_member_idx = xr.DataArray(
        a_nei_idx,
        dims = ['num_hex', 'ncells']
        )

    kwargs = {'area_member_idx' : area_member_idx}

    grid  = grid.assign(**kwargs)

    return grid

def coarse_area(grid):
    '''
        Sums the areas of its members in area_member_idx to coarse_area
        input:
            grid : xr.dataset containing 'cell_area', 'area_member_idx'
        returns:
            grid : xr.dataset with variable 'coarse_area' added
    '''

    co_ar = np.array([0.0 for i in range(0, grid.dims['ncells'])])
    cell_a = grid['cell_area'].values
    a_nei_idx = grid['area_member_idx'].values



    for i in range(grid.dims['ncells']):
        areas = cell_a[np.where(a_nei_idx[:,i] > -1)[0]]
        co_ar[i] = np.sum(areas)

    coarse_a = xr.DataArray(
        co_ar,
        dims=['ncells'])
    kwargs = {'coarse_area' : coarse_a}
    grid = grid.assign(**kwargs)

    return grid

def get_gradient_nfo(grid):
    '''
        computes the coordinates of neighbors for gradient computations
        input:
            grid : xr.dataset containing,
                'coarse_area', 'cell_area'
        output:
            grid, updated with,
                coords: contains the neighbor point coordinates
                    numpy([ncells, corners, [lon, lat])
                members_idx: contains the indices of members
                    [ncells, numpy(idxcs)]
                members_rad: contains the relative distance of members to center
                    [ncells, numpy(radii)]
    '''
    ncells = grid.dims['ncells']
    coarse_area = grid['coarse_area'].values
    cell_area = grid['cell_area'].values
    lon = grid['lon'].values
    lat = grid['lat'].values
    print(lon.shape)
    print(lat.shape)
    print(lon[1])

    # compute the coordinates of the four corners for gradient
    print(' --------------')
    print(' computing corner coordinates')
    coords = np.empty([ncells, 4, 2])
    coords.fill(0)
    for i in range(ncells):
        lonlat  = [lon[i], lat[i]]
        area    = coarse_area[i]
        r  = 2 * mo.radius(area)
        coords[i, :, :] = gradient_coordinates(lonlat, r)

    # logic: num_hexagons in area of r = d_hexagon = 2 * r_hexagon
    #     <= num_hexagons in d rings of hexagons
    # number of hexagons in an area of num_rings of hexagons
    times_rad = 2
    max_members = 1 + 6 * times_rad/2 * (times_rad/2 + 1) / 2

    # compute radii for finding members
    print(' --------------')
    print(' computing bounding radii')
    check_rad = np.empty([ncells], dtype = float)
    check_rad.fill(0)
    for i in range(ncells):
        check_rad[i] = times_rad * mo.radius(cell_area[i])

    test_rad = np.equal(check_rad, 0)
    if True in test_rad:
        print("bad stuff just happened")
        return None
    del test_rad


    # get bounding box to find members
    print(' --------------')
    print(' computing bounding boxes')
    bounds = np.empty([ncells, 4, 2, 2, 2])
    bounds.fill(0)
    for i in range(ncells):
        for j in range(4):
            lonlat = coords[i, j, :]
            bounds[i, j, :, :, :] = max_min_bounds(lonlat, check_rad[i])

    print(' --------------')
    print(' finding members for gradient approximation')
    candidates = np.empty([ncells, 4, 400], dtype = int)
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
            check = candidates[i , j, np.where(candidates[i,j,:] > -1)[0]]
            cntr = 0
            for k, idx in enumerate(check):
                check_r = mo.arc_len(
                        coords[i, j, :],
                        [lon[idx], lat[idx]])
                if check_r <= check_rad[i]:
                    member_idx[i, j, cntr] = idx
                    member_rad[i, j, cntr] = check_r
                    cntr += 1

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

def gradient_coordinates(lonlat, r):
    '''
        computes the locations around which to
        compute the averages for the gradient computation
        input:
            lonlat : array [lon, lat]
            r : radius of area to search in
    '''



    lat_min = lonlat[1] - r
    lat_max = lonlat[1] + r
    if lat_max > np.pi/2:
        # northpole in query circle:
        lat_max = lat_max - np.pi
    elif lat_min < -np.pi/2:
        # southpole in query circle:
        lat_min = lat_min + np.pi
    r = r / np.cos(lonlat[1])
    lon_min = lonlat[0] - r
    lon_max = lonlat[0] + r

    # in case lon passes the median
    if lon_min < -np.pi:
        lon_min = lon_min + np.pi
    elif lon_max > np.pi:
        lon_max = lon_max - np.pi

    coords = np.array([
        np.array([lon_min, lat_min]),
        np.array([lon_min, lat_max]),
        np.array([lon_max, lat_min]),
        np.array([lon_max, lat_max])
        ])
    return coords

def max_min_bounds(lonlat, r):
    '''
        computes the maximum and minimum lat and lon values on a circle on a sphere
        input:
            lonlat : array [lat, lon]
            r : radius of circle on sphere
        output:
            bounds : array [i, j, k]
                i = {0,1} for special cases around poles two sets
                j = {0,1} für {min, max}
                k = {0,1} für {lat, lon}
            '''
    # get radius
    # coords[0,:] latitude values
    # coords[1,:] longditude values
    # u is merdional wind, along longditudes
    #     -> dx along longditude
    # v is zonal wind, along latitudes
    #     -> dy along latitude
    # coords[:,:2] values for dx
    # coords[:,:2] values for dy
    # computing minimum and maximum latitudes
    lock = False
    lat_min = lonlat[1] - r
    lat_max = lonlat[1] + r
    bounds = np.empty([2, 2, 2])
    bounds.fill(-4)
    if lat_max > np.pi/2:
        # northpole in query circle:
        bounds[0, :, :] = [[lat_min, -np.pi],[np.pi/2, np.pi]]
        lock = True
    elif lat_min < -np.pi/2:
        # southpole in query circle:
        bounds[0, :, :] = [[-np.pi/2, -np.pi],[lat_max, np.pi]]
        lock = True
    else:
        # no pole in query circle
        bounds[0, :, :] = [[lat_min, -np.pi], [lat_max, np.pi]]
  # computing minimum and maximum longditudes:
    if not lock:
        lat_T = np.arcsin(np.sin(lonlat[1])/np.cos(r))
        d_lon = np.arccos(
            (np.cos(r) - np.sin(lat_T) * np.sin(lonlat[1]))
            /(np.cos(lat_T)*np.cos(lonlat[1])))
        lon_min = lonlat[0] - d_lon
        lon_max = lonlat[0] + d_lon

        if lon_min < -np.pi:
            bounds[1, :, :] = bounds[0, :, :]
            bounds[0, 0, 1] = lon_min + 2 * np.pi
            bounds[0, 1, 1] = np.pi
            #and
            bounds[1, 0, 1] = - np.pi
            bounds[1, 1, 1] = lon_max
        elif lon_max > np.pi:
            bounds[1, :, :] = bounds[0, :, :]
            bounds[0, 0, 1] = lon_min
            bounds[0, 1, 1] = np.pi
            #and
            bounds[1, 0, 1] = - np.pi
            bounds[1, 1, 1] = lon_max - 2 * np.pi
        else:
            bounds[0, 0, 1] = lon_min
            bounds[0, 1, 1] = lon_max

    return bounds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare ICON gridfiles for Coarse-Graining.')
    parser.add_argument(
        'path_to_file',
        metavar = 'path',
        type = str,
        nargs = '+',
        help='a string specifying the path to the gridfile'
    )
    parser.add_argument(
        'num_rings',
        metavar = 'num_rings',
        type = int,
        nargs = '+',
        help = 'an integer specifying the number of rings to be coarse-grained over.'
    )
    args = parser.parse_args()
    print(
        'preparing the grid file {} for coarse graining over {} rings'
    ).format(args.path_to_file[0], args.num_rings[0])
    prepare_grid(args.path_to_file[0], args.num_rings[0])


