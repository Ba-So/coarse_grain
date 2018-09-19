#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
from glob import glob
import numpy as np
import sys
from itertools import compress
import custom_io as cio
import math_op as mo
import grid_prepare as gp
import histogramms as hi
# Sampling.py
# Works independently from the main program, can however be invoked from there

def load_grids(path_grid_fine, path_grid_coarse):
    '''load_grids(str path_grid_fine, str path_grid_coarse)
        loads two ICON gridfiles and returns them as
        xarray grid_fine and xarray grid_coarse'''

    # read grid
    grid_fine = cio.read_netcdfs(path_grid_fine)
    grid_coarse = cio.read_netcdfs(path_grid_coarse)
    # extract relevant information
    variables = ['vertex_index', 'vertices_of_vertex', 'dual_area_p']
    grid_fine = cio.extract_variables(grid_fine, variables)
    grid_coarse = cio.extract_variables(grid_coarse, variables)
    del variables
    # rename dimensions to standard
    new_names = {
        'vertex': 'ncells',
        'vlon': 'lon',
        'vlat': 'lat',
        'vertex_index': 'cell_idx',
        'vertices_of_vertex': 'cell_neighbor_idx',
        'dual_area_p': 'cell_area'
    }
    grid_fine = cio.rename_dims_vars(grid_fine, new_names)
    grid_coarse = cio.rename_dims_vars(grid_coarse, new_names)
    # rename variables
    new_attr_names = {
        'cell_idx': 'cell index',
        'cell_neighbor_idx': 'cell neighbor index',
        'cell_area': 'cell area'
    }
    grid_fine = cio.rename_attr(grid_fine, 'long_name', new_attr_names)
    grid_coarse = cio.rename_attr(grid_coarse, 'long_name', new_attr_names)

    return grid_fine, grid_coarse

def approximate_num_rings(grid_fine, grid_coarse):
    '''approximate_num_rings(xarray grid_fine, xarray grid_coarse)
        approximates the closest number of rings for
        coarse-graining the fine grid towards the resolution of the coarse grid.
        computes the average grid-sizes, then computes the coarse area
        by iteratively computing the coarse areas and comparing the ratios
        of coarse to coarse-grained areas.
        prints the optimal ratio.
        returns int num_rings'''

    fine_area = np.mean(grid_fine['cell_area'].values)
    coarse_area = np.mean(grid_coarse['cell_area'].values)

    check = False
    ratio_old = 1
    ratio_new = abs(1 - fine_area / coarse_area)
    num_rings = 0
    while ratio_old > ratio_new:
        num_rings += 1
        ratio_old = ratio_new
        ratio_new = abs(1 - fine_area * mo.num_hex_from_rings(num_rings) / coarse_area)
    num_rings -= 1
    ratio = fine_area * mo.num_hex_from_rings(num_rings) / coarse_area
    print('the difference in area is:{}'.format(ratio*coarse_area - coarse_area))
    print('num_rings:{}'.format(num_rings))
    return num_rings

def find_closest(grid_fine, grid_coarse):
    '''find_closest(xarray grid_fine, xarray grid_coarse)
     find ncells of grid_fine members closest to grid_coarse members,
     used for discrete comparison of areas.'''
    # Thoughts: simply going through and comparing all the distances is too
    # expensive. Use the functions I have already written for finding elements
    # within a circle?
    ncells_c = grid_coarse.dims['ncells']
    cell_area_c = grid_coarse['cell_area'].values
    ncells_f = grid_fine.dims['ncells']
    lon_coarse = grid_coarse['lon'].values
    lat_coarse = grid_coarse['lat'].values
    lon_fine = grid_fine['lon'].values
    lat_fine = grid_fine['lat'].values
    area_aspec_ratio = np.average(cell_area_c)/ np.average(grid_fine['cell_area'].values)
    closest_fine_cell = np.full((ncells_c), -1, dtype = 'int')
    center_offset = np.empty((ncells_c))
    print_at = 1
    # restructure. instead of cellwise do the np.all comparisons all at once
    # simmilar to how it is done in grid prepare
    for i in range(ncells_c):
        lonlat = [lon_coarse[i], lat_coarse[i]]
        # compute a bounding box
        check_rad = mo.radius(cell_area_c[i]) * 4 / area_aspec_ratio
        bounds = gp.max_min_bounds(lonlat, check_rad)
        # check stuff
        test_lat_1 = np.all([
            np.greater_equal(lat_fine, bounds[0, 0, 0]),
            np.less_equal(lat_fine, bounds[0, 1, 0])
        ], 0)

        test_lat_2 = np.all([
            np.greater_equal(lat_fine, bounds[1, 0, 0]),
            np.less_equal(lat_fine, bounds[1, 1, 0])
        ], 0)

        test_lat = np.any([test_lat_2, test_lat_1], 0)

        test_lon_1 = np.all([
            np.greater_equal(lon_fine, bounds[0, 0, 1]),
            np.less_equal(lon_fine, bounds[0, 1, 1])
        ], 0)

        test_lon_2 = np.all([
            np.greater_equal(lon_fine, bounds[1, 0, 1]),
            np.less_equal(lon_fine, bounds[1, 1, 1])
        ], 0)

        test_lon = np.any([test_lon_2, test_lon_1], 0)

        test = np.all([test_lat, test_lon], 0)

        candidates = list(compress(range(ncells_f), test))

        check_min = check_rad
        index_min = 0
        check_r = check_min
        if len(candidates) == 0:
            print(candidates)
            print('something is off, list of candidates is zero')
            print('icell  {}'.format(i))
            sys.exit('wierd stuff')

        for j, candidate in enumerate(candidates):
        #    print('candidate number {}'.format(j))
            check_r = mo.arc_len(
                lonlat,
                [lon_fine[candidate], lat_fine[candidate]]
            )
            if check_r < check_min:
                check_min = check_r
                closest_fine_cell[i] = candidate
                center_offset[i] = check_min

    if np.any(closest_fine_cell == -1):
        print('something is off indice {} appeared'.format(
            closest_fine_cell[np.where(closest_fine_cell == -1)])
        )
        sys.exit('Error: invalid index integer')

    print('The average offset from center is {}'.format(np.average(center_offset)))
    return closest_fine_cell

def filter_and_write(fine_experiment_path, coarse_experiment_path, key=None):
    '''filter_continuous_cg(str fine_experiment_path, str coarse_experiment_path)
        filter Routine, to turn continuoufines into discrete dataset for direct
        comparison of coarse and fine turbulentfriction values.
        Output: file, 'fine_experiment_path'+'discretized'
        '''
    # To Do
    # load grid files from both experiments. use glob to find them in
    # experiment older.
    # get closest_fine_cell
    # load coarse_grained datasets, find files using glob, load sucessively:
    #    filter coarse_grained dataset
    #    write dataset as path_of_dataset_filtered_BXX, where BXX is the
    #    corresponding grid mesh.
    # Done
    ## Read up on use of glob: glob(pathname) searches for patterns
    closest_fine_cell = find_closest_inexperiments(fine_experiment_path, coarse_experiment_path)
    # sort of a hickup due to file naming here. Problem to be solved outside
    fine_cg_file_paths = find_experiment_data_in(fine_experiment_path, key)
    print(closest_fine_cell.shape)
    print(fine_cg_file_paths)

    for i, path in enumerate(fine_cg_file_paths):

        print('editing file {}'.format(path))

        fine_data = cio.read_netcdfs(path)
        fine_dara = fine_data[['VORC','U','V']]
        out_data = fine_data.isel(cell2 = closest_fine_cell)

        cio.write_netcdf(path[:-3] + '_disc.nc', out_data)

def find_experiment_data_in(path, key = None):
    print('globbing for file paths in {}'.format(path))
    if key:
        path = path + '/' + key
    else:
        path = path + '/' +'*_refined_*.nc'
    file_paths = glob(path)
    file_paths = [path for path in file_paths if 'grid' not in path]
    if not file_paths:
        sys.exit("Error: I didn't find a file in {}".format(path))
    return file_paths

def find_closest_inexperiments(fine_experiment_path, coarse_experiment_path):
    fine_grid_path = glob(fine_experiment_path + '/*grid.nc')
    coarse_grid_path = glob(coarse_experiment_path + '/*grid.nc')
    print(fine_grid_path, coarse_grid_path)
    if not fine_grid_path or not coarse_grid_path:
        sys.exit('ERROR: couldnt find the friggin gird files!')
    print('reading files')
    grid_fine, grid_coarse = load_grids(path_grid_fine, path_grid_coarse)
    print('finding indices for discretization')
    closest_fine_cell = find_closest(grid_fine, grid_coarse)
    return closest_fine_cell

def compare_coarse_and_fine(fine_experiment_path, coarse_experiment_path, key):
    '''compares blabla'''
    # What is this supposed to do?
    # I want to do a comparison, similar to that of Lucarini, where I show how
    # large the Error due to a coarse resolution model is, mean and maximum
    # values. It would therefore be interesting to have a numberdistribution
    # plot of the percentage difference´- assuming the high resolution model is
    # accurate - that difference is the "error".
    # To Do:
    #    load data of grids
    #    discretize fine data based of of grid information
    # closest_fine_cell = find_closest_inexperiments(fine_experiment_path, coarse_experiment_path)
    #    load coarse_data, load find_data and immediately discard superflous info
    coarse_data_path = find_experiment_data_in(coarse_experiment_path, '*_coarse*.nc')
    print(coarse_data_path)
    coarse_data = cio.read_netcdfs(coarse_data_path[0])
    data_co = coarse_data['KMH'].values
    del coarse_data
    fine_data_path = find_experiment_data_in(fine_experiment_path, '*disc*.nc')
    data_shape = data_co.shape

    compared_data = np.zeros(data_shape)
    ntime_start = 0
    ntime_fine = 0
    for data_path in fine_data_path:
        fine_data_subset = cio.read_netcdfs(data_path)
        data_fine = fine_data_subset[key].values
        del fine_data_subset
        ntime_fine += data_fine.shape[0]
        print(ntime_start)
        print(ntime_fine)
        print(data_path)
        compared_data[ntime_start:ntime_fine, :, :] = compare(data_co[ntime_start:ntime_fine, :,:], data_fine)
        ntime_start += data_fine.shape[0]
    key = key + 'comp'
    hi.plot_histogram(compared_data, key)



def compare(data_coarse, data_fine):
    # To Do clarify which quantities I want to compare and make sure they're in
    # the data set -> run ICON, do tomorrow
    return np.subtract(data_coarse, data_fine)

def get_optimal_coarse_graining(coarse_experiment_path, fine_experiment_path):
    '''get_optimal_coarse_graining(str coarse_experiment_path, str fine_experiment_path)
        returns optimal number of rings for coarse graining and comparison.
        output int number_of_rings'''
    fine_grid_path = glob(fine_experiment_path + '/*grid.nc')
    coarse_grid_path = glob(coarse_experiment_path + '/*grid.nc')
    grid_fine, grid_coarse = load_grids(path_grid_fine, path_grid_coarse)
    num_rings = approximate_num_rings(grid_fine, grid_coarse)
    return num_rings

if __name__ == '__main__':
    dir_path = u'/home1/kd031/projects/icon/grids/'
    path_grid_coarse = dir_path + 'iconR2B04-grid_spr0.95.nc'
    path_grid_fine = dir_path + 'iconR2B07-grid_spr0.95.nc'
    grid_fine, grid_coarse = load_grids(path_grid_fine, path_grid_coarse)
    num_rings = approximate_num_rings(grid_fine, grid_coarse)
    #closest_fine_cell = find_closest(grid_fine, grid_coarse)
    dir_path = u'/home1/kd031/projects/icon/experiments'
    fine_experiment = 'BCWcold'
    coarse_experiment = 'BCW_coarse'
    key = "BCWcold_R2B07_slice_onestep.nc"
    filter_and_write(dir_path + '/' + fine_experiment, dir_path + '/' + coarse_experiment, key)
 #   compare_coarse_and_fine(dir_path + '/' + fine_experiment, dir_path + '/' + coarse_experiment, key)
    # TO DO
    # build a parser




