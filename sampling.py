#!/usr/bin/env python
# coding=utf-8
import numpy as np
import custom_io as cio
import math_op as mo
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
    print('the difference from the optimal ratio is: {}'.format(ratio))
    print('the difference in area is:{}'.format(ratio*coarse_area - coarse_area))
    print('num_rings:{}'.format(num_rings))
    return num_rings

if __name__ == '__main__':
    dir_path = u'/home1/kd031/projects/icon/grids/'
    path_grid_coarse = dir_path + 'iconR2B04-grid_spr0.95.nc'
    path_grid_fine = dir_path + 'iconR2B07-grid_spr0.95.nc'
    grid_fine, grid_coarse = load_grids(path_grid_fine, path_grid_coarse)
    num_rings = approximate_num_rings(grid_fine, grid_coarse)




