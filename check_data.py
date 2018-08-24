#!/usr/bin/env python
# coding=utf-8

import os
import sys
import glob
import numpy as np
import custom_io as cio
import data_op as dop
import math_op as mo
import phys_op as po
import main as ma
import xarray as xr
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

def plot_data(data, name):
    fig, ax = plt.subplots()
    density, bins= np.histogram(data, normed = True, density = True, bins = 50 )
    unity_density = density/density.sum()
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + unity_density

    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    barpath = path.Path.make_compound_path_from_polys(XY)

    patch = patches.PathPatch(barpath)
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    ax.set_yscale('log')
    ax.set_ylim(bottom.min(), top.max())
    plt.savefig(name,dpi=73)
    plt.show()

    plt.close()

if __name__ == '__main__':
   # kwargs = user()
    kwargs = {
        'experiment' : 'BCW_CG_15_days',
        'num_rings' : 3,
        'num_files' : 9
    }
    kwargs['filep'] = u'/home1/kd031/projects/icon/experiments/'+kwargs['experiment']+'/'
    kwargs['files'] = [
        n for n in
        glob.glob(kwargs['filep']+'time_slice'+'_*refined*.nc') if
        os.path.isfile(n)]
    kwargs['grid'] = [
        n for n in
        glob.glob(kwargs['filep']+'*grid*.nc') if
        os.path.isfile(n)]
    print kwargs['grid']
    print kwargs['files']
    kwargs['variables'] = ['T', 't_fric', 't_diss', 'K']
    if not kwargs['files'] or not kwargs['grid']:
        sys.exit('Error:missing gridfiles or datafiles')
    grid = ma.read_grid(kwargs)
    grid_nfo = {
        'area_neighbor_idx'   : grid['area_member_idx'].values,
        'coarse_area'         : grid['coarse_area'].values,
        'cell_area'           : grid['cell_area'].values,
        'i_cell'              : 0
        }
    gradient_nfo = {
        'coords' : grid['coords'].values,
        'member_idx' : grid['member_idx'].values,
        'member_rad' : grid['member_rad'].values
    }
    if kwargs['num_files'] > len(kwargs['files']):
        fin = len(kwargs['files'])
    else:
        fin = kwargs['num_files']

    i = 0
    quarks = {}
    quarks['variables'] = kwargs['variables']
    data = cio.read_netcdfs(kwargs['files'][0])
    data = cio.rename_dims_vars(data)
    data_K = data['K'].values
    for i, file in enumerate(kwargs['files'][1:]):
        data = cio.read_netcdfs(kwargs['files'][i])
        data = cio.rename_dims_vars(data)
        data_K = np.concatenate((data_K, data['K'].values))
    data_K = np.multiply(data_K, -1)
    #data = cio.extract_variables(data, kwargs['variables'])
    #plot_data(data['t_fric'].values, 't_fric')
    #plot_data(data['t_diss'].values, 't_fric')
    a = [
        np.mean(data_K),
        np.average(data_K),
        np.std(data_K),
        np.var(data_K),
    ]
    print('mean {}, average {}, std {}, var {}'.format(a[0], a[1], a[2], a[3]))
    plot_data(data_K, 'K')
    K_filtered = data_K[data_K < 1e14]
    K_filtered = K_filtered[K_filtered > -1e14]
    plot_data(K_filtered, 'K_filtered_1e14')
    K_filtered = data_K[data_K < 1e10]
    K_filtered = K_filtered[K_filtered > -1e10]
    plot_data(K_filtered, 'K_filtered_around zero')
