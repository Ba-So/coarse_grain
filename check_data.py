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

if __name__ == '__main__':
   # kwargs = user()
    kwargs = {
        'experiment' : 'HS_FT_6000_days',
        'num_rings' : 3,
        'num_files' : 1
    }
    kwargs['filep'] = u'/home1/kd031/projects/icon/experiments/'+kwargs['experiment']+'/'
    kwargs['files'] = [
        n for n in
        glob.glob(kwargs['filep']+kwargs['experiment']+'_*refined*.nc') if
        os.path.isfile(n)]
    kwargs['grid'] = [
        n for n in
        glob.glob(kwargs['filep']+'*grid*.nc') if
        os.path.isfile(n)]
    print kwargs['grid']
    kwargs['variables'] = ['U', 'V', 'RHO', 'T', 't_fric', 't_diss']
    if not kwargs['files'] or not kwargs['grid']:
        sys.exit('Error:missing gridfiles or datafiles')
    grid = ma.read_grid(kwargs)
    grid_nfo = {
        'area_num_hex'        : grid['area_num_hex'].values,
        'area_neighbor_idx'   : grid['area_neighbor_idx'].values,
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
    func = lambda ds, quarks: cio.extract_variables(ds, **quarks)
    print kwargs['files']
    data = cio.read_netcdfs([kwargs['files'][i]], 'time', quarks, func)

