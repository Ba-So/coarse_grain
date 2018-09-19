#!/usr/bin/env python
# coding=utf-8
import custom_io as cio
import sampling as smp
import numpy as np

def global_integral(data):
    r_e = 6.3781e6
    int_sum = 0
    for i,area in enumerate(data['area']):
        prev_height = 0
        for j,height in enumerate(data['ZF3'][:,i]):
            dz = height - prev_height
            if dz < 0:
                print(dz)
            prev_height = height
            dA = area * (height + r_e)**2 / (data['ZF3'][0,i] + r_e)**2
            int_sum += data['value'][i,j] * dA * dz
    int_sum = int_sum/(4*np.pi*r_e**2)
    return int_sum

def discretize_data(f_path, c_path):
    grid_fine, grid_coarse = smp.load_grids(f_path, c_path)
    return smp.find_closest(grid_fine, grid_coarse)

if __name__=='__main__':
    dir_path = u'/home1/kd031/projects/icon/grids/'
    f_path = dir_path + 'iconR2B07-grid_spr0.95.nc'
    c_path = dir_path + 'iconR2B04-grid_spr0.95.nc'
    print('finding coarse, discrete indices')
    #coarse_indices = discretize_data(f_path,c_path)
    coarse_indices = range(1000)
    print('loading data sets')

    grid = {
        'path' : '~/projects/icon/experiments/BCWcold/iconR2B07-grid_refined_4.nc'
    }
    geo = {
        'path' : '~/projects/icon/experiments/BCWcold/BCWcold_R2B07_slice_onestep.nc'
    }
    data = {
        'path' : '~/projects/icon/experiments/BCWcold/BCWcold_R2B07_slice_refined_4.nc'
    }

    dataset = {}
    geo['data'] = cio.read_netcdfs(geo['path'])
    dataset['ZF3'] = geo['data']['ZF3'].values
    del geo
    dataset['ZF3'] = dataset['ZF3'][::-1,:]
    grid['data'] = cio.read_netcdfs(grid['path'])
    dataset['area'] = grid['data']['coarse_area'].values[coarse_indices]
    del grid
    data['data'] = cio.read_netcdfs(data['path'])
    print(np.shape(data['data']['t_diss'].values))
    dataset['value'] = data['data']['t_diss'].values[2,:,coarse_indices]
    del data
    print(np.shape(dataset['value']))
    dataset['value'] = dataset['value'][:, ::-1]
    print('integrating')
    print(global_integral(dataset))






