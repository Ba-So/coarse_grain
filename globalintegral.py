#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os.path as path
import modules.cio as cio


def global_integral(d_area, d_ZF3, data):
    r_e = 6.3781e6
    int_sum = 0
    for i,area in enumerate(d_area):
        prev_height = 0
        for j,height in enumerate(d_ZF3[:,i]):
            dz = height - prev_height
            if dz < 0 :
                print(dz)
            prev_height = height
            dA = area * (height + r_e)**2 / (d_ZF3[0,i] + r_e)**2
            int_sum += data[i,j] * dA * dz
    int_sum = int_sum/(4*np.pi*r_e**2)
    return int_sum

if __name__=='__main__':
    dir_path = u'/home1/kd031/projects/icon/experiments/BCWcold'
    f_path = 'iconR2B07-grid_refined_3.nc'
    g_path = 'BCWcold_slice.nc'
    IO = cio.IOcontroller(dir_path, f_path, g_path)
    d_area = IO.load_from('grid', 'coarse_area')
    d_ZF3 = IO.load_from('data', 'ZF3')
    data = IO.load_from('data', 'T_ERICH')
    print(global_integral(d_area, d_ZF3, data))

