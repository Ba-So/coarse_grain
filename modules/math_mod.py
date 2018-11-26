#!/usr/bin/env python
# coding=utf-8
import numpy as np
from debugdecorators import TimeThis, PrintArgs
from paralleldecorators import Mp, ParallelNpArray
from functiondecorators import requires
#pass data as full, but give c_area in slices.
#race conditions?
mp = Mp(8, False) # standart be 8 cores, and deactivated

#@TimeThis
@requires()
@ParallelNpArray(mp)
def func(ina, inb, ret):
    """function sceleton"""
    foo = ina * inb
    ret = foo
#--------------------
#@TimeThis
@requires({
    'full' : ['data'],
    'slice' : ['c_area', 'ret']
})
@ParallelNpArray(mp)
def avg_bar(data, c_area, ret):
    """computes the bar average of data
        data - needs to be full stack
        c_area - can be part"""
    vals = np.sum(data, 0)
    ret[:] = np.divide(vals, coarse_area)
#--------------------
#@TimeThis
@requires({
    'full' : ['data', 'rho'],
    'slice' : ['rho_bar', 'c_area', 'ret']
})
@ParallelNpArray(mp)
def avg_hat(data, rho, rho_bar, c_area, ret):
    """computes the hat average of data,
    requires the average of rho"""
    vals = np.sum(np.multiply(data, rho, 0))
    ret[:] = np.divide(vals, (c_area, rho_bar))
#--------------------
#@TimeThis
@requires({
    'full' : [],
    'slice' : ['ret', 'data', 'data_avg']
})
@ParallelNpArray(mp)
def fluctsof(data, data_avg, ret):
    """computes deviations from local mean"""
    ret[:] = np.subtract(data_avg, data)
#--------------------
#@TimeThis
@requires({
    'full' : ['u', 'v', 'grid_nfo'],
    'slice' : ['gradient_nfo', 'gradient']
})
@ParallelNpArray(mp)
def uv_2D_gradient(u, v, grid_nfo, gradient_nfo, gradient):
    """computes the gradient of velocity components in u,v using
    the information in gradient_nfo
        return: gradient
    """
    r_e = 6.37111*10**6
    for i, g_nfo in gradient_nfo:
        neighs_u = []
        neighs_v = []
        for j in range(4):
            grid_nfo_fltrd = grid_nfo[g_nfo['member_idx'][j]]
            cell_areas = [k['cell_area'] of i,k in enumerate(grid_nfo_fltrd)]
            helper = circ_dist_avg_vec(
                x_values,
                y_values,
                g_nfo['coords'][j,:,],
                grid_nfo_fltrd
            )
            neighs_u.append(helper[0])
            neighs_v.append(helper[1])
        area = grid_nfo[g_nfo['icell']]['coarse_area']
        d = 2 * radius(area) * r_e
        for j in range(2): # dx, dy
            gradient[i, j, 0, :,] = central_diff(
                neighs_u[(2*j)],
                neighs_u[(2*j)+1],
                d
            )
            gradient[i, j, 1, :,] = central_diff(
                neighs_v[(2*j)],
                neighs_v[(2*j)+1],
                d
            )

#@TimeThis
def dist_avg_vec(x_values, y_values, center_coords, grid_nfo):
    '''
    does distance weighted average of values within a circular area on a sphere.
    needs:  coordinates of center, indices of area members,
            distances of members from center,'''
    #shift lat lon grid to a local pole.
    val_coords = [
        [k['lat'] for i,k in enumerate(grid_nfo)],
        [k['lon'] for i,k in enumerate(grid_nfo)],
    ]
    plon, plat = get_polar(center_coords[0], center_coords[1])
    x_vec, y_vec = rotate_multiple_to_local(
        val_coords[0],
        val_coords[1],
        plon,
        plat,
        x_values,
        y_values,
    )
    #turn back to global is unneccesary
    x_vec = dist_avg_scalar(x_vec, grid_nfo)
    y_vec = dist_avg_scalar(y_vec, grid_nfo)

    return x_vec, y_vec


#@TimeThis
def dist_avg_scalar(x_values, grid_nfo):
    factor = 0
    weights = [k['cell_area'] * k['radii'] for i,k in enumerate(grid_nfo)]
    factor = np.sum(weights)

    average = 0
    for i,weight in enumerate(weights):
        average = average + x_values[i] * weight
    return average / factor

def central_diff(xl, xr, d):
    return np.divide(np.subtract(xl, xr), 2 * d)

def radius(area):
    '''returns radius of circle on sphere in rad'''
    r_e = 6.37111*10**6
    r = np.sqrt(area / np.pi) / r_e
    return r

def get_polar(lon, lat):
    plon = 0.0
    if 0 < lon <= np.pi:
        plon = lon - np.pi
    elif -np.pi < lon < 0:
        plon = lon + np.pi

    plat = np.pi/2
    if 0 < lat <= np.pi/2:
        plat = np.pi/2 - lat
    elif -np.pi/2 <= lat < 0:
        plat = -np.pi/2 - lat

    return plon, plat

def rotate_multiple_to_local(lon, lat, plon, plat, x, y):
    '''wrapper to iterate over multiple values being turned onto the same local grid'''
    len_x = x.shape[0]
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)
    for i in range(len_x):
        x_tnd[i, :, ], y_tnd[i, :, ] = rotate_vec_to_local(
            plon, plat,
            lon[i], lat[i],
            x[i, :, ], y[i, :, ]
        )
    return x_tnd, y_tnd

def rotate_vec_to_local(plon, plat, lon, lat, x, y):
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)

    sin_d, cos_d = rotate_latlon_vec(lon, lat, plon, plat)

    x_tnd = x * cos_d - y * sin_d
    y_tnd = x * cos_d + y * sin_d

    return x_tnd, y_tnd

def rotate_latlon_vec(lon, lat, plon, plat):
    ''' returns entries of rotation matrix according to documentation of COSMOS pp. 27'''

    z_lamdiff = lon - plon
    z_a = np.cos(plat) * np.sin(z_lamdiff)
    z_b = np.cos(lat) * np.sin(plat) - np.sin(lat) * np.cos(plat) * np.cos(z_lamdiff)
    z_sq = np.sqrt(z_a * z_a + z_b * z_b)
    sin_d = z_a / z_sq
    cos_d = z_b / z_sq

    return sin_d, cos_d

