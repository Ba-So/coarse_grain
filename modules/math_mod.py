#!/usr/bin/env python
# coding=utf-8
import numpy as np
import itertools
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
    'slice' : ['c_area', 'c_mem_idx', 'ret']
})
@ParallelNpArray(mp)
def bar_scalar(data, c_area, c_mem_idx, ret):
    '''computes area weighted average of data values,
        over the area specified by c_area and c_mem_idx,
        takes data of kind [[xdata, cell_area, [lat, lon], ...]
        returns numpy'''
    for i, idx_set in c_mem_idx:
        ret[i] = avg_bar([data[j] for j in idx_set], c_area[i])

#--------------------
#@TimeThis
@requires({
    'full' : ['data'],
    'slice' : ['c_area', 'c_mem_idx', 'retx', 'rety']
})
@ParallelNpArray(mp)
def bar_2Dvector(xdata, ydata, c_area, c_mem_idx, retx, rety):
    #TODO figure out how to have a dual pipe back
    for i, idx_set in c_mem_idx:
        x_set = [xdata[j] for j in idx_set]
        y_set = [ydata[j] for j in idx_set]
        plon, plat = get_polar(x_set[0][2][0], x_set[0][2][1])
        x_set, y_set = rotate_multiple_to_local(plon, plat, x_set, y_set)

        #after averaging, the latlon and area info becomes obsolete
        retx[i,] = avg_bar(x_set, c_area[i])
        rety[i,] = avg_bar(y_set, c_area[i])
#--------------------
#@TimeThis
def avg_bar(data, c_area):
    """computes the bar average of data
        data - needs to be full stack
        c_area - can be part
        takes data of shape
        [[xdata(numpy), cell_area(float), [lon, lat], ..]
        returns numpy array of xdata.shape"""
    #create empty ntim, lev array for summation
    xsum = np.zeros(data[0][0].shape())
    for i, xbit in enumerate(data):
        # multiply data with area
        vals = np.multiply(xbit[0], xbit[1], 0)
        # add that to the sum of all
        xsum = np.add(xsum, vals)
    # divide by total area & return
    return np.divide(xsum, c_area)
#--------------------
#@TimeThis
@requires({
    'full' : ['data', 'rho'],
    'slice' : ['rho_bar', 'c_area', 'c_mem_idx', 'ret']
})
@ParallelNpArray(mp)
def hat_scalar(data, rho, rho_bar, c_area, c_mem_idx, ret):
    for i, idx_set in c_mem_idx:
        data_set = [[data[j], rho[j]] for j in idx_set]
        rho_set = [rho[j] for j in idx_set]
        ret[i] = avg_hat(data_set, rho_set, rho_bar[i], c_area[i])

#--------------------
#@TimeThis
@requires({
    'full' : ['xdata', 'ydata', 'rho'],
    'slice' : ['rho_bar', 'c_area', 'c_mem_idx', 'retx', 'rety']
})
@ParallelNpArray(mp)
def hat_2Dvector(xdata, ydata, rho, rho_bar, c_area, c_mem_idx, retx, rety):
    for i, idx_set in c_mem_idx:
        x_set = [xdata[j] for j in idx_set]
        y_set = [ydata[j] for j in idx_set]
        x_set, y_set = rotate_multiple_to_local(x_set, y_set)
        rho_set = [rho[j] for j in idx_set]
        #after averaging, the latlon and area info becomes obsolete
        retx[i,] = avg_hat(x_set, rho_set, rho_bar[j], c_area[i])
        rety[i,] = avg_hat(y_set, rho_set, rho_bar[j], c_area[i])

#--------------------
#@TimeThis
def avg_hat(data, rho, rho_bar, c_area):
    """computes the hat average of data,
    requires the average of rho
    takes data of kind
    [[[xdata, cell_area, [lat, lon]], rho], ...]
    returs numpy"""
    #create empty ntim, lev array for summation
    xsum = np.zeros(data[0][0].shape())
    for i, xbit in enumerate(data):
        # multiply data with area
        vals = np.multiply(xbit[0], xbit[1], 0)
        # multiply data with density
        vals = np.multiply(vals, rho[i], 0)
        # add that to the sum of all
        xsum = np.add(xsum, vals)
    # divide by rho_bar(numpy) * total area(scalar) & return
    return np.divide(xsum, np.multiply(rho_bar, c_area))

#--------------------
# TODO: scrub data structures out of there.
#@TimeThis
def vec_flucts(x_data, y_data, x_avg, y_avg):
    """computes deviations from local mean"""
    plon, plat = get_polar(x_data[0][2][0], x_data[0][2][1])
    x_tnd, y_tnd = rotate_multiple_to_local(plon, plat, x_data, y_data)
    x_avg = scalar_flucts(x_tnd, x_avg)
    y_avg = scalar_flucts(y_tnd, y_avg)

    return x_avg, y_avg

#--------------------
# TODO: scrub data structures out of there.
#@TimeThis
def scalar_flucts(xdata, avg_data):
    return np.array([avg_data - x_dat[0] for x_dat in xdata])

#--------------------
# TODO: scrub data structures out of there.
#@TimeThis
@requires({
    'full' : ['u', 'v', 'grid_nfo'],
    'slice' : ['gradient_nfo', 'gradient']
})
@ParallelNpArray(mp)
def uv_2D_gradient(u, v, grad_mem_idx, grad_coords_rads, coarse_area, gradient):
    """computes the gradient of velocity components in u,v using
    the information in gradient_nfo
        u, v are of shape [[data, cell_area, [lon, lat]], ...]
        return: gradient

    """
    r_e = 6.37111*10**6
    for g_idx, g_coordrad, c_area in itertools.izip(grad_mem_idx, grad_coords_rads, coarse_area):
        neighs_u = []
        neighs_v = []
        for j in range(4):
            x_set = [u[k] for k in g_idx[j]]
            y_set = [u[k] for k in g_idx[j]]
            # This is broken and wrong
            helper = dist_avg_vec(
                x_set,
                y_set,
                g_coordrad[j]
            )
            neighs_u.append(helper[0])
            neighs_v.append(helper[1])
        d = 2 * radius(c_area) * r_e
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

# TODO: scrub data structures out of there.
#@TimeThis
def dist_avg_vec(x_values, y_values, grid_nfo):
    '''
    does distance weighted average of values within a circular area on a sphere.
    needs:  coordinates of center, indices of area members,
            distances of members from center,'''
    #shift lat lon grid to a local pole.
    plon, plat = get_polar(grid_nfo[0][1], grid_nfo[0][1])
    x_vec, y_vec = rotate_multiple_to_local(
        plon,
        plat,
        x_values,
        y_values,
    )
    #turn back to global is unneccesary
    x_vec = dist_avg_scalar(x_vec, grid_nfo)
    y_vec = dist_avg_scalar(y_vec, grid_nfo)

    return x_vec, y_vec


# TODO: scrub data structures out of there.
#@TimeThis
def dist_avg_scalar(x_values, grid_nfo):
    factor = 0
    # multiply cell_area and distance from center to recieve weight
    weights = [x_val[1]* x_rad for x_val, x_rad in itertools.izip(x_values, grid_nfo[1])]
    factor = np.sum(weights)

    average = 0
    for x_val,weight in itertools.izip(x_values, weights):
        average = average + x_values[0] * weight
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

def rotate_multiple_to_local(plon, plat, x, y):
    '''wrapper to iterate over multiple values being turned onto the same local grid'''
    len_x = len(x)
    x_tnd = []
    y_tnd = []
    for xi,yi in itertools.izip(x, y):
        out = rotate_vec_to_local(
            plon, plat,
            x, y
        )
        x_tnd.append(out[0])
        y_tnd.append(out[1])
    return x_tnd, y_tnd

def rotate_vec_to_local(plon, plat, x, y):
    '''x, y are assumed to be datapoints of the shape:
        [data, cell_area, [lon, lat]]
        returns the data shape
        with data changed according to turn
        '''
    sin_d, cos_d = rotation_jacobian(x[2][0], x[2][1], plon, plat)

    x_tnd = [x[0] * cos_d - y[0] * sin_d, x[1], x[2]]
    y_tnd = [x[0] * cos_d + y[0] * sin_d, y[1], y[2]]

    return x_tnd, y_tnd

def rotation_jacobian(lon, lat, plon, plat):
    ''' returns entries of rotation matrix according to documentation of COSMOS pp. 27'''

    z_lamdiff = lon - plon
    z_a = np.cos(plat) * np.sin(z_lamdiff)
    z_b = np.cos(lat) * np.sin(plat) - np.sin(lat) * np.cos(plat) * np.cos(z_lamdiff)
    z_sq = np.sqrt(z_a * z_a + z_b * z_b)
    sin_d = z_a / z_sq
    cos_d = z_b / z_sq

    return sin_d, cos_d

