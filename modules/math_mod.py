#!/usr/bin/env python
# coding=utf-8
import numpy as np
import itertools
import sys
from decorators.debugdecorators import TimeThis, PrintArgs
from decorators.paralleldecorators import gmp, ParallelNpArray
from decorators.functiondecorators import requires
#pass data as full, but give c_area in slices.
#race conditions?

#@TimeThis
#@requires()
#@ParallelNpArray(mp)
#def func(ina, inb, ret):
#    """function sceleton"""
#    foo = ina * inb
#    ret = foo
#--------------------
@TimeThis
@requires({
    'full' : ['data'],
    'slice' : ['c_area', 'c_mem_idx', 'ret']
})
@ParallelNpArray(gmp)
def bar_scalar(data, c_area, c_mem_idx, ret):
    '''computes area weighted average of data values,
        over the area specified by c_area and c_mem_idx,
        takes data of kind [[xdata, cell_area, [lat, lon], ...]
        returns numpy'''
    for idx_set, c_a, reti in itertools.izip(c_mem_idx, c_area, ret):
        try:
            idx_set = idx_set[np.where(idx_set > -1)[0]]
            reti[:,] = avg_bar([data[j] for j in idx_set], c_a)
        except:
            sys.exit('Mist idx_set:{}, j:{}'.format(idx_set, j))

#--------------------
#@TimeThis
@requires({
    'full' : ['xdata', 'ydata'],
    'slice' : ['c_area', 'c_mem_idx', 'retx', 'rety']
})
@ParallelNpArray(gmp)
def bar_2Dvector(xdata, ydata, c_area, c_mem_idx, retx, rety):
    #TODO figure out how to have a dual pipe back
    for idx_set, c_a, rxi, ryi in itertools.izip(c_mem_idx, c_area, retx, rety):
        idx_set = idx_set[np.where(idx_set > -1)[0]]
        x_set = [xdata[j] for j in idx_set]
        y_set = [ydata[j] for j in idx_set]
        plon, plat = get_polar(x_set[0][2][0], x_set[0][2][1])
        x_set, y_set = rotate_multiple_to_local(plon, plat, x_set, y_set)
        #after averaging, the latlon and area info becomes obsolete
        rxi[:,] = avg_bar(x_set, c_a)
        ryi[:,] = avg_bar(y_set, c_a)

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
    xsum = np.zeros(data[0][0].shape)
    for xbit in data:
        # multiply data with area
        vals = xbit[0] * xbit[1]
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
@ParallelNpArray(gmp)
def hat_scalar(data, rho, rho_bar, c_area, c_mem_idx, ret):
    for idx_set, rbi, cai, reti in itertools.izip(c_mem_idx, rho_bar, c_area, ret):
        data_set = [data[j] for j in idx_set]
        rho_set = [rho[j] for j in idx_set]
        reti[:,] = avg_hat(data_set, rho_set, rbi, cai)

#--------------------
@TimeThis
@requires({
    'full' : ['xdata', 'ydata', 'rho'],
    'slice' : ['rho_bar', 'c_area', 'c_mem_idx', 'retx', 'rety']
})
@ParallelNpArray(gmp)
def hat_2Dvector(xdata, ydata, rho, rho_bar, c_area, c_mem_idx, retx, rety):
    for idx_set, r_b_i, c_a_i, rxi, ryi in itertools.izip(c_mem_idx, rho_bar, c_area, retx, rety):
        idx_set = idx_set[np.where(idx_set > -1)[0]]
        x_set = [xdata[j] for j in idx_set]
        y_set = [ydata[j] for j in idx_set]
        plon, plat = get_polar(x_set[0][2][0], x_set[0][2][1])
        x_set, y_set = rotate_multiple_to_local(plon, plat, x_set, y_set)
        rho_set = [rho[j] for j in idx_set]
        #after averaging, the latlon and area info becomes obsolete
        rxi[:,] = avg_hat(x_set, rho_set, r_b_i, c_a_i)
        ryi[:,] = avg_hat(y_set, rho_set, r_b_i, c_a_i)

#--------------------
#@TimeThis
def avg_hat(data, rho, rho_bar, c_area):
    """computes the hat average of data,
    requires the average of rho
    takes data of kind
    [[[xdata, cell_area, [lat, lon]], rho], ...]
    returs numpy"""
    #create empty ntim, lev array for summation
    xsum = np.zeros(data[0][0].shape)
    for xbit, rbit in itertools.izip(data, rho):
        # multiply data with area
        vals = xbit[0] * xbit[1]
        # multiply data with density
        vals = np.multiply(vals, rbit[0])
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
    x_flucts = scalar_flucts(x_tnd, x_avg)
    y_flucts = scalar_flucts(y_tnd, y_avg)

    return x_flucts, y_flucts

#--------------------
#@TimeThis
def scalar_flucts(xdata, xavgdata):
    # maintain data structures
    return [[np.subtract(xavgdata, xdat[0]), xdat[1], xdat[2]] for xdat in xdata]

#--------------------
# TODO: scrub data structures out of there.
@TimeThis
@requires({
    'full' : ['x', 'y'],
    'slice' : ['grad_mem_idx', 'grad_coords_rads', 'coarse_area', 'gradient']
})
@ParallelNpArray(gmp)
def xy_2D_gradient(x, y, grad_mem_idx, grad_coords_rads, coarse_area, gradient):
    """computes the gradient of velocity components in u,v using
    the information in gradient_nfo
        u, v are of shape [[data, cell_area, [lon, lat]], ...]
        return: gradient ... manage this somehow

    """
    r_e = 6.37111*10**6

    print(np.shape(grad_mem_idx), np.shape(grad_coords_rads), np.shape(coarse_area), np.shape(gradient))
    for g_idx, g_coordrad, c_area, grad in itertools.izip(grad_mem_idx, grad_coords_rads, coarse_area, gradient):
        neighs_x = []
        neighs_y = []
        for j in range(4): #E, W, N, S contributors
            g_idxj = g_idx[np.where(g_idx > -1)[0]]
            x_set = [x[k] for k in g_idxj]
            y_set = [y[k] for k in g_idxj]
            helper = dist_avg_vec(
                x_set,
                y_set,
                g_coordrad[j]
            )
            neighs_x.append(helper[0])
            neighs_y.append(helper[1])
        d = 2 * radius(c_area) * r_e
        for j in range(2):
            # dx 0: E values - W values,
            # dy 1: N values - S values
            # x component of vector
            grad[j, 0, :,] = central_diff(
                neighs_x[(2*j)],
                neighs_x[(2*j)+1],
                d
            )
            # y component of vector
            grad[j, 1, :,] = central_diff(
                neighs_y[(2*j)],
                neighs_y[(2*j)+1],
                d
            )

#@TimeThis
def dist_avg_vec(x_values, y_values, grid_nfo):
    '''
    does distance weighted average of values within a circular area on a sphere.
    needs:  coordinates of center, indices of area members,
            distances of members from center,'''
    #shift lat lon grid to a local pole.
    plon, plat = get_polar(grid_nfo[0][0], grid_nfo[0][1])
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
    '''computes the distance averaged value of a set of scalar values'''
    factor = 0
    # multiply cell_area and distance from center to recieve weight
    weights = [x_val[1]* x_rad[1] for x_val, x_rad in itertools.izip(x_values, grid_nfo)]
    factor = np.sum(weights)

    average = 0
    for x_val,weight in itertools.izip(x_values, weights):
        average = average + x_val[0] * weight
    return average / factor

def central_diff(xr, xl, d):
    ''' little routine, which computes the
    central difference between two values with distance 2 * d'''
    return np.divide(np.subtract(xr, xl), 2 * d)

def radius(area):
    '''returns radius of circle on sphere in rad'''
    r_e = 6.37111*10**6
    r = np.sqrt(area / np.pi) / r_e
    return r

def get_polar(lon, lat):
    ''' gets the coordinates of a shifted pole, such that
    the given coordinate pair falls onto the equator
    of a transformed lat lon grid'''

    # move pole to the opposite side of earth.
    if 0 < lon <= np.pi:
        plon = lon - np.pi
    elif -np.pi <= lon < 0:
        plon = lon + np.pi
    else:
        plon = np.pi

    if 0 <= lat <= np.pi/2:
        plat = np.pi/2 - lat
    # if point is 'below' the equator, keep plon = lon (90deree angle)
    elif -np.pi/2 <= lat < 0:
        plat = np.pi/2 + lat
        plon = lon

    return plon, plat

def rotate_multiple_to_local(plon, plat, x, y):
    '''wrapper to iterate over multiple values being turned onto the same local grid'''
    len_x = len(x)
    x_tnd = []
    y_tnd = []
    for xi,yi in itertools.izip(x, y):
        out = rotate_vec_to_local(
            plon, plat,
            xi, yi
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
    y_tnd = [x[0] * sin_d + y[0] * cos_d, y[1], y[2]]

    return x_tnd, y_tnd

def rotate_vec_to_global(plon, plat, x, y):
    '''x, y are assumed to be datapoints of the shape:
        [data, cell_area, [lon, lat]]
        returns the data shape
        with data changed according to turn
        '''
    sin_d, cos_d = rotation_jacobian(x[2][0], x[2][1], plon, plat)

    x_tnd = [x[0] * cos_d + y[0] * sin_d, x[1], x[2]]
    y_tnd = [-x[0] * sin_d + y[0] * cos_d, y[1], y[2]]

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

def num_hex_from_rings(num_rings):
    '''computes the number of hexagons in an area made up of
    rings of hexagons around a central hexagon'''
    num_hex = 1 + 6 * num_rings * (num_rings + 1) / 2
    return num_hex

def arc_len(p_x, p_y):
    '''
    computes the length of a geodesic arc on a sphere (in radians)
    p_x : lon, lat
    p_y : lon, lat
    out r in radians
    '''
    try:
        r = np.arccos(
            np.sin(p_x[1]) * np.sin(p_y[1])
            + np.cos(p_x[1]) * np.cos(p_y[1]) * np.cos(p_y[0] - p_x[0])
        )
    except:
        sys.exit('mist: p_x1{}, p_x2{}, p_y1{}, p_y2{}'.format(p_x[0], p_x[1], p_y[0], p_y[1]))
    return r

