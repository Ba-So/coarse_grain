#!/usr/bin/env python
# coding=utf-8
import numpy as np
import itertools
import sys
import pprint
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
            idx_set = np.extract(np.greater(idx_set, -1), idx_set)
            reti[:,] = avg_bar([data[j] for j in idx_set], c_a)
        except:
            sys.exit('Mist idx_set:{}'.format(idx_set))

#--------------------
#@TimeThis
@requires({
    'full' : ['xdata', 'ydata'],
    'slice' : ['c_area', 'c_mem_idx', 'retx', 'rety']
})
@ParallelNpArray(gmp)
def bar_2Dvector(xdata, ydata, c_area, c_mem_idx, retx, rety):
    for idx_set, c_a, rxi, ryi in itertools.izip(c_mem_idx, c_area, retx, rety):
        idx_set = np.extract(np.greater(idx_set, -1), idx_set)
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
        idx_set = np.extract(np.greater(idx_set, -1), idx_set)
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

#@TimeThis
def dist_avg_vec(x_values, y_values, grid_nfo):
    '''
    does distance weighted average of values within a circular area on a sphere.
    needs:  coordinates of center, indices of area members,
            distances of members from center,
    '''
    #shift lat lon grid to a local pole.
    plon, plat = get_polar(grid_nfo[0][0], grid_nfo[0][1])
    x_vec, y_vec = rotate_multiple_to_local(
        plon,
        plat,
        x_values,
        y_values
    )
    #turn back to global is unneccesary
    x_vec = dist_avg_scalar(x_vec, grid_nfo)
    y_vec = dist_avg_scalar(y_vec, grid_nfo)

    return x_vec, y_vec

def lst_sq_intp_vec(x_values, y_values, c_coord, mem_dist):
    ''' Prepares least squares interpolation of vectors.
        needs: coordinates of center, indices of area members,
               coordinates of members
     '''
    plon, plat = get_polar(c_coord[0], c_coord[1])
    x_vec = x_values
    y_vec = y_values
    x_vec, y_vec = rotate_multiple_to_local(
        plon,
        plat,
        x_values,
        y_values
    )
    #turn back to global is unneccesary
    x_vec = lst_sq_intp(x_vec, c_coord, mem_dist)
    y_vec = lst_sq_intp(y_vec, c_coord, mem_dist)

    return x_vec, y_vec

# TODO: scrub data structures out of there.
#@TimeThis
def dist_avg_scalar(x_values, grid_nfo):
    '''computes the distance averaged value of a set of scalar values
        Input: x_values[]

    '''
    factor = 0
    # multiply cell_area and distance from center to recieve weight
    weights = [x_val[1]* x_rad for x_val, x_rad in itertools.izip(x_values, grid_nfo[1])]
    factor = np.sum(weights)

    average = 0
    for x_val,weight in itertools.izip(x_values, weights):
        average = average + x_val[0] * weight
    retval = 0.0
    np.seterr(all='raise')
    try:
        retval = average / factor
    except:
        print('avg {}'.format(average))
        print('fac {}'.format(factor))
        sys.exit('tjoar: {}'.format(average))
    return retval

def lst_sq_intp(x_values, c_coord, distances):
    ''' improved method to compute the interpolated value at a specific point
        least squares interpolation
        onto point on sphere
        of values within a circle around that point.
        Using Taylor expansion: f_i = f_p + df_i/dx|p (xi-xp) + df_i/dy|p(yi-yp)
        and least squares A x = y (y={f_i}, A={1, (xi-xp), (yi-yp)}
        for yi-yp we use the fact that even with shifted poles,
        the meridians remain great circles on the sphere.
        for xi-xp we use the formulae for right-angled triangles
        on a sphere and distance d between c and i,
        yielding the great-arc distance between i and c in x.
        (which would otherwise be on small circles)
    '''
    # define dimensions of Problem
    m = len(x_values)
    n = 3
    # define A matrix
    A = np.zeros((m,n))
    lon_c, lat_c = c_coord
    for i in range(m):
        dx_i, dy_i = distances[i]
        A[i,:] = [1, dx_i, dy_i]
    # define b vector

    ntime = np.shape(x_values[0][0])[0]
    u = np.zeros(np.shape(x_values[0][0]))
    dxu = np.zeros(np.shape(x_values[0][0]))
    dyu = np.zeros(np.shape(x_values[0][0]))
    for i in range(ntime):
        b = np.array([x_val[0][i,:] for x_val in x_values])
        u[i,:], dxu[i,:], dyu[i,:] = np.linalg.lstsq(A,b,rcond=None)[0]
    return [u, x_values[0][1], [lon_c, lat_c]]

def triangle_b_from_ac(a,c):
    ''' requires c in radians returns b in m '''
    np.seterr(all='raise')
    r_e = 6.37111*10**6
    z = np.cos(c) / np.cos(a)
    try:
        b = np.arccos(z)
    except:
        print(c)
        print(a)
        print(z)
        b = 0

    return b

def central_diff(xr, xl, d):
    ''' little routine, which computes the
    central difference between two values with distance 2 * d'''
    diff = np.zeros(np.shape(xr))
    epsilon = 500
    if d > 0 + epsilon :
        diff = np.divide(np.subtract(xr, xl), 2 * d)
    return diff

def radius(area):
    '''returns radius of circle on sphere in rad'''
    r_e = 6.37111*10**6
    r = np.sqrt(area / np.pi) / r_e
    return r

def radius_m(area):
    '''returns radius of circle on sphere in rad'''
    r_e = 6.37111*10**6
    r = np.sqrt(area / np.pi)
    return r

def get_polar(lon, lat):
    ''' gets the coordinates of a shifted pole, such that
    the given coordinate pair falls onto the equator
    of a transformed lat lon grid'''

    plat = lat + np.pi/2
    plon = lon
    if plat > np.pi/2 :
        plat = np.pi - plat
        plon = plon + np.pi
    if plon > np.pi:
        # move pole to the opposite side of earth.
        plon = plon - 2 * np.pi

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

    x_tnd = [np.subtract(np.multiply(x[0],cos_d),np.multiply(y[0],sin_d)), x[1], x[2]]
    y_tnd = [np.add(np.multiply(x[0], sin_d), np.multiply(y[0], cos_d)), y[1], y[2]]

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
    ''' returns entries of rotation matrix according to documentation of COSMOS Part I pp. 27'''

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
    ''' computes the length of a geodesic arc on a sphere (in radians)
        using the haversine formula. Is precise for distances smaller than
        half the circumference of earth.
        p_x : lon, lat
        p_y : lon, lat
        out d in radians'''
    dlon = p_y[0] - p_x[0]
    dlat = p_y[1] - p_x[1]
    hav = np.sin(dlat / 2)**2 + np.cos(p_x[1]) * np.cos(p_y[1]) * np.sin(dlon / 2)**2
    d = 2 * np.arcsin(np.sqrt(hav))
    return d

def get_dx_dy(c_coord, o_coord):
    ''' computes dx(along lon) dy(along lat) for interpolation on sphere.
        Uses relative coordinate distance. '''
    # distance on fixed latitude
    dx = 99999
    dy = 99999
    dlon = c_coord[0] - o_coord[0]
    cos_lat = np.cos(c_coord[1])
    if cos_lat <= 0.01:
        dx = 0.0
    else:
        dx = dlon / np.cos(c_coord[1])
    # distance on fixed longditude
    dy = c_coord[1] - o_coord[1]
    return dx, dy






