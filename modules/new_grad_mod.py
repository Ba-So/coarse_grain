#!/usr/bin/env python
# coding=utf-8
import itertools
import sys
import numpy as np
from decorators.paralleldecorators import gmp, ParallelNpArray, shared_np_array
from decorators.debugdecorators import TimeThis, PrintArgs, PrintReturn
from decorators.functiondecorators import requires
import modules.math_mod as math

#TODO:
# Routine that prepares all the Information for computaiton during runtime:
## for vector orinentation:
### central lon,lat coordinates
## for Interpolation: indices of grid points to interpolate
### distance to interpolate around
### indices of grid points to interpolate
### distances of grid points to center
##############################
##############################
# preparation
##############################
@TimeThis
@requires({
    'full' : ['coords'],
    'slice' : ['coarse_area', 'cell_area', 'cell_idx',
               'grad_coords', 'grad_dist',
               'int_idx', 'int_dist']
})
@ParallelNpArray(gmp)
def compute_coarse_gradient_nfo(coords, coarse_area, cell_area, cell_idx, grad_coords, grad_dist, int_idx, int_dist):
    ''' parallelized preparation of gradient computation
    '''
    i_gd = -1
    for cidx, corea_i, cerea_i, g_coord_i, i_idx_i, i_dist_i in itertools.izip(cell_idx, coarse_area, cell_area, grad_coords, int_idx, int_dist):
        i_gd += 1
        c_coord = coords[cidx]
        # compute gradient coordinates
        distance = 2.0 * math.radius(corea_i)
        grad_dist[i_gd] = distance
        mer_grad = mer_points(c_coord, distance)
        zon_grad = zon_points(c_coord, distance)

        # correct order: (E,W,N,S)(xi, yi)
        g_coord_i[:] = zon_grad[:] + mer_grad[:]
        # compute interpolation_values
        radius = 2 * math.radius(cerea_i)
        for j in range(4):
            idxes, distances = prepare_interpolation(g_coord_i[j], radius, coords)
            i_idx_i[j, :len(idxes)] = idxes
            i_dist_i[j, :len(distances)] = distances

@TimeThis
@requires({
    'full' : ['coords'],
    'slice' : ['coarse_area', 'cell_area', 'cell_idx',
               'grad_coords', 'grad_dist',
               'int_idx', 'int_dist']
})
@ParallelNpArray(gmp)
def compute_fine_gradient_nfo(coords, cell_area, cell_idx, grad_coords, grad_dist, int_idx, int_dist):
    ''' parallelized preparation of gradient computation
    '''
    i_gd = -1
    for cidx, cerea_i, g_coord_i, i_idx_i, i_dist_i in itertools.izip(cell_idx, coarse_area, cell_area, grad_coords, int_idx, int_dist):
        i_gd += 1
        c_coord = coords[cidx]
        # compute gradient coordinates
        distance = 2.0 * math.radius(cerea_i)
        grad_dist[i_gd] = distance
        mer_grad = mer_points(c_coord, distance)
        zon_grad = zon_points(c_coord, distance)

        # correct order: (E,W,N,S)(xi, yi)
        g_coord_i[:] = zon_grad[:] + mer_grad[:]
        # compute interpolation_values
        radius = 2 * math.radius(cerea_i)
        for j in range(4):
            idxes, distances = prepare_interpolation(g_coord_i[j], radius, coords)
            i_idx_i[j, :len(idxes)] = idxes
            i_dist_i[j, :len(distances)] = distances

###############################
# find neighbouring points for gradient.
def zon_points(c_coords, distance):
    ''' computes coordinates of distance along latitude
        to the East and West of c_coords,
        for computation of zonal gradients.
    '''
    east = c_coords[:]
    west = c_coords[:]
    d_lon = get_distance(east[1], distance)
    if d_lon:
        east[0] = meridian_care(east[0] + d_lon)
        west[0] = meridian_care(west[0] - d_lon)
    else:
        # east and west remain the center, thus the gradient turns zero.
        # not very resource saving, but safe enough
        pass
    return [east, west]

def get_distance(lat, distance):
    ''' translates the fraction of a great circle arc into a fraction of a
        small circle arc parallel to equator at lat.
    '''
    # the latitudes are small circles parallel to the equator.
    # the distance a degree of latitude covers therefore depends on its
    # latitude. Namely: dist(lat)[rad] = dist(equator)[rad] / cos(lat)
    # careful: colapses with lat -> +- pi/2
    cos_lat = np.cos(lat)
    # avoid collapse of formula
    if cos_lat <= 0.001:
        # just leave
        return None

    frac = distance / cos_lat
    # we want points to be at least pi apart.
    # therefore we restrict frac to pi/2
    if frac > np.pi/2 :
        # just leave
        return None

    # only if all is passed
    return frac

def meridian_care(lon):
    ''' shifts longitude to appropriate value if overshooting the 180 meridian
        '''
    rlon = lon
    if lon > np.pi:
        rlon = lon - 2 * np.pi
    elif lon < -np.pi:
        rlon = lon + 2 * np.pi

    return rlon

def mer_points(c_coord, distance):
    ''' computes coordinates of distance along longitude
        to North and South of c_coords,
        for computation of meridional gradients.
    '''
    # all meridians are great circles around the globe.
    # distance is compited as a fraction of 2 pi, the circumference of the
    # globe. These points are therefore simple addition/ substraction of
    # distance form c_coord longitude.
    north = c_coord[:] # careful here, [:], so that values are passed not pointer...
    south = c_coord[:]
    north[1] = north[1] + distance
    south[1] = south[1] - distance
    north = pole_care(north)
    south = pole_care(south)

    return [north, south]

def pole_care(coord):
    # edge case: Pole, in case distance overshoots the pole, I want the point
    # to be on its opposite.
    # the longitude has to be changed as well, by np.pi, meridian_care
    # recalibrates.
    rcoord = coord[:]
    if coord[1] > np.pi/2 :
        rcoord[1] = np.pi - coord[1]
        rcoord[0] = meridian_care(rcoord[0] + np.pi)
    elif coord[1] < -np.pi/2 :
        rcoord[1] = -np.pi - coord[1]
        rcoord[0] = meridian_care(rcoord[0] + np.pi)

    return rcoord

###############################
# preparation, for interpolation:
def prepare_interpolation(c_coord, radius, f_coord):
    ''' indentifies f_lon-f_lat pairs in (full)f_coord, within radius around c_lon, c_lat coordinate,
        returns indices of these cells and their geodesic distance to c_lon, c_lat
        '''
    indices = []
    distances = []
    bounds = get_bounds(c_coord, radius)
    for i, f_i in enumerate(f_coord):
        if check_bounds(f_i, bounds):
            distance = math.arc_len(c_coord, f_i)
            if distance <= radius:
                indices.append(i)
                dlon, dlat = math.get_dx_dy(c_coord, f_i)
                distances.append([dlon, dlat])
    return indices, distances

def get_bounds(c_coords, r):
    ''' computes min/max lon/lat for a circle of radius r around [c_lon, c_lat]
        '''
    # lats first, because simple:
    ispole = contains_pole(c_coords[1], r)

    if not ispole:
        # standard case:
        lat_min = c_coords[1] - r
        lat_max = c_coords[1] + r
    elif ispole == 'NP':
        # since this is a search circle, we put it pretty much just ontop of
        # the pole
        lat_min = c_coords[1] - r
        lat_max = np.pi/2
    elif ispole == 'SP':
        # same here:
        lat_min = -np.pi/2
        lat_max = c_coords[1] + r

    lat_bounds = [lat_min, lat_max]

    # lons second:
    lon_min, lon_max = get_tangent_lons(c_coords, r)

    # handle meridian edge cases
    lon_bounds = handle_meridians(lon_min, lon_max)

    return [lon_bounds, lat_bounds]
def get_tangent_lons(coords, r):
    # for these we need to compute the tangent longditudes to a circle on a
    # sphere, see also Bronstein
    # solution breaks down in case either pole is within the circle.
    # safe function
    ratio = np.sin(coords[1]) / np.cos(r)
    epsilon = 0.00001
    if (ratio < 1 - epsilon) & (ratio > - 1 + epsilon):
        # if no pole we get some additional constraints
        lat_t = np.arcsin(np.sin(coords[1]) / np.cos(r))

        d_lon = np.arccos(
            (np.cos(r) - np.sin(lat_t) * np.sin(coords[1]))
            /(np.cos(lat_t) * np.cos(coords[1]))
        )
        lon_min = coords[0] - d_lon
        lon_max = coords[0] + d_lon
    else:
        # we only have lat/lon restraints
        lon_min = -np.pi
        lon_max = np.pi

    return [lon_min, lon_max]

def handle_meridians(lon_min, lon_max):
    lon_bounds = [[],[]]
    if lon_min < -np.pi:
        lon_bounds[0][:] = [-np.pi, lon_max]
        lon_bounds[1][:] = [lon_min + 2 * np.pi, np.pi]
    elif lon_max > np.pi:
        lon_bounds[0][:] = [lon_min, np.pi]
        lon_bounds[1][:] = [-np.pi, lon_max - 2 * np.pi]
    else:
        lon_bounds[0][:] = [lon_min, lon_max]
        lon_bounds[1][:] = [-4, -4]

    return lon_bounds

def contains_pole(lat, r):
    ''' checks if lat +- r goes across a pole, return Np or Sp '''
    ispole = None
    lat_min = lat - r
    lat_max = lat + r
    if lat_max >= np.pi/2 :
        ispole='NP'
    elif lat_min <= -np.pi/2 :
        ispole='SP'

    return ispole

def check_bounds(coords, bounds):
    ''' checks if given lon, lat pair lies within the bounds '''
    # check lats first:
    check = False
    if bounds[1][0] < coords[1] < bounds[1][1]:
        if bounds[0][0][0] < coords[0] < bounds[0][0][1]:
            check = True
        elif bounds[0][1][0] < coords[0] < bounds[0][1][1]:
            check = True

    return check
################################
##############################
# computation
##############################
# x, y contain: [[data(lev,ntime), cell_area, [lon, lat]](ncell)
@TimeThis
@requires({
    'full' : ['x', 'y'],
    'slice' : ['idx', 'grad_coords', 'grad_dist', 'int_mem_idx', 'int_dist', 'gradient']
})
@ParallelNpArray(gmp)
def vector_gradient(x, y, idx, grad_coords, grad_dist, int_mem_idx, int_dist, gradient):
    ''' computes the gradient of a vector (x,y)
    '''
    for idx_i, gc_i, gd_i, imi_i, id_i, grad_i in itertools.izip(idx, grad_coords, grad_dist, int_mem_idx, int_dist, gradient):
        # interpolation of values onto grad_coords[i][j] (E,W,N,S)
        U_vals = [] # (E, W, N, S)
        V_vals = []
        clat, clon = x[idx_i][2] # lat lon info contained in x[i] and y[i]
        for j in range(4):
            # filter dummy values (negative integers)
            imij = clean_indices(imi_i[j])
            # get set members
            x_set = filter_values(x, imij)
            y_set = filter_values(y, imij)
            # and distances
            idij = clean_values(id_i[j], imi_i[j])
            # get coordinates
            c_coord = gc_i[j]
            # do the interpolation
            grad_vals = math.lst_sq_intp_vec(x_set, y_set, c_coord, idij)
            U_vals.append(grad_vals[0])
            V_vals.append(grad_vals[1])

        # get coordinates of pole for equator projection:
        # lat, lon info contained in x and y
        plon, plat = math.get_polar(clat, clon)
        # project onto equator
        U_vals, V_vals = math.rotate_multiple_to_local(
            plon,
            plat,
            U_vals,
            V_vals
        )
        # compute gradient values.
        d = convert_rad_m(gd_i)
        for j in range(2): #dx / dy
            grad_i[j, 0, :,] = math.central_diff(
                U_vals[(2*j)][0],
                U_vals[(2*j)+1][0],
                d
            )[:]
            grad_i[j, 1, :,] = math.central_diff(
                V_vals[(2*j)][0],
                V_vals[(2*j)+1][0],
                d
            )[:]

@TimeThis
@requires({
    'full' : ['x'],
    'slice' : ['grad_coords', 'grad_dist', 'int_mem_idx', 'int_dist', 'gradient']
})
@ParallelNpArray(gmp)
def scalar_gradient(x, grad_coords, grad_dist, int_mem_idx, int_dist, gradient):
    ''' computes the gradient of a scalar
    '''
    for gc_i, gd_i, imi_i, id_i, grad_i in itertools.izip(grad_coords, grad_dist, int_mem_idx, int_dist, gradient):
        # interpolation of values onto grad_coords[i][j] (N,S,E,W)
        X_vals = [] # (N, S, E, W)
        for j in range(4):
            # filter dummy values (negative integers)
            imij = clean_indices(imi_i[j])
            x_set = filter_values(x, imij)
            idij = clean_values(id_i[j], imi_i[j])
            c_coord = gc_i[j]
            grad_vals = math.lst_sq_intp(x_set, c_coord, idij)
            X_vals.append(grad_vals)

        # compute gradient values.
        d = convert_rad_m(gd_i)
        for j in range(2): #dx / dy
            grad_i[j, :,] = math.central_diff(
                X_vals[(2*j)][0],
                X_vals[(2*j)+1][0],
                d
            )[:]

def convert_rad_m(dist):
    ''' converts a distance in radians in meters '''
    r_e = 6.37111*10**6
    U = np.pi * r_e * 2
    partial_U = U * dist / (2 * np.pi)
    if partial_U < 0:
        sys.exit('invalid partial_u')
    return partial_U

def clean_indices(idx):
    ''' ensures only positive values within an array of integers '''
    idc_set = [int(k) for k in idx if k > -1]
    return idc_set

def clean_values(lst, check):
    out = []
    for lst_i, check_i in itertools.izip(lst, check):
        if check_i > -1:
            out.append(lst_i)
    return np.array(out)

def clean_values_old(lst):
    '''removes negative and zero values'''
    return np.array([k for k in lst if np.all(np.less(k,10000.0))])
def filter_values(x, indices):
    ''' returns values of x at indices '''
    return [x[i] for i in indices]















