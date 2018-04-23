import os
import sys
import xarray as xr
import numpy as np
import data_op as dop

def coarse_area(grid):
    '''Sums the coarse_area from the areas of its members'''
    co_ar = np.array([0.0 for i in range(0, grid.dims['ncells'])])
    cell_a = grid['cell_area'].values
    a_nei_idx = grid['area_neighbor_idx'].values
    a_num_hex = grid['area_num_hex'].values

    for i in range(0, grid.dims['ncells']):
        for j in range(0, a_num_hex[i]):
            ij = a_nei_idx[j, i]
            co_ar[i] += cell_a[ij]

    coarse_a = xr.DataArray(
        co_ar,
        dims=['ncells'])
    kwargs = {'coarse_area' : coarse_a}
    grid = grid.assign(**kwargs)

    return grid

def area_avg(kind, grid_nfo, data, var, vec=False):
    '''computes the bar average over all vars (multiplied)'''
    name = []
    kwargs = {}

    if not vec:
        len_vec = 1
        name.append('')
        for va in var['vars']:
            name[0] = name[0]+va
        name[0] = name[0]+'_'+kind
    else:
        len_vec = len(var['vector'])
        name = var['vector'][:]
        for i, item in enumerate(name):
            name[i] = name[i]+'_'+kind

    if kind == 'hat':

        func = lambda val, fac, kwargs: avg_hat(val, fac, **kwargs)
        kwargs = {
            'ntim': grid_nfo['ntim'],
            'nlev': grid_nfo['nlev']
            }
        # fatal: RHO missing
        if not 'RHO' in data:
            sys.exit('ERROR: area_avg kind = "hat": missing "RHO"')

        if not 'RHO' in var['vars']:
            var['vars'].append('RHO')

        # semi fatal: RHO_bar missing
        if not 'RHO_bar' in data:
            data = area_avg('bar', grid_nfo, data, {'vars':['RHO']})

        factor = {}
        factor['RHO_bar'] =data['RHO_bar']
        factor['coarse_area'] = grid_nfo['coarse_area']
        factor =  mult(factor)

    elif kind == 'bar':
        func = lambda val, fac, kwargs: avg_bar(val, fac, **kwargs)
        factor = grid_nfo['coarse_area']

    else:
        print 'ERROR area_avg: unknown averaging type'
        return None

    if vec:
        if not all(key in grid_nfo for key in ['lat', 'lon']):
            sys.exit('ERROR: avg_area kind kind = vec, missing "lat" or "lon"')
        kwargs = {
            'i_cell'   : 0,
            'grid_nfo' : grid_nfo,
            'vec'      : var['vector'],
            'func'     : func,
            'kwargs'   : kwargs
            }
        func = lambda val, fac, kwargs: avg_vec(val, fac, **kwargs)
    dims = [len_vec, grid_nfo['ntim'], grid_nfo['nlev'], grid_nfo['ncells']]
    stack = np.empty(dims, order='F')

    #create kwargs for get_members:
    for i in range(0, grid_nfo['ncells'] ):
        values =  dop.get_members(grid_nfo, data, i, var['vars'])
        values.update(dop.get_members(grid_nfo, grid_nfo, i, ['cell_area']))
        # divide by factor (area or rho_bar)
        kwargs['i_cell'] = i
        values = func(values, factor, kwargs)
        stack[:,:,:,i] = values
    # put into xArray
    for i in range(len_vec):
        data[name[i]] = stack[i,:,:,:]
    return data

# call functions for area_avg
# -----
def avg_bar(values, factor, i_cell):
    # multiply var area values (weighting)
    values=  mult(values)
    # Sum Rho*var(i) up
    values=  np.sum(values,0)
    return values/factor[i_cell]

def avg_hat(values, factor, i_cell, ntim, nlev):
    # multiply var area values (weighting)
    values=  mult(values)
    # Sum Rho*var(i) up
    values=  np.sum(values,0)
    for k in range(ntim):
        for j in range(nlev):
            values[k,j] = values[k,j]/factor[k,j,i_cell]
    return values
    # -----

def avg_vec(values, factor, i_cell, grid_nfo, vec, func, kwargs):
    # from here on we stay in local_coordinates. Only revert back after all the
    # operations have been performed.
    kwargs['i_cell'] = i_cell
    coords =  dop.get_members(grid_nfo, grid_nfo, i_cell, ['lat','lon'])
    # rotating vectors.
    rot_vec = np.empty([len(vec),
        grid_nfo['area_num_hex'][i_cell],
        grid_nfo['ntim'],
        grid_nfo['nlev']
        ])
    rot_vec.fill(0)
    rot_vec[:,:,:,:]  = rotate_ca_members_to_local(
        coords['lon'], coords['lat'],
        values[vec[0]][:,:,:], values[vec[1]][:,:,:]
        )
    # splitting dictionary up
    # magic that:
    help_dic  = {}

    for key in values:
        if key not in vec:
            help_dic[key] = values[key]
    values_vec = []
    for j, item  in enumerate(vec):
        values_vec.append({vec[j] : rot_vec[j,:,:,:]})
        values_vec[j].update(help_dic)

  # computing averages
    helper = np.empty([len(vec),grid_nfo['ntim'],grid_nfo['nlev']])
    for j, item  in enumerate(vec):
        # func is either avg_hat or avg_bar
        helper[j, :, :]  = func(values_vec[j], factor, kwargs)
    helper[:, :, :] = rotate_single_to_global(
        coords['lon'][0], coords['lat'][0],
        helper[0, :, :], helper[1, :, :])
    return helper

def compute_flucts(values, grid_dic, num_hex, vars, kind):
    '''computes the fluctuations relative to the hat quantities.'''

    kwargs = {
        'num_hex' : num_hex,
        'values' : values,
        'grid_dic': grid_dic,
        'vars'    : vars
        }
    if kind == 'vec':
    #  func = ...
        func= lambda kwargs: vector_flucts(**kwargs)
    elif kind == 'scalar':
    #  func = ...
        func= lambda kwargs: scalar_flucts(**kwargs)
    else:
        sys.exit('ERROR: compute_flucts not ia valid "kind" {}').format(kwargs['kind'])

    values = func(kwargs)

    return values

def scalar_flucts(values, grid_dic, num_hex, vars):

    result    = np.empty([num_hex, grid_dic['ntim'], grid_dic['nlev']])
    for i in range(len(vars) ):
        result.fill(0)
        for j in range(num_hex):
            result[j,:,:]     = values[vars[i]][ j, :, :] - values[vars[i]+'_bar'][:,:]
        values[vars[i]+'_f'] = result

    return values

def vector_flucts(values, grid_dic, num_hex, vars):
    rot_vec = np.empty([len(vars),num_hex, grid_dic['ntim'], grid_dic['nlev']])
    rot_vec.fill(0)
    rot_vec[:, :,:,:]  = rotate_ca_members_to_local(
        values['lon'], values['lat'],
        values[vars[0]][:, :, :], values[vars[1]][:, :, :])
    rot_bar = np.empty([len(vars), grid_dic['ntim'], grid_dic['nlev']])
    rot_bar.fill(0)
    rot_bar[:, :, :] =rotate_single_to_local(
        values['lon'][0], values['lat'][0],
        values[vars[0] + '_bar'][:, :],
        values[vars[1] + '_bar'][:, :])
    result    = np.empty([
        len(vars), num_hex, grid_dic['ntim'], grid_dic['nlev']
        ])
    result.fill(0)
    for i in range(len(vars) ):
        for j in range(num_hex): # a bit ad hoc
            result[i, j, :, :]     = rot_vec[i, j, :, :] - rot_bar[i, :, :]

    for i in range(len(vars)):
        values[vars[i]+'_f'] = result[i, :, :, :]

    return values

def compute_dyads(values, grid_nfo, i_cell, vars):
    '''computes the dyadic products of v'''
    if not(all(key in values for key in ['RHO'])):
        sys.exit('ERROR: compute_dyads, "RHO" missing in values')
    if not(all(key in grid_nfo for key in ['cell_area'])):
        sys.exit('ERROR: compute_dyads, "cell_area" missing in grid_nfo')

    # dimension of dyad
    l_vec   = len(vars)

    if not('dyad' in values):
        # in case of first call build output file
        # output slot, outgoing info
        values['dyad']        = np.empty([
            l_vec,
            l_vec,
            grid_nfo['ntim'],
            grid_nfo['nlev']
            ])
  # the product of dyadic multiplication (probably term dyadic is misused here)
    product = np.empty([
            l_vec,
            l_vec,
            grid_nfo['area_num_hex'][i_cell],
            grid_nfo['ntim'],
            grid_nfo['nlev']
            ])

    product.fill(0)
    # helper, containing the constituents for computation
    constituents = []
    for var in vars:
        constituents.append(values[var])
    # helper as handle for avg_bar, values in it are multiplied and averaged over
    helper                = {}
    helper['cell_area']   = values['cell_area']
    helper['RHO']         = values['RHO']

    for i in range(l_vec):
        for j in range(l_vec):
            product[i,j,:,:,:] = constituents[i] * constituents[j]
    for i in range(l_vec):
        for j in range(l_vec):
            helper['product']    = product[i,j,:,:,:]
            values['dyad'][i,j,:,:] = avg_bar(
                helper,
                grid_nfo['coarse_area'],
                i_cell)
    return values['dyad']

def gradient(data, grid_nfo, gradient_nfo, var):

    data['gradient']  = np.empty([
        2, 2,
        grid_nfo['ntim'], grid_nfo['nlev'], grid_nfo['ncells']
        ])
    data['gradient'].fill(0)

    info_bnd = 0
    for i in range(grid_nfo['ncells']):
        # chekc for correct key
        # define distance
        # assume circle pi r^2 => d= 2*sqrt(A/pi)
        if i == info_bnd:
            print('progress: {} cells of {}').format(info_bnd, grid_nfo['ncells'])
            info_bnd = info_bnd + 1000

        # check for correct keys
        neighs  = circ_dist_avg(data, grid_nfo, gradient_nfo, i, var)
        area    = grid_nfo['cell_area'][i]
        d       = 2 * radius(area)
        data['gradient'][0,0,:,:,i]   = central_diff(
            neighs['U'][0], data['U'][:, :, i], neighs['U'][1],
            d
            )
        data['gradient'][0,1,:,:,i]  = central_diff(
            neighs['V'][0], data['V'][:, :, i], neighs['V'][1],
            d
            )
        data['gradient'][1,0,:,:,i]   = central_diff(
            neighs['U'][2], data['U'][:, :, i], neighs['U'][3],
            d
            )
        data['gradient'][1,1,:,:,i]   = central_diff(
            neighs['V'][2], data['V'][:, :, i], neighs['V'][3],
            d
            )

    # find/interpolate value at distance d in x/y direction
    #     -> use geodesics / x/y along longditudes/latitudes
    # turn values at distance (with rot_vec)
    # compute d/dz use fassinterpol x_i-1+2x_i+x_i+1/dx including grid value?
    return data

def central_diff(xl, x, xr, d):
        # no turning necessary here since gradients are along lats / longs

        return (xl-2*x+xr)/d**2

def radius(area):
    '''returns radius of circle on sphere in radians'''
    re        = 6.37111*10**6
    r         = np.sqrt(area/np.pi)/ re
    return r

def max_min_bounds(lonlat, r):
    # get radius
    # coords[0,:] latitude values
    # coords[1,:] longditude values
    # u is merdional wind, along longditudes
    #     -> dx along longditude
    # v is zonal wind, along latitudes
    #     -> dy along latitude
    # coords[:,:2] values for dx
    # coords[:,:2] values for dy
    # computing minimum and maximum latitudes
    lock = False
    lat_min = lonlat[1] - r
    lat_max = lonlat[1] + r
    bounds = np.empty([2, 2, 2])
    bounds.fill(-4)
    if lat_max > np.pi/2:
        # northpole in query circle:
        bounds[0, :, :] = [[lat_min, -np.pi],[np.pi/2, np.pi]]
        lock = True
    elif lat_min < -np.pi/2:
        # southpole in query circle:
        bounds[0, :, :] = [[-np.pi/2, -np.pi],[lat_max, np.pi]]
        lock = True
    else:
        # no pole in query circle
        bounds[0, :, :] = [[lat_min, -np.pi], [lat_max, np.pi]]
  # computing minimum and maximum longditudes:
    if not lock:
        lat_T = np.arcsin(np.sin(lonlat[1])/np.cos(r))
        d_lon = np.arccos(
            (np.cos(r) - np.sin(lat_T) * np.sin(lonlat[1]))
            /(np.cos(lat_T)*np.cos(lonlat[1])))
        lon_min = lonlat[0] - d_lon
        lon_max = lonlat[0] + d_lon

        if lon_min < -np.pi:
            bounds[1, :, :] = bounds[0, :, :]
            bounds[0, 0, 1] = lon_min + 2 * np.pi
            bounds[0, 1, 1] = np.pi
            #and
            bounds[1, 0, 1] = - np.pi
            bounds[1, 1, 1] = lon_max
        elif lon_max > np.pi:
            bounds[1, :, :] = bounds[0, :, :]
            bounds[0, 0, 1] = lon_min
            bounds[0, 1, 1] = np.pi
            #and
            bounds[1, 0, 1] = - np.pi
            bounds[1, 1, 1] = lon_max - 2 * np.pi
        else:
            bounds[0, 0, 1] = lon_min
            bounds[0, 1, 1] = lon_max

    return bounds

def gradient_coordinates(lonlat, area):

    r = 2 * radius(area)
    lat_min = lonlat[1] - r
    lat_max = lonlat[1] + r
    if lat_max > np.pi/2:
        # northpole in query circle:
        lat_max = lat_max - np.pi
    elif lat_min < -np.pi/2:
        # southpole in query circle:
        lat_min = lat_min + np.pi
    r = r / np.cos(lonlat[1])
    lon_min = lonlat[0] - r
    lon_max = lonlat[0] + r
    if lon_min < -np.pi:
        lon_min = lon_min + np.pi
    elif lon_max > np.pi:
        lon_max = lon_max - np.pi

    coords = np.array([
        np.array([lon_min, lat_min]),
        np.array([lon_min, lat_max]),
        np.array([lon_max, lat_min]),
        np.array([lon_max, lat_max])
        ])
    return coords


def circ_dist_avg(data, grid_nfo, gradient_nfo, i_cell, var):
    values = {
        name: np.empty(
            [4,grid_nfo['ntim'],grid_nfo['nlev']])
            for name in var['vars']}
    for name in var['vars']:
        values[name].fill(0)
    #how large is check radius?
    for j in range(4):
        coords = gradient_nfo['coords'][i_cell, j, :]
        member_idx = gradient_nfo['member_idx'][i_cell][j]
        member_rad = gradient_nfo['member_rad'][i_cell][j]
    # compute distance weighted average of area weighted members:
        members    = dop.get_members_idx(data, member_idx, var['vars'])
        # if you suffer from boredom: This setup gives a lot of flexebility.
        #     may be transferred to other parts of this program.
        if 'vector' in var.iterkeys():
            #turn vectors
            lats = np.array([grid_nfo['lat'][i] for i in member_idx])
            lons = np.array([grid_nfo['lon'][i] for i in member_idx])
            vec   = var['vector']
            rot_vec = np.empty([
                len(vec),
                len(member_idx),
                grid_nfo['ntim'],
                grid_nfo['nlev']])
            rot_vec.fill(0)
            plon, plat = get_polar(coords[0], coords[1])
            rot_vec[:,:,:,:]  =  rotate_members_to_local(
                lons, lats, plon, plat,
                members[vec[0]][:,:,:], members[vec[1]][:,:,:])
            members[vec[0]]   = rot_vec[0,:,:,:]
            members[vec[1]]   = rot_vec[1,:,:,:]
        helper         = dist_avg(members, member_idx, member_rad, grid_nfo, var['vars'])
        if 'vector' in var.iterkeys():
            rot_vec =rotate_single_to_global(
                coords[0], coords[1],
                helper[vec[0]][:,:], helper[vec[1]][:,:])
            helper[vec[0]]   = rot_vec[0]
            helper[vec[1]]   = rot_vec[1]

        for name in var['vars']:
            values[name][j]   = helper[name]

    return values

def dist_avg(members, idxcs, radii, grid_nfo, vars):
    len_i  = len(idxcs)
    # define weights.
    weight = np.empty([len_i])
    weight.fill(0)
    factor = 0
    for i in range(len_i):
        weight[i] = grid_nfo['cell_area'][idxcs[i]]* radii[i]
        factor    = factor + weight[i]
    # compute distance averages.
    sum    = {name : 0 for name in vars}
    for k in vars:
        for i in range(len_i):
            sum[k]    = sum[k] + members[k][i]*weight[i]
        sum[k]    = sum[k]/factor

    return sum

def arc_len(p_x, p_y):
    '''computes length of geodesic arc on a sphere (in rad)'''
    r = np.arccos(
        np.sin(p_x[1])* np.sin(p_y[1])
       +np.cos(p_x[1])* np.cos(p_y[1]) * np.cos(p_y[0]-p_x[0])
       )
    return r

def turn_spherical(lonlat, grid_nfo):
    lons  = grid_nfo['lon']
    lats  = grid_nfo['lat']
    n_lats = np.empty(len(lats))
    n_lats.fill(0.0)
    n_lons = np.empty(len(lats))
    n_lats.fill(0.0)
    turn_by   = np.empty([2])
    turn_by[:]= lonlat[:]
    turn_by   = turn_by* -1
    n_lons    = np.arctan2(
        np.sin(lons) , np.tan(lats) * np.sin(turn_by[1])
        + np.cos(lons) * np.cos(turn_by[1])) - turn_by[0]

    n_lats    = np.arcsin(
        np.cos(turn_by[1]) * np.sin(lats)
        - np.sin(turn_by[1]) * np.cos(lats) * np.cos(lons))

    for i in range(len(n_lons)):
        if not (-np.pi <= n_lons[i] <= np.pi):
            if n_lons[i] < -np.pi:
                n_lons[i]   = n_lons[i]+2*np.pi
            if np.pi < n_lons[i]:
                n_lons[i]   = n_lons[i]-2*np.pi

    return n_lons, n_lats

def mult(dataset):
    dims= np.array([])
    for val in dataset.itervalues():
        if len(val.shape) >=len(dims):
            dims = val.shape

    helper = np.empty(dims)
    helper.fill(1)
    for data in dataset.itervalues():
        if data.shape == helper.shape:
            helper = np.multiply(helper, data)
        elif data.ndim == 1:
            if data.shape[0] == helper.shape[0]:
                    for i in range(data.shape[0]):
                        helper[i,:,:] = helper[i,:,:]*data[i]
            elif data.shape[0] == helper.shape[1]:
                    for i in range(data.shape[0]):
                        helper[:,i,:] = helper[:,i,:]*data[i]
            elif data.shape[0] == helper.shape[2]:

                    for i in range(data.shape[0]):
                        helper[:,:,i] = helper[:,:,i]*data[i]
            else:
                    print 'I can not find a way to cast {} on {}'.format(data.shape,
                        helper.shape)
    return helper


# see COSMO DynamicsNumerics description
#........................phi|................................
#...........................|................................
#...........................|                          ///
#...........................|                         ///
#\                          |                       ///
#\\\  phi'                  |                     /// lam'
#..\\\                      |                   ///
#   \\\                     |                 ///
#     \\\                   |            .  ///
#       \\\                 |             ///
#         \\\               |           ///
#           \\\             |         ///
#             \\\           |       ///.
#               \\\         |     ///  ..
#                 \\\       |    //     ..                  .
#                   \\\     |  ///      ..
#                     \\\   |///  d      ..
#                       \\\.|//          ..               lam
#____________________________________________________________
#
#      |cos(d) -sin(d) 0|
#  P = |sin(d)  cos(d) 0|
#      |0       0      1|

#e_i= Sum_n(P_in*e'_n) & e'_i= Sum_n(P_ni* e_n)
#A_i= Sum_n(P_in*A'_n) & A'_i= Sum_n(P_ni* A_n)
#Where e the unit vectors and A_i the physical components of vector A
#specifically:
# u= u_g* cos_d -v_g*sin_d
# v= u_g* sin_d -v_g*cos_d
# and
# u_g= u* cos_d +v*sin_d
# v_g=-u* sin_d +v*cos_d
# Where the subscript _g defines the geographic grid.
def rotate_ca_members_to_local(lon, lat, x, y):
    # takes in the array of number of hexagons
    #, where x and y contain [num_hex, ntim, nlev] values
    len_x  = x.shape[0]
    x_tnd  = np.empty(x.shape)
    y_tnd  = np.empty(y.shape)
    x_tnd.fill(0)
    y_tnd.fill(0)
    plon, plat = get_polar(lon[0],lat[0])
    for i in range(len_x):
            x_tnd[i,:,:], y_tnd[i,:,:]= rotate_vec_to_local(
                plon, plat,
                lon[i], lat[i],
                x[i,:,:], y[i,:,:]
                )
    return x_tnd, y_tnd

def rotate_members_to_local(lon, lat, plon, plat, x, y):
    # takes in the array of number of hexagons
    #, where x and y contain [num_hex, ntim, nlev] values
    len_x  = x.shape[0]
    x_tnd  = np.empty(x.shape)
    y_tnd  = np.empty(y.shape)
    x_tnd.fill(0)
    y_tnd.fill(0)
    for i in range(len_x):
            x_tnd[i,:,:], y_tnd[i,:,:]= rotate_vec_to_local(
                plon, plat,
                lon[i], lat[i],
                x[i,:,:], y[i,:,:]
                )
    return x_tnd, y_tnd

def rotate_single_to_local(lon, lat, x, y):
    # shouldn't be neccessary. The entropy production value should remain the same
    # independent rotation back should therefore not be neccessary and additional
    # computing time.

    plon, plat = get_polar(lon, lat)
    x_tnd, y_tnd  = rotate_vec_to_local(
        plon, plat,
        lon, lat,
        x, y)
    return x_tnd, y_tnd

def rotate_ca_members_to_global(lon, lat, x, y):
    # takes in the array of number of hexagons
    #, where x and y contain [num_hex, ntim, nlev] values
    len_x  = x.shape[0]
    x_tnd  = np.empty(x.shape)
    y_tnd  = np.empty(y.shape)
    x_tnd.fill(0)
    y_tnd.fill(0)
    plon, plat = get_polar(lon[0],lat[0])
    for i in range(len_x):
            x_tnd[i,:,:], y_tnd[i,:,:]= rotate_vec_to_global(
                plon, plat,
                lon[i], lat[i],
                x[i,:,:], y[i,:,:])
    return x_tnd, y_tnd

def rotate_members_to_global(lon, lat, plon, plat, x, y):
    # takes in the array of number of hexagons
    #, where x and y contain [num_hex, ntim, nlev] values
    len_x  = x.shape[0]
    x_tnd  = np.empty(x.shape)
    y_tnd  = np.empty(y.shape)
    x_tnd.fill(0)
    y_tnd.fill(0)
    for i in range(len_x):
            x_tnd[i,:,:], y_tnd[i,:,:]= rotate_vec_to_global(
                plon, plat,
                lon[i], lat[i],
                x[i,:,:], y[i,:,:])
    return x_tnd, y_tnd


def rotate_single_to_global(lon, lat, x, y):
    # shouldn't be neccessary. The entropy production value should remain the same
    # independent rotation back should therefore not be neccessary and additional
    # computing time.

    plon, plat = get_polar(lon, lat)
    x_tnd, y_tnd  = rotate_vec_to_global(
        plon, plat,
        lon, lat,
        x, y)
    return x_tnd, y_tnd

def get_polar(lon, lat):
    plon   = 0.0
    if 0 < lon <= np.pi:
        plon = lon - np.pi
    elif (-np.pi < lon < 0):
        plon = lon + np.pi
    #else:
        # no turning needed here

    plat   = np.pi/2
    if 0 < lat <= np.pi/2:
        plat = np.pi/2 - lat
    elif -np.pi/2 <= lat < 0:
        plat = -np.pi/2-lat
    #else:
        # no turning needed here

    return plon, plat


def rotate_vec_to_local(plon, plat, lon, lat, x,y):
    ''' rotates vectors using rotat_latlon_vec'''
    x_tnd= np.empty(x.shape)
    y_tnd= np.empty(y.shape)
    x_tnd.fill(0)
    y_tnd.fill(0)

    sin_d, cos_d = rotate_latlon_vec(lon,lat,plon,plat)

    x_tnd = x*cos_d -y*sin_d
    y_tnd = x*sin_d +y*cos_d

    return x_tnd, y_tnd

def rotate_vec_to_global(plon, plat, lon, lat, x,y):
    ''' rotates vectors using rotate_latlon_vec'''
    x_tnd= np.empty(x.shape)
    y_tnd= np.empty(y.shape)
    x_tnd.fill(0)
    y_tnd.fill(0)

    sin_d, cos_d = rotate_latlon_vec(lon,lat,plon,plat)

    x_tnd = x*cos_d +y*sin_d
    y_tnd =-x*sin_d +y*cos_d

    return x_tnd, y_tnd

def rotate_latlon_vec(lon, lat, plon, plat):
    '''gives entries of rotation matrix for vector rotation
        see Documentation of COSMOS pp. 27'''

    z_lamdiff = lon - plon
    z_a       = np.cos(plat)*np.sin(z_lamdiff)
    z_b       = np.cos(lat)*np.sin(plat)-np.sin(lat)*np.cos(plat)*np.cos(z_lamdiff)
    z_sq      = np.sqrt(z_a*z_a + z_b*z_b)
    sin_d     = z_a/z_sq
    cos_d     = z_b/z_sq

    return sin_d, cos_d

