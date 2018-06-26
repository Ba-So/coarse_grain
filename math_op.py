import sys
import numpy as np
import data_op as do
import data_op_p as dop
import math_op_p as mop
import global_vars as gv
import update as up
from multiprocessing import Pool


# changing data structure in area_avg
def area_avg(kind, var):
# the no return structure could be hindering for multiprocessing
    '''computes the bar average over all vars (multiplied)'''
    update = up.Updater()
    name = []

    # prepare names:
    name = []
    for item in var['vars']:
        name.append(item[0] + '_' + kind)

    print name

    # in case of vector averaging:

    dims = [
        gv.globals_dict['grid_nfo']['ncells'],
        len(var['vars']),
        gv.globals_dict['grid_nfo']['ntim'],
        gv.globals_dict['grid_nfo']['nlev']
    ]
    stack = np.zeros(dims, order='F')
    dims.pop(1)
    allocate_new = { item : np.zeros(dims) for item in name }
    update.up_entry('data_run', allocate_new)

    del allocate_new

    if kind == 'hat':
        # fatal: RHO missing
        if 'RHO' not in gv.globals_dict['data_run']:
            sys.exit('ERROR: area_avg kind = "hat": missing "RHO"')

        for i, vals in enumerate(var['vars']):
            if 'RHO' not in vals:
                var['vars'][i].append('RHO')

        # semi fatal: RHO_bar missing
        if 'RHO_bar' not in gv.globals_dict['data_run']:
            area_avg('bar', {'vars':['RHO']})

        factor = {}
        factor['RHO_bar'] = gv.globals_dict['data_run']['RHO_bar']
        factor['coarse_area'] = gv.globals_dict['grid_nfo']['coarse_area']
        factor = mult(factor) # make this a no return function, too???

    elif kind == 'bar':
        factor = gv.globals_dict['grid_nfo']['coarse_area']

    else:
        sys.exit('ERROR: area_avg kind = {}: invalid').format(kind)

    update.up_entry('data_run', {'factor' : factor, 'var' : var})

    if gv.mp.get('mp'):
        # go parallel
        pool = Pool(processes = gv.mp['n_procs'])
        result = np.array(pool.map(area_avg_sub, gv.mp['iterator'], 10))
        # reattribute names to variables
        out = {}
        for i, item in enumerate(name):
            out.update({item : result[:, i,]})
        del result
        # write variables to global file
        update.up_entry('data_run', out)
        del out

    else:
        for i in range(gv.globals_dict['grid_nfo']['ncells']):
            stack[i, :, :, :] = area_avg_sub(i)
        # reattribute names to variables
        up_dict = { item : stack[:, i,] for i, item in enumerate(name)}
        # write variables to global file
        update.up_entry('data_run', up_dict)

    return None

# changed according to new array structure - check
# (nothing to do)
def area_avg_sub(i_cell):
    '''
        To manage those pesky vectors. additionally turns vectors
        to align a local spherical coordinate system, then calls avg_hat or avg_bar
        and turns the resulting averaged vector back onto global coordinate system.
        Nu isses ne eierscheissende Wollmilchsau
    '''
    var = gv.globals_dict['data_run']['var']

    values = []
    from_grid = ['cell_area']
    if var.get('vector'):
        from_grid.append('lat')
        from_grid.append('lon')

    areas = do.get_members('grid_nfo', i_cell, from_grid)

    for i, val in enumerate(var['vars']):
        values.append({})
        values[i].update(do.get_members('data_run', i_cell, val))
        values[i].update({'cell_area' : areas['cell_area']})

    if var.get('vector'):
        indx = []
        for k in var['vector']:
            for i, val in enumerate(values):
                if k in val:
                    indx.append(i)

        rot_vec = np.zeros([
            len(var['vector']),
            areas['lat'].shape[0],
            gv.globals_dict['grid_nfo']['ntim'],
            gv.globals_dict['grid_nfo']['nlev']
        ])

        rot_vec[:, :, :, :] = rotate_ca_members_to_local(
            areas['lon'],
            areas['lat'],
            values[indx[0]][var['vector'][0]][:, :, :],
            values[indx[1]][var['vector'][1]][:, :, :]
        )

        values[indx[0]][var['vector'][0]][:, :, :] = rot_vec[0, :, :, :]
        values[indx[1]][var['vector'][1]][:, :, :] = rot_vec[1, :, :, :]

    for i, vals in enumerate(values):
        vals = mult(vals)
        # Sum Rho*var(i) up
        vals = np.sum(vals, 0)
        values[i] = np.divide(vals, gv.globals_dict['data_run']['factor'][i_cell])

# the below may well be superlouus
#    if var.get('vector'):
#        values[indx[0]], values[indx[1]] = rotate_single_to_global(
#            areas['lon'][0],
#            areas['lat'][0],
#            values[indx[0]][:, :],
#            values[indx[1]][:, :]
#        )
    del indx
    return values

# changed according to new array structure - check
# (nothing to do)
def compute_flucts(values, variables, kind):
    '''computes the fluctuations relative to the hat quantities.'''

    kwargs = {
        'values' : values,
        'vars' : variables
        }
    if kind == 'vec':
    #  func = ...
        func = lambda kwargs: vector_flucts(**kwargs)
    elif kind == 'scalar':
    #  func = ...
        func = lambda kwargs: scalar_flucts(**kwargs)
    else:
        sys.exit('ERROR: compute_flucts not ia valid "kind" {}').format(kwargs['kind'])

    values = func(kwargs)

    return values

# changed according to new array structure - check
# (nothing to do)
def scalar_flucts(values, variables):

    for item in variables.iteritems():
        values[item + '_f'] = np.subtract(
            values[item],
            values[item + '_hat'][np.newaxis, :]
        )

    return values

# changed according to new array structure - check
# (nothing to do)
def vector_flucts(values, variables):
    rot_vec = np.zeros([
        len(variables),
        len(gv.globals_dict['grid_nfo']['area_member_idx'][0]),
        gv.globals_dict['grid_nfo']['ntim'],
        gv.globals_dict['grid_nfo']['nlev']
    ])
    rot_vec[:, :, :, :] = rotate_ca_members_to_local(
        values['lon'],
        values['lat'],
        values[variables[0]][:, :, :],
        values[variables[1]][:, :, :]
    )
    rot_bar = np.zeros([
        len(variables),
        gv.globals_dict['grid_nfo']['ntim'],
        gv.globals_dict['grid_nfo']['nlev']
    ])
    rot_bar[:, :, :] = rotate_single_to_local(
        values['lon'][0],
        values['lat'][0],
        values[variables[0] + '_hat'][:, :],
        values[variables[1] + '_hat'][:, :]
    )
    result = np.zeros([
        len(variables),
        len(gv.globals_dict['grid_nfo']['area_member_idx'][0]),
        gv.globals_dict['grid_nfo']['ntim'],
        gv.globals_dict['grid_nfo']['nlev']
        ])
    result = np.subtract(
        rot_vec,
        rot_bar[:, np.newaxis, :, :]
    )
    for i, item in enumerate(variables):
        values[item + '_f'] = result[i, :, :, :]

    return values

# changed according to new array structure - check
def compute_dyads(values, i_cell, variables):
    '''computes the dyadic products of v'''
    if all(key not in values for key in ['RHO']):
        sys.exit('ERROR: compute_dyads, "RHO" missing in values')
    if all(key not in gv.globals_dict['grid_nfo'] for key in ['cell_area']):
        sys.exit('ERROR: compute_dyads, "cell_area" missing in grid_nfo')

    # dimension of dyad
    l_vec = len(variables)

    # in case of first call build output file
    # output slot, outgoing info
    dyad = np.zeros([
        l_vec,
        l_vec,
        gv.globals_dict['grid_nfo']['ntim'],
        gv.globals_dict['grid_nfo']['nlev']
    ])

    # helper, containing the constituents for computation
    constituents = []
    for var in variables:
        constituents.append(values[var])
    constituents = np.array(constituents)
    # helper as handle for avg_bar, values in it are multiplied and averaged over
    product = np.einsum('ilmk,jlmk->ijlmk', constituents, constituents)
    # average over coarse_area
    dyad = np.einsum(
        'ijlmk,l,lmk->ijmk',
        product,
        values['cell_area'],
        values['RHO']
    )
    dyad = np.divide(dyad, gv.globals_dict['grid_nfo']['coarse_area'][i_cell])

    return dyad

# changed according to new array structure
def gradient():

    var = {
        'vars' :['U_hat', 'V_hat'],
        'vector' :['U_hat', 'V_hat']
        }

    r_e = 6.37111*10**6
    l_vec = len(var['vars'])
    out = np.zeros([
        gv.globals_dict['grid_nfo']['ncells'],
        gv.globals_dict['grid_nfo']['ntim'],
        gv.globals_dict['grid_nfo']['nlev'],
        l_vec, l_vec
        ])
    info_bnd = 5000
    for i in range(gv.globals_dict['grid_nfo']['ncells']):
        # chekc for correct key
        # define distance
        # assume circle pi r^2 => d= 2*sqrt(A/pi)
        if i == info_bnd:
            print('progress: {} cells of {}').format(
                info_bnd,
                gv.globals_dict['grid_nfo']['ncells']
            )
            info_bnd = info_bnd + 5000

# rework for flexible length of array. Currently only horizontal gradients.
        # check for correct keys
        neighs = circ_dist_avg(i, var)
        area = gv.globals_dict['grid_nfo']['coarse_area'][i]
        d = 2 * radius(area) * r_e
        out[i, :, :, 0, 0] = central_diff(
            neighs['U_hat'][:, :, 0],
            neighs['U_hat'][:, :, 1],
            d
        )
        out[i, :, :, 0, 1] = central_diff(
            neighs['V_hat'][:, :, 0],
            neighs['V_hat'][:, :, 1],
            d
        )
        out[i, :, :, 1, 0] = central_diff(
            neighs['U_hat'][:, :, 2],
            neighs['U_hat'][:, :, 3],
            d
        )
        out[i, :, :, 1, 1] = central_diff(
            neighs['V_hat'][:, :, 2],
            neighs['V_hat'][:, :, 3],
            d
        )

    return out

# changed according to new array structure
# (nothing to do)
def central_diff(xl, xr, d):

    return np.divide(np.subtract(xl, xr), 2 * d)


# changed according to new array structure
# (nothing to do)
def radius(area):
    '''returns radius of circle on sphere in radians'''
    r_e = 6.37111*10**6
    r = np.sqrt(area / np.pi) / r_e
    return r

# changed according to new array structure - check
def circ_dist_avg(i_cell, var):
    values = {
        name: np.zeros([
            gv.globals_dict['grid_nfo']['ntim'],
            gv.globals_dict['grid_nfo']['nlev'],
            4
        ]) for name in var['vars']
    }
    #how large is check radius?
    for j in range(4):
        coords = gv.globals_dict['gradient_nfo']['coords'][i_cell, j, :]
        member_idx = gv.globals_dict['gradient_nfo']['member_idx'][i_cell][j]
        member_idx = member_idx[np.where(member_idx > -1)[0]]
        member_rad = gv.globals_dict['gradient_nfo']['member_rad'][i_cell][j]
    # compute distance weighted average of area weighted members:
        members = do.get_members_idx(gv.globals_dict['data_run'], member_idx, var['vars'])
        # if you suffer from boredom: This setup gives a lot of flexebility.
        #     may be transferred to other parts of this program.
        if 'vector' in var.iterkeys():
            #turn vectors
            lats = np.array([gv.globals_dict['grid_nfo']['lat'][i] for i in member_idx])
            lons = np.array([gv.globals_dict['grid_nfo']['lon'][i] for i in member_idx])
            vec = var['vector']
            rot_vec = np.zeros([
                len(vec),
                len(member_idx),
                gv.globals_dict['grid_nfo']['ntim'],
                gv.globals_dict['grid_nfo']['nlev']
            ])
            plon, plat = get_polar(coords[0], coords[1])
            rot_vec[:, :, :, :] = rotate_members_to_local(
                lons,
                lats,
                plon,
                plat,
                members[vec[0]][:, :, :],
                members[vec[1]][:, :, :]
            )
            members[vec[0]] = rot_vec[0, :, :, :]
            members[vec[1]] = rot_vec[1, :, :, :]
        helper = dist_avg(
            members,
            member_idx,
            member_rad,
            var['vars']
        )
        if 'vector' in var.iterkeys():
            rot_vec = rotate_single_to_global(
                coords[0],
                coords[1],
                helper[vec[0]][:, :],
                helper[vec[1]][:, :]
            )
            helper[vec[0]] = rot_vec[0]
            helper[vec[1]] = rot_vec[1]

        for name in var['vars']:
            values[name][:, :, j] = helper[name]

    return values

# changed according to new array structure
# (nothing to do)
def dist_avg(members, idxcs, radii, variables):
    len_i = len(idxcs)
    # define weights.
    weight = np.zeros([len_i])
    factor = 0
    for i in range(len_i):
        weight[i] = gv.globals_dict['grid_nfo']['cell_area'][idxcs[i]] * radii[i]
        factor = factor + weight[i]
    # compute distance averages.
    sum_of = {name : 0 for name in variables}
    for k in variables:
        for i in range(len_i):
            sum_of[k] = sum_of[k] + members[k][i]*weight[i]
        sum_of[k] = sum_of[k]/factor

    return sum_of

# changed according to new array structure
# (nothing to do)
def arc_len(p_x, p_y):
    '''computes length of geodesic arc on a sphere (in rad)'''
    r = np.arccos(
        np.sin(p_x[1]) * np.sin(p_y[1])
        + np.cos(p_x[1]) * np.cos(p_y[1]) * np.cos(p_y[0] - p_x[0])
    )
    return r

# changed according to new array structure - check
# (nothing to do)
def mult(dataset):
    dims = np.array([])
    for val in dataset.itervalues():
        if len(val.shape) >= len(dims):
            dims = val.shape

    helper = np.ones(dims)
    for data in dataset.itervalues():
        if data.shape == helper.shape:
            helper = np.multiply(helper, data)
        elif data.ndim == 1:
            if data.shape[0] == helper.shape[0]:
                helper = np.multiply(helper, data[:, np.newaxis, np.newaxis])
            elif data.shape[0] == helper.shape[1]:
                helper = np.multiply(helper, data[np.newaxis, :, np.newaxis])
            elif data.shape[0] == helper.shape[2]:
                helper = np.multiply(helper, data[np.newaxis, np.newaxis, :])
            else:
                print 'I can not find a way to cast {} on {}'.format(
                    data.shape,
                    helper.shape
                )
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
# changed according to new array structure - check
# (nothing to do)
def rotate_ca_members_to_local(lon, lat, x, y):
    # takes in the array of number of hexagons
    #, where x and y contain [num_hex, ntim, nlev] values
    len_x = x.shape[0]
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)
    plon, plat = get_polar(lon[0], lat[0])
    for i in range(len_x):
        x_tnd[i, :, :], y_tnd[i, :, :] = rotate_vec_to_local(
            plon, plat,
            lon[i], lat[i],
            x[i, :, :], y[i, :, :]
        )
    return x_tnd, y_tnd

# changed according to new array structure - check
# (nothing to do)
def rotate_members_to_local(lon, lat, plon, plat, x, y):
    # takes in the array of number of hexagons
    #, where x and y contain [num_hex, ntim, nlev] values
    len_x = x.shape[0]
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)
    for i in range(len_x):
        x_tnd[i, :, :], y_tnd[i, :, :] = rotate_vec_to_local(
            plon, plat,
            lon[i], lat[i],
            x[i, :, :], y[i, :, :]
        )
    return x_tnd, y_tnd

# changed according to new array structure - check
# (nothing to do)
def rotate_single_to_local(lon, lat, x, y):
    # shouldn't be neccessary. The entropy production value should remain the same
    # independent rotation back should therefore not be neccessary and additional
    # computing time.

    plon, plat = get_polar(lon, lat)
    x_tnd, y_tnd = rotate_vec_to_local(
        plon, plat,
        lon, lat,
        x, y)
    return x_tnd, y_tnd

# changed according to new array structure - check
# (nothing to do)
def rotate_ca_members_to_global(lon, lat, x, y):
    # takes in the array of number of hexagons
    #, where x and y contain [num_hex, ntim, nlev] values
    len_x = x.shape[0]
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)
    plon, plat = get_polar(lon[0], lat[0])
    for i in range(len_x):
        x_tnd[i, :, :], y_tnd[i, :, :] = rotate_vec_to_global(
            plon, plat,
            lon[i], lat[i],
            x[i, :, :], y[i, :, :]
        )
    return x_tnd, y_tnd

# changed according to new array structure - check
# (nothing to do)
def rotate_members_to_global(lon, lat, plon, plat, x, y):
    # takes in the array of number of hexagons
    #, where x and y contain [num_hex, ntim, nlev] values
    len_x = x.shape[0]
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)
    for i in range(len_x):
        x_tnd[i, :, :], y_tnd[i, :, :] = rotate_vec_to_global(
            plon, plat,
            lon[i], lat[i],
            x[i, :, :], y[i, :, :]
        )
    return x_tnd, y_tnd


# changed according to new array structure - check
# (nothing to do)
def rotate_single_to_global(lon, lat, x, y):
    # shouldn't be neccessary. The entropy production value should remain the same
    # independent rotation back should therefore not be neccessary and additional
    # computing time.

    plon, plat = get_polar(lon, lat)
    x_tnd, y_tnd = rotate_vec_to_global(
        plon, plat,
        lon, lat,
        x, y
    )
    return x_tnd, y_tnd

# changed according to new array structure - check
# (nothing to do)
def get_polar(lon, lat):
    plon = 0.0
    if 0 < lon <= np.pi:
        plon = lon - np.pi
    elif -np.pi < lon < 0:
        plon = lon + np.pi
    #else:
        # no turning needed here

    plat = np.pi/2
    if 0 < lat <= np.pi/2:
        plat = np.pi/2 - lat
    elif -np.pi/2 <= lat < 0:
        plat = -np.pi/2-lat
    #else:
        # no turning needed here

    return plon, plat


# changed according to new array structure - check
# (nothing to do)
def rotate_vec_to_local(plon, plat, lon, lat, x, y):
    ''' rotates vectors using rotat_latlon_vec'''
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)

    sin_d, cos_d = rotate_latlon_vec(lon, lat, plon, plat)

    x_tnd = x * cos_d - y * sin_d
    y_tnd = x * sin_d + y * cos_d

    return x_tnd, y_tnd

# changed according to new array structure - check
# (nothing to do)
def rotate_vec_to_global(plon, plat, lon, lat, x, y):
    ''' rotates vectors using rotate_latlon_vec'''
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)

    sin_d, cos_d = rotate_latlon_vec(lon, lat, plon, plat)

    x_tnd = x * cos_d + y * sin_d
    y_tnd = -x * sin_d + y * cos_d

    return x_tnd, y_tnd

# changed according to new array structure - check
# (nothing to do)
def rotate_latlon_vec(lon, lat, plon, plat):
    '''gives entries of rotation matrix for vector rotation
        see Documentation of COSMOS pp. 27'''

    z_lamdiff = lon - plon
    z_a = np.cos(plat) * np.sin(z_lamdiff)
    z_b = np.cos(lat) * np.sin(plat) - np.sin(lat) * np.cos(plat) * np.cos(z_lamdiff)
    z_sq = np.sqrt(z_a * z_a + z_b * z_b)
    sin_d = z_a / z_sq
    cos_d = z_b / z_sq

    return sin_d, cos_d
