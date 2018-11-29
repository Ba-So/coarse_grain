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
        result = np.array(pool.map(area_avg_sub, gv.mp['iterator'], gv.mp['chunksize']))
        # reattribute names to variables
        out = {}
        for i, item in enumerate(name):
            out.update({item : result[:, i,]})
        del result
        # write variables to global file
        update.up_entry('data_run', out)
        pool.close()
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
        for vector in var['vector']:
            indx = []
            for k in vector:
                for i, val in enumerate(values):
                    if k in val:
                        indx.append(i)

            rot_vec = np.zeros([
                len(vector),
                areas['lat'].shape[0],
                gv.globals_dict['grid_nfo']['ntim'],
                gv.globals_dict['grid_nfo']['nlev']
            ])
            rot_vec[:, :, :, :] = rotate_ca_members_to_local(
                areas['lon'],
                areas['lat'],
                values[indx[0]][vector[0]],
                values[indx[1]][vector[1]]
            )

            values[indx[0]][vector[0]][:, :, :] = rot_vec[0, :, :, :]
            values[indx[1]][vector[1]][:, :, :] = rot_vec[1, :, :, :]

    for i, vals in enumerate(values):
        vals = mult(vals)
        # Sum Rho*var(i) up
        vals = np.sum(vals, 0)
        values[i] = np.divide(vals, gv.globals_dict['data_run']['factor'][i_cell])
    del indx
    return values

# changed according to new array structure - check
# (nothing to do)
def compute_flucts(values):
    '''computes the fluctuations relative to the hat quantities.'''

    variables = gv.globals_dict['data_run']['fluctsof']

    kind = gv.globals_dict['data_run']['kind']
    if kind == 'vec':
    #  func = ...

        rot_vec = rotate_ca_members_to_local(
            values['lon'],
            values['lat'],
            values[variables[0]][:, :, :],
            values[variables[1]][:, :, :]
        )

        rot_hat = rotate_single_to_local(
            values['lon'][0],
            values['lat'][0],
            values[variables[0] + '_hat'][:, :],
            values[variables[1] + '_hat'][:, :]
        )

        for i, item in enumerate(variables):
            values[item] = rot_vec[i]
            values[item + '_hat'] = rot_hat[i]

    elif kind == 'scalar':
        pass

    else:
        sys.exit('ERROR: compute_flucts not ia valid "kind" {}').format(kind)

    for item in variables:
        values[item + '_f'] = np.subtract(
            values[item],
            values[item + '_hat'][np.newaxis, :]
        )

    return values

def trace(var):
    ''' computes the trace of a matrix nxn = sum_i^n m_ii
        var - dictionary
        For future additions this function could -
        technically - be turned into a function using a lambda argument.'''

    update = up.Updater()
    # in order to have var in the global memory for parallel computing
    traces = []
    for xvar in var.iteritems():
        traces.append({})
        update.up_entry('data_run', {'var' : xvar})
        if gv.mp.get('mp'):
            pool = Pool(processes = gv.mp['n_procs'])
            out = pool.map(trace_sub, gv.mp['iterator'], gv.mp['chunksize'])
            pool.close()
        else:
            out = []
            for i in range(gv.globals_dict['grid_nfo']['ncells']):
                out.append(trace_sub(i))
        traces[xvar+'_trace'] = out

    for xtrace, i in enumerate(traces):
        update.up_entry('data_run', xtrace)

def trace_sub(i_cell):
    ''' this function does the actual work in computing the trace. Uses numpy.trace
        by convetion the trace is computed over the first and second axis of the input.'''

    xvar = gv.globals_dict['data_run']['var']
    trace = np.trace(gv.globals_dict['data_run'][xvar][i_cell, :], axis1=0, axis2=1)

    return trace

# changed according to new array structure - check
def compute_dyads(var):
    update = up.Updater()
    update.up_entry('data_run', var)

    if gv.mp.get('mp'):
        pool = Pool(processes = gv.mp['n_procs'])
        out = pool.map(compute_dyads_sub, gv.mp['iterator'], gv.mp['chunksize'])
        pool.close()
    else:
        out = []
        for i in range(gv.globals_dict['grid_nfo']['ncells']):
            out.append(compute_dyads_sub(i))

    update.append_entry('data_run', {'dyad' : np.array(out)})
    return None


def compute_dyads_sub(i_cell):

    '''computes the dyadic products of v'''
    if all(key not in gv.globals_dict['data_run'] for key in ['RHO']):
        sys.exit('ERROR: compute_dyads, "RHO" missing in values')
    if all(key not in gv.globals_dict['grid_nfo'] for key in ['cell_area']):
        sys.exit('ERROR: compute_dyads, "cell_area" missing in grid_nfo')

    var = gv.globals_dict['data_run']['vars']
    dyad_name = gv.globals_dict['data_run']['dyad_name']
    # dimension of dyad

    # get area myembers
    values = do.get_members('data_run', i_cell, var)
    # add lat lon coordinates to values, and areas
    values.update(do.get_members('grid_nfo', i_cell, ['lat', 'lon', 'cell_area']))
    # get coarse values
    for name in var[:2]:
        values[name+'_hat'] = gv.globals_dict['data_run'][name+'_hat'][i_cell, :, :]
    # compute fluctuations
    values = compute_flucts(values)

    # helper, containing the constituents for computation
    constituents = []
    for name in dyad_name:
        constituents.append(values[name])
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
def gradient(var):
    update = up.Updater()

    update.up_entry('data_run', {'var' : var})

    if gv.mp.get('mp'):
        pool = Pool(processes = gv.mp['n_procs'])
        out = pool.map(gradient_sub, gv.mp['iterator'], 10)
        pool.close()

    else:
        out = []
        for i in range(gv.globals_dict['grid_nfo']['ncells']):
            out.append(gradient_sub(i))

    update.up_entry('data_run', {'gradient' : np.array(out)})

    return None

def gradient_sub(i_cell):
    # chekc for correct key
    # define distance
    # assume circle pi r^2 => d= 2*sqrt(A/pi)
    var = gv.globals_dict['data_run']['var']
    l_vec = len(var['vars'])
    out = np.zeros([
        l_vec, l_vec,
        gv.globals_dict['grid_nfo']['ntim'],
        gv.globals_dict['grid_nfo']['nlev']
    ])

    r_e = 6.37111*10**6
    # rework for flexible length of array. Currently only horizontal gradients.
    # check for correct keys
    neighs = circ_dist_avg(i_cell, var)
    area = gv.globals_dict['grid_nfo']['coarse_area'][i_cell]
    d = 2 * radius(area) * r_e
    # dxU
    out[0, 0, :, :] = central_diff(
        neighs['U_hat'][:, :, 0],
        neighs['U_hat'][:, :, 1],
        d
    )
    #dxV
    out[0, 1, :, :] = central_diff(
        neighs['V_hat'][:, :, 0],
        neighs['V_hat'][:, :, 1],
        d
    )
    #dyU
    out[1, 0, :, :] = central_diff(
        neighs['U_hat'][:, :, 2],
        neighs['U_hat'][:, :, 3],
        d
    )
    #dyV
    out[1, 1, :, :] = central_diff(
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
        members = do.get_members_idx('data_run', member_idx, var['vars'])
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


def rotate_vec_to_local(plon, plat, lon, lat, x, y):
    ''' rotates vectors using rotat_latlon_vec'''
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)

    sin_d, cos_d = rotate_latlon_vec(lon, lat, plon, plat)

    x_tnd = x * cos_d - y * sin_d
    y_tnd = x * sin_d + y * cos_d

    return x_tnd, y_tnd

def rotate_vec_to_global(plon, plat, lon, lat, x, y):
    ''' rotates vectors using rotate_latlon_vec'''
    x_tnd = np.zeros(x.shape)
    y_tnd = np.zeros(y.shape)

    sin_d, cos_d = rotate_latlon_vec(lon, lat, plon, plat)

    x_tnd = x * cos_d + y * sin_d
    y_tnd = -x * sin_d + y * cos_d

    return x_tnd, y_tnd

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

def num_hex_from_rings(num_rings):
    '''returns number of hexagons in patch of hexagons with num_rings rings around it's center'''
    num_hex = 1 + 6 * num_rings * (num_rings + 1) / 2
    return num_hex

