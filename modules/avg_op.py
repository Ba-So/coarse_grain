#!/usr/bin/env python
# coding=utf-8

class operations():

    @requires_variables()
    def avg_bar(var):

        return area_avg_sub(vars_in)

    def avg_hat(var):

        return area_avg_sub(vars_in)


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
@multiprocess()
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


