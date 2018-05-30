'''The main of the Coarse-Graining Post Processing Program
    The actual computation of thetrubulent diffusion is done here.'''
import os
import sys
import glob
import time
import numpy as np
import custom_io as cio
import data_op as do
import data_op_p as dop
import math_op as mo
import math_op_p as mop
import phys_op as po
import xarray as xr
from multiprocessing import Pool

# the main file
def count_time(t1=0):
    if t1 == 0:
        t1 = time.time()
    elif t1 != 0:
        t2 = time.time()
        t1 = t2-t1
        print('Time gone {}').format(t1)
    else:
        print('Something went horribly wrong!')
    return t1

def user():
    '''User interfacce routine.
        returns experiment name and area to be averaged over.'''
    kwargs = {}
    print 'enter the experiment you wish to postprocess:'
    kwargs['experiment'] = raw_input()
    print 'how many rings of hexagons shall be averaged over?'
    while True:
        helper = int(input())
        if isinstance(helper, int):
            kwargs['num_rings'] = helper
            break
    print 'how many files shall be processed?'
    while True:
        helper = int(input())
        if isinstance(helper, int):
            kwargs['num_files'] = helper
            break
    return kwargs

def read_grid(kwargs):
    '''Opens the grid file.
        Automatically checks the grid files for a refined one,
        if none present creates one.'''
    switch = True
    func = None
    quarks = {}
    for grid in kwargs['grid']:
        print grid
        if 'refined_{}'.format(kwargs['num_rings']) in grid:
            switch = False
            kwargs['grid'] = [grid]
        print switch
    if switch:
        for grid in kwargs['grid']:
            if not 'refined' in grid:
                kwargs['grid'] = [grid]
        path = kwargs['grid'][0]
        quarks = {
            'num_rings': kwargs['num_rings'],
            'path' : path
            }
        func = lambda ds, quarks: cio.read_grid(ds, **quarks)
    return cio.read_netcdfs(kwargs['grid'], 'time', quarks, func)


def read_data(kwargs, i):
    '''Reads the data files.'''
    quarks = {}
    quarks['variables'] = kwargs['variables']
    func = lambda ds, quarks: cio.extract_variables(ds, **quarks)
    return cio.read_netcdfs([kwargs['files'][i]], 'time', quarks, func)

def do_the_average(data, grid_nfo, kwargs):
    '''computes the u and v hat values'''
    # fairly simple
    var = {
        'vars'      :['U', 'V'],
        'vector'       :['U', 'V']
        }
    # compute \bar{U} and \bar{V}


    print('  ------------')
    print('  computing bar averages')
    var = {
        'vars'      :['U', 'V'],
        'vector'       :['U', 'V']
        }
    data = mo.area_avg('bar', grid_nfo, data, var, True)
    var = {
        'vars'      :['T', 'RHO'],
        }
    data = mo.area_avg('bar', grid_nfo, data, var, False)
    # compute \hat{U} and \hat{V}
    print('  ------------')
    print('  computing hat averages')
    var = {
        'vars'      :['U', 'V'],
        'vector'       :['U', 'V']
        }
    data = mo.area_avg('hat', grid_nfo, data, var, True)
    var = {
        'vars'      :['T'],
        }
    data = mo.area_avg('hat', grid_nfo, data, var)

    print('mean VelocityU  : bar:{}, hat:{}').format(
        np.mean(data['U_bar']),
        np.mean(data['U_hat'])
    )
    print('mean VelocityV  : bar:{}, hat:{}').format(
        np.mean(data['V_bar']),
        np.mean(data['V_hat'])
    )
    print('mean Density    : bar:{}, hat: - ').format(
        np.mean(data['RHO_bar'])
    )
    print('mean Temperature: bar:{}, hat:{}').format(
        np.mean(data['T_bar']),
        np.mean(data['T_hat'])
    )

    return data

def do_the_gradients_mp(data, grid_nfo, gradient_nfo, kwargs):
    # prepare mutiprocessing in here.
    print('    ------------- ')
    print('    preparing multiprocessing: neighbours')
    '''computes the gradients of u and v'''
    var = {
        'vars'      :['U_hat', 'V_hat'],
        'vector'      :['U_hat', 'V_hat']
        }
    # do circ_dist_avg in seperate step.
    # data[var], grid_nfo, gradient_nfo['coords', 'member_idx']
    t1 = count_time()
    print(t1)
    num_proc = 30
    chunks_vec = []
    for i in range(grid_nfo['ncells']):
        # variable specific information
        chunk = []
        for j,x in enumerate(var['vars']):
            chunk.append([])
            for k in range(4):
                # construct member array (len(vars))
                helper = data[x][:, :, np.where(gradient_nfo['member_idx'][i][k] >-1 )[0]]
                chunk[j].append(np.moveaxis(helper, -1, 0))
        # preparation for general information
        latlon = []
        for k in range(4):
            # coordinates of members (for turning)
            latlon.append([
                grid_nfo['lat'][np.where(gradient_nfo['member_idx'][i][k] > -1)[0]],
                grid_nfo['lon'][np.where(gradient_nfo['member_idx'][i][k] > -1)[0]],
                grid_nfo['cell_area'][np.where(gradient_nfo['member_idx'][i][k] > -1)[0]]
            ])
        # general information, always positioned at [-1]
        chunk.append([
            # distance of member from center (for distance weighted average)
            gradient_nfo['member_rad'][i],
            # center coordinate (for turning of vectors)
            gradient_nfo['coords'][i],
            # coordinates of members for turning
            latlon
        ])
        chunks_vec.append([chunk, [
           [data[x][:, :, i] for j,x in enumerate(var['vars'])],
           grid_nfo['coarse_area'][i]]
        ])

    print(chunks_vec[0])
    # I don't know why, but this doesn't function...
    t2 = count_time()
    print(t2-t1)

    t1 = count_time()
    print('    ------------- ')
    print('    starting multiprocessing on {} processors').format(num_proc)
    chunks = [chunks_vec[i::num_proc] for i in range(num_proc)]
    #pool = Pool(processes=num_proc)
    #chunks_out = pool.map(neighs_and_grad, chunks)
    chunks_out = neighs_and_grad(chunks[0])
    #gradients = np.moveaxis(dop.reorder(chunks_out), 0, -1)
    #chunks = [chunks_vec[i::num_proc] for i in range(num_proc)]
    #pool = Pool(processes=num_proc)
    #chunks_out = pool.map_async(mop.circ_dist_avg_vec, chunks).get()
    #neighbours = dop.reorder(chunks_out)
    #t2 = count_time()
    print(t2-t1)
    #print('    ------------- ')
    #print('    preparing multiprocessing: gradients')

    #t1 = count_time()
    #chunks_vec = []
    #for i in range(grid_nfo['ncells']):
    #    chunk = []
    #    chunk.append(neighbours[i])
    #    chunk.append([data[x][:, :, i] for j,x in enumerate(var['vars']) ])
    #    chunk.append(grid_nfo['coarse_area'][i])
    #    chunks_vec.append(chunk)

    #chunks = [chunks_vec[i::num_proc] for i in range(num_proc)]

    #t2 = count_time()
    #print(t2-t1)
    #print('    ------------- ')
    #print('    starting multiprocessing on {} processors').format(num_proc)

    #t1 = count_time()
    #chunks_vec = []
    #pool = Pool(processes=num_proc)
    #chunks_out = pool.map_async(mop.gradient_mp, chunks).get()
    #gradients = np.moveaxis(dop.reorder(chunks_out), 0, -1)
    #t2 = count_time()
    #print(t2-t1)

    data['gradient'] = gradients

    print('    ------------- ')
    print('mean velocity gradient u_x: {}').format(np.mean(data['gradient'][0,0]))
    print('mean velocity gradient u_y: {}').format(np.mean(data['gradient'][0,1]))
    print('mean velocity gradient v_x: {}').format(np.mean(data['gradient'][1,0]))
    print('mean velocity gradient v_y: {}').format(np.mean(data['gradient'][1,1]))
    return data

def neighs_and_grad(chunk):

    neighs_chunk = mop.circ_dist_avg_vec(chunk[0])
    neighs_chunk.append(chunk[1])
    chunk = mop.gradient_mp(neighs_chunk)

    return chunk

def do_the_gradients(data, grid_nfo, gradient_nfo, kwargs):
    '''computes the gradients of u and v'''
    var = {
        'vars'      :['U_hat', 'V_hat'],
        'vector'      :['U_hat', 'V_hat']
        }

    data = mo.gradient(data, grid_nfo, gradient_nfo, var)

    print('mean velocity gradient u_x: {}').format(np.mean(data['gradient'][0,0]))
    print('mean velocity gradient u_y: {}').format(np.mean(data['gradient'][0,1]))
    print('mean velocity gradient v_x: {}').format(np.mean(data['gradient'][1,0]))
    print('mean velocity gradient v_y: {}').format(np.mean(data['gradient'][1,1]))
    return data

def do_the_dyads_mp(data, grid_nfo):
    '''computes the dyadic product of uv and rho plus averaging'''
    # check if everything is there
    needs = ['U', 'U_bar', 'V', 'V_bar', 'RHO', 'RHO_bar']
    if not all(need in data for need in needs):
        this = [need for need in needs if need not in data]
        sys.exit('ERROR: do_the_dyads I miss quantities to do the computing {}'.format(this))
      # define kwargs for computation
    kwargs = {
        'vars'    : ['U', 'V', 'RHO'],
        'UV'      : {'vars'   : ['U', 'V'],
                     'kind'   : 'vec'
                    },
        'dyad'    : {'vars'   : ['U_f', 'V_f']
                    }
        }

      #start cellwise iteration
    doprint = 5000
    if not('dyad' in data):
        l_vec = len(kwargs['UV']['vars'])
        # in case of first call build output file
        # output slot, outgoing info
        print('creating array values["dyad"]')
        data['dyad']        = np.empty([
            l_vec,
            l_vec,
            grid_nfo['ntim'],
            grid_nfo['nlev'],
            grid_nfo['ncells']
            ])
    chunk_vec = []
    for i in range(grid_nfo['ncells']):
        chunk = []
        if i == doprint:
            print('cell {} of {}').format(i, grid_nfo['ncells'])
            doprint = doprint + 5000
        # get area members
        values = do.get_members(grid_nfo, data, i, kwargs['vars'])
        chunk.append([var for var in values.iteritems()])
        # add lat lon coordinates to values
        values = do.get_members(grid_nfo, grid_nfo, i, ['lat', 'lon'])
        chunk.append([var for var in values.iteritems()])
        # get individual areas
        values = do.get_members(grid_nfo, grid_nfo, i, ['cell_area'])
        chunk.append([var for var in values.iteritems()])
        # get coarse values
        values = {}
        for var in kwargs['vars'][:2]:
            values[var+'_hat'] = data[var+'_hat'][:, :, i]
        chunk.append([var for var in values.iteritems()])
        # compute fluctuations
        chunk_vec.append(chunk)
    chunks = [chunk_vec[i::n_procs] for i in range(n_procs)]


    values = mo.compute_flucts(values, grid_nfo, grid_nfo['area_num_hex'][i], **kwargs['UV'])
    data['dyad'][:,:,:,:,i] = mo.compute_dyads(values, grid_nfo, i, **kwargs['dyad'])

    return data

def dyads_parallel(chunk):

    return chunk

def do_the_dyads(data, grid_nfo):
    '''computes the dyadic product of uv and rho plus averaging'''
    # check if everything is there
    needs = ['U', 'U_bar', 'V', 'V_bar', 'RHO', 'RHO_bar']
    if not all(need in data for need in needs):
        this = [need for need in needs if need not in data]
        sys.exit('ERROR: do_the_dyads I miss quantities to do the computing {}'.format(this))
      # define kwargs for computation
    kwargs = {
        'vars'    : ['U', 'V', 'RHO'],
        'UV'      : {'vars'   : ['U', 'V'],
                     'kind'   : 'vec'
                    },
        'dyad'    : {'vars'   : ['U_f', 'V_f']
                    }
        }

      #start cellwise iteration
    doprint = 5000
    if not('dyad' in data):
        l_vec = len(kwargs['UV']['vars'])
        # in case of first call build output file
        # output slot, outgoing info
        print('creating array values["dyad"]')
        data['dyad']        = np.empty([
            l_vec,
            l_vec,
            grid_nfo['ntim'],
            grid_nfo['nlev'],
            grid_nfo['ncells']
            ])

    for i in range(grid_nfo['ncells']):
        if i == doprint:
            print('cell {} of {}').format(i, grid_nfo['ncells'])
            doprint = doprint + 5000
        # get area members
        values = do.get_members(grid_nfo, data, i, kwargs['vars'])
        # add lat lon coordinates to values
        values.update(do.get_members(grid_nfo, grid_nfo, i, ['lat', 'lon']))
        # get individual areas
        values.update(do.get_members(grid_nfo, grid_nfo, i, ['cell_area']))
        # get coarse values
        for var in kwargs['vars'][:2]:
            values[var+'_hat'] = data[var+'_hat'][:, :, i]
        # compute fluctuations
        values = mo.compute_flucts(values, grid_nfo, grid_nfo['area_num_hex'][i], **kwargs['UV'])
        data['dyad'][:,:,:,:,i] = mo.compute_dyads(values, grid_nfo, i, **kwargs['dyad'])

    return data

def perform(data, grid_nfo, gradient_nfo, kwargs):
    '''performs the compuations neccessary'''
    # compute u and v hat and bar
    print('--------------')
    print('averaging variable fields')
    data = do_the_average(data, grid_nfo, kwargs)
    # '_bar' and '_hat' contain averages
    # compute gradient
    print('--------------')
    print('computing the gradients')
   # t1 = count_time()
   # print(time.time())
   # data = do_the_gradients_mp(data, grid_nfo, gradient_nfo, kwargs)
   # t1 = count_time(t1)
    t2 = count_time()
    print(time.time())
    data = do_the_gradients(data, grid_nfo, gradient_nfo, kwargs)
    t2 = count_time(t2)
    print(time.time())
   # print('Speedup: {}').format(t2-t1)
    # 'gradient' [necells, 0:1, 0:1, ntim, nlev]
    # 0:1 : d/dx d/dy; 0:1 : u, v
    # compute and average the dyads plus comute their primes
    print('--------------')
    print('computing the dyads')
    data = do_the_dyads(data, grid_nfo)
    # 'dyad' [0:1,0:1, ntim, nlev, ncells]
    for key in data.iterkeys():
        print key
    print('--------------')
    print('computing the turbulent friction')
    data['turb_fric'] = np.empty([grid_nfo['ntim'],
                                 grid_nfo['nlev'],
                                 grid_nfo['ncells']])
    data['turb_fric'].fill(0.0)
    doprint = 0

    data['turb_fric'] = np.einsum('ijklm,ijklm->klm', data['dyad'], data['gradient'])
    #equivalent to:
    #for i in range(2):
    #    for j in range(2):
    #        data['turb_fric'][:, :, :] = data['turb_fric'] + np.multiply(
    #            data['dyad'][i, j, :, :, :],
    #            data['gradient'][i, j, :, :, :]
    #            )
    data['turb_diss'] = -1 * np.divide(data['turb_fric'], data['RHO_bar'])
    data['turb_fric'] = -1 * np.divide(data['turb_fric'], data['T_bar'])
    return data

def prepare_mp(data, grid_nfo, gradient_nfo, kwargs):
    mp_array = []
    mp_sub_dict = {
        'data' : {},  # muss gesplittet werden
        'grid_nfo' : {},  # muss gesplittet werden
        'gradient_nfo' : {}, # muss gesplittet werden
        'kwargs' : {} # kwargs ist immer gleich für alle procs
    }

    data_sets = {
        'data' : data,
        'grid_nfo' : grid_nfo,
        'gradient_nfo' : gradient_nfo
    }
    num_procs = kwargs['mp']['num_procs']

    for i in range(num_procs):
        for name, data_set in data_sets.iteritems():
            for key, item in data_set.iteritems():
                index_ncells = data_set[key].shape.index(grid_nfo['ncells'])
                num_dims = len(data_set[key].shape)
                if index_ncells == 0:
                    mp_sub_dict['name']['key'] = np.array([data_set[key][i::num_proc,]])
                elif index_ncells == 2:
                    mp_sub_dict['name']['key'] = np.array([data_set[key][:, i::num_proc,]])
                elif index_ncells == 3:
                    mp_sub_dict['name']['key'] = np.array([data_set[key][:, :, i::num_proc,]])
                elif index_ncells == 4:
                    mp_sub_dict['name']['key'] = np.array([data_set[key][:, :, :, i::num_proc,]])
                elif index_ncells == 5:
                    mp_sub_dict['name']['key'] = np.array([data_set[key][:, :, :, :, i::num_proc,]])
        mp_sub_dict['kwargs'] = kwargs
        mp_array.append(mp_sub_dict)
    return mp_array

if __name__ == '__main__':
   # kwargs = user()
    kwargs = {
        'experiment' : 'HS_FT_6000_days',
        'num_rings' : 3,
        'num_files' : 179
    }
    kwargs['filep'] = u'/home1/kd031/projects/icon/experiments/'+kwargs['experiment']+'/'
    kwargs['files'] = [
        n for n in
        glob.glob(kwargs['filep']+kwargs['experiment']+'_*.nc') if
        os.path.isfile(n)]
    kwargs['grid'] = [
        n for n in
        glob.glob(kwargs['filep']+'*grid*.nc') if
        os.path.isfile(n)]
    print kwargs['grid']
    kwargs['variables'] = ['U', 'V', 'RHO', 'THETA_V', 'EXNER']
    if not kwargs['files'] or not kwargs['grid']:
        sys.exit('Error:missing gridfiles or datafiles')
    grid = read_grid(kwargs)
    grid_nfo = {
        'area_num_hex'        : grid['area_num_hex'].values,
        'area_neighbor_idx'   : grid['area_neighbor_idx'].values,
        'coarse_area'         : grid['coarse_area'].values,
        'cell_area'           : grid['cell_area'].values,
        'i_cell'              : 0
        }
    gradient_nfo = {
        'coords' : grid['coords'].values,
        'member_idx' : grid['member_idx'].values,
        'member_rad' : grid['member_rad'].values
    }
    kwargs['mp'] = {
        'switch' : True,
        'num_procs' : 20
    }

    if kwargs['num_files'] > len(kwargs['files']):
        fin = len(kwargs['files'])
    else:
        fin = kwargs['num_files']

    for i in range(fin):
        print ('file {} of {}').format(i, fin)
        data = read_data(kwargs, i)
        grid_nfo.update({
            'ntim'                : data.dims['time'],
            'nlev'                : data.dims['lev'],
            'ncells'              : data.dims['ncells'],
            'lat'                 : data['clat'].values,
            'lon'                 : data['clon'].values
            })
        data_run = {}
        for var in kwargs['variables']:
            data_run[var] = data[var].values
        #compute Temperature T for computation
        data_run['T'] = po.potT_to_T_exner(data_run['THETA_V'], data_run['EXNER'])

        data_run.pop('EXNER')
        data_run.pop('THETA_V')

        # critcally wrong still:
        data_run = perform(data_run, grid_nfo, gradient_nfo, kwargs)

        print('min {} and max {}').format(np.min(data_run['turb_fric']), np.max(data_run['turb_fric']))
        print('globally averaged entropy production rate: {}').format(np.mean(data_run['turb_fric']))
        t_fric = xr.DataArray(
            data_run['turb_fric'],
            dims = ['time', 'lev', 'ncells']
        )

        t_diss = xr.DataArray(
            data_run['turb_diss'],
            dims = ['time', 'lev', 'ncells']
        )

        T = xr.DataArray(
            data_run['T'],
            dims = ['time', 'lev', 'ncells']
        )

        data = data.assign(t_fric = t_fric)
        data = data.assign(t_diss = t_diss)
        data = data.assign(T = T)

        cio.write_netcdf(kwargs['files'][i][:-3]+'_refined_{}.nc'.format(kwargs['num_rings']),
                         data)


# make rho and v Grid, weigh them

# make T Grid, weigh them

# make \hat v Grid

# make v'' Grid

# delete v Grid

# make \bar( rho v'' v'') Grid

# delete rho Grid

# delete v'' Grid

# make partial \hat v Grid

# delete \hat v Grid

