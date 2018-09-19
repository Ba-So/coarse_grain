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
import global_vars as gv
import update as up
import xarray as xr

from multiprocessing import Pool

# the main file
def count_time(t1=0):
    if t1 == 0:
        t1 = time.time()
    elif t1 != 0:
        t2 = time.time()
        t1 = t2-t1
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
    kwargs['grid'] = None
    grids = [
        n for n in
        glob.glob(kwargs['filep']+'*grid*.nc') if
        os.path.isfile(n)]
    for grid in grids:
        if 'refined_{}'.format(kwargs['num_rings']) in grid:
            kwargs['grid'] = grid
            break
    if not kwargs['grid']:
        sys.exit('Error:missing refined gridfile, run refine_grid first.')
        # maybe call refining routine in this case

    return cio.read_netcdfs(kwargs['grid'])


def read_data(kwargs, i):
    '''Reads the data files.'''
    ds = cio.read_netcdfs(kwargs['files'][i])
    ds = cio.rename_dims_vars(ds)
    ds = cio.extract_variables(ds, kwargs['variables'])
    return ds
# changed according to new array structure
def do_the_average():
    '''computes the u and v hat values'''

    # compute \bar{U} and \bar{V} and RHO and T
    print('  ------------')
    print('  computing bar averages')
    var = {
        'vars'      :[['U'], ['V'], ['G1_X'], ['G1_Y'], ['T'], ['ZSTAR'], ['RHO']],
        'vector'    :[['U', 'V'],['G1_X', 'G1_Y']]#
        }
    mo.area_avg('bar', var)
    # this must become more involved for mp

    # compute \hat{U} and \hat{V}
    print('  ------------')
    print('  computing hat averages')
    var = {
        'vars'      :[['U'], ['V'], ['G1_X'], ['G1_Y'], ['T']],
        'vector'    :[['U', 'V'], ['G1_X', 'G1_Y']]
        }

    mo.area_avg('hat', var)

    print('mean Velocity U  : bar:{}, hat:{}').format(
        np.mean(gv.globals_dict['data_run']['U_bar']),
        np.mean(gv.globals_dict['data_run']['U_hat'])
    )
    print('mean Velocity V  : bar:{}, hat:{}').format(
        np.mean(gv.globals_dict['data_run']['V_bar']),
        np.mean(gv.globals_dict['data_run']['V_hat'])
    )
    print('mean Density    : bar:{}, hat: - ').format(
        np.mean(gv.globals_dict['data_run']['RHO_bar'])
    )
    print('mean Temperature: bar:{}, hat:{}').format(
        np.mean(gv.globals_dict['data_run']['T_bar']),
        np.mean(gv.globals_dict['data_run']['T_hat'])
    )

    return None

def neighs_and_grad(chunk):

    neighs_chunk = mop.circ_dist_avg_vec(chunk[0])
    neighs_chunk.append(chunk[1])
    chunk = mop.gradient_mp(neighs_chunk)

    return chunk

# changed according to new array structure
def do_the_gradients():
    '''computes the gradients of u and v'''

    var = {
        'vars' :['U_hat', 'V_hat'],
        'vector' :['U_hat', 'V_hat']
        }


    mo.gradient(var)

    print('mean velocity gradient u_x: {}').format(np.mean(gv.globals_dict['data_run']['gradient'][0, 0]))
    print('mean velocity gradient u_y: {}').format(np.mean(gv.globals_dict['data_run']['gradient'][0, 1]))
    print('mean velocity gradient v_x: {}').format(np.mean(gv.globals_dict['data_run']['gradient'][1, 0]))
    print('mean velocity gradient v_y: {}').format(np.mean(gv.globals_dict['data_run']['gradient'][1, 1]))
    return None

# changed according to new array structure - check
def do_the_dyads():
    '''computes the dyadic product of uv and rho plus averaging'''
    update = up.Updater()
    update.up_entry('data_run', {'dyad' : []})
    # check if everything is there
    needs = ['U', 'U_bar', 'V', 'V_bar', 'RHO', 'RHO_bar']
    if not all(need in gv.globals_dict['data_run'] for need in needs):
        this = [need for need in needs if need not in gv.globals_dict['data_run']]
        sys.exit('ERROR: do_the_dyads I miss quantities to do the computing {}'.format(this))
    ## Do the regular dyads now lands in position [0] of ['data_run']['dyads']
      # define kwargs for computation
    var = {
        'vars'    : ['U', 'V', 'RHO'],
        'fluctsof'   : ['U', 'V'],
        'kind'   : 'vec',
        'dyad_name'    : ['U_f', 'V_f']
        }

    mo.compute_dyads(var)
    update.rm_entry('data_run', ['U','V'])
    ## Do the active wind dyads is written to [1]

    var = {
        'vars'    : ['G1_X', 'G1_Y', 'RHO'],
        'fluctsof'   : ['G1_X', 'G1_Y'],
        'kind'   : 'vec',
        'dyad_name'    : ['G1_X_f', 'G1_Y_f']
        }

    mo.compute_dyads(var)
    update.rm_entry('data_run', ['G1_X', 'G1_Y', 'RHO'])

    return None

def do_turb_diss():
    update = up.Updater()

    var = {
        'index' : 0,
        'name' : 't_diss'
        }
    po.turb_diss(var)
    var = {
        'index' : 1,
        'name' : 't_diss_ac'
        }
    po.turb_diss(var)
    update.up_entry('data_run',
                    {'t_diss_ac' : np.divide(
                        gv.globals_dict['data_run']['t_diss_ac'][1],
                        np.square(gv.globals_dict['data_run']['ZSTAR_bar'])
                    )})
    return None


# changed according to new array structure - check
def perform(kwargs):
    t1 = count_time()
    update = up.Updater()
    '''performs the compuations neccessary'''
    # compute u and v hat and bar
    print('--------------')
    print('averaging variable fields')
    t2 = count_time()
    do_the_average() #is ok maybe...
    t2 = count_time(t2)
    print('multiprocessing took {} minutes').format(t2/60)
    # '_bar' and '_hat' contain averages
    # compute gradient
    print('--------------')
    print('computing the gradients')
    t2 = count_time()
    do_the_gradients()
    t2 = count_time(t2)
    print(t2/60)

   # print('Speedup: {}').format(t2-t1)
    # 'gradient' [ncells, 0:1, 0:1, ntim, nlev]
    # 0:1 : d/dx d/dy; 0:1 : u, v
    # compute and average the dyads plus comute their primes
    print('--------------')
    print('computing the dyads')
    #update.up_mp({'mp' : False})
    do_the_dyads()

    # 'dyad' [ncells, ntim, nlev, 0:1, 0:1]
    print('--------------')
    print('computing the turbulent disstion')

    do_turb_diss()

    print('--------------')
    print('computing the dissipation coeficient')
    po.K('t_diss')
    po.K('t_diss_ac')

    update.up_entry('data_run', {
        't_diss': gv.globals_dict['data_run']['t_diss'],
        't_diss_ac': gv.globals_dict['data_run']['t_diss_ac'],
    })

    for key in gv.globals_dict['data_run'].iterkeys():
        print key

    t1 = count_time(t1)
    print('computation finished after {}').format(t1/60)
    return None

if __name__ == '__main__':
   # kwargs = user()
    update = up.Updater()
    kwargs = {
        'experiment' : 'BCWcold',
        'file_name' : 'BCWcold_R2B07_slice_onestep',
        'num_rings' : 4,
        'num_files' : 10
    }
    kwargs['filep'] = u'/home1/kd031/projects/icon/experiments/'+kwargs['experiment']+'/'
    kwargs['files'] = [
        n for n in
        glob.glob(kwargs['filep']+kwargs['file_name']+'.nc') if
        os.path.isfile(n)]
    new = []
    for i, xfile in enumerate(kwargs['files']):
        if not 'refined' in xfile:
            new.append(xfile)
    kwargs['files'] = new
    del new
    for xfile in kwargs['files']:
        print(xfile)
    kwargs['variables'] = ['U', 'V', 'G1_X', 'G1_Y', 'ZSTAR', 'RHO', 'THETA_V', 'EXNER']

    if not kwargs['files']:
        sys.exit('Error:missing gridfiles or datafiles')

    print('reading gridfile')
    grid = read_grid(kwargs)

    grid_nfo = {
        'area_neighbor_idx'   : grid['area_member_idx'].values,
        'cell_area'           : grid['cell_area'].values,
        'coarse_area'         : grid['coarse_area'].values,
        }

    for var in grid_nfo.iterkeys():
        grid_nfo[var] = np.moveaxis(grid_nfo[var], -1, 0)

    gradient_nfo = {
        'coords' : grid['coords'].values,
        'member_idx' : grid['member_idx'].values,
        'member_rad' : grid['member_rad'].values
    }
    update.complete('gradient_nfo', gradient_nfo)
    update.complete('grid_nfo', grid_nfo)

    del grid, grid_nfo, gradient_nfo

    # define number of loops, through number of files.

    if kwargs['num_files'] > len(kwargs['files']):
        fin = len(kwargs['files'])
    else:
        fin = kwargs['num_files']

    for i in range(fin):

        print ('file {} of {}').format(i+1, fin)
        # read the data from file
        data = read_data(kwargs, i)


        # update the grid_nfo
        grid_nfo = {
            'i_cell' : 0,
            'ntim'                : data.dims['time'],
            'nlev'                : data.dims['lev'],
            'ncells'              : data.dims['ncells'],
            'lat'                 : data['clat'].values,
            'lon'                 : data['clon'].values
            }
        update.up_entry('grid_nfo', grid_nfo)
        del grid_nfo

        # give run parameters as output, just so something nice happens on
        # screen...
        print('time {}, levels {} and cells {}').format(
            gv.globals_dict['grid_nfo']['ntim'],
            gv.globals_dict['grid_nfo']['nlev'],
            gv.globals_dict['grid_nfo']['ncells']
        )

        # extract variables neccessary for computation
        data_run = {}

        # move last axisi to first position, (ncells to first position)
        for var in kwargs['variables']:
            data_run[var] = np.moveaxis(data[var].values, -1, 0)
        for key in data_run.iterkeys():
            print key
        #delete data this doubles the read operation but goes easy on memory
        del data # just to cleanup memory
        update.complete('data_run', data_run)
        del data_run # keep that thing clean!

        # preliminary w20ork towards multiprocessing, not yet properly implemented
        num_procs = 20
        update.up_mp(
            {
                'mp' : True
            }
        )
        if gv.mp['mp']:
            dop.prepare_mp(num_procs)

        #compute Temperature T for computation
        print('-'*10)
        print('computing the Temperature field')

        po.potT_to_T_exner()

        update.rm_entry(
            'data_run',
            ['EXNER', 'THETA_V']
        )

        # Do it all
        perform(kwargs)

        # move cells axis back to original position.
        data_out = {}
        keys_for_out=['t_diss', 't_diss_ac', 'T', 'Kt_diss', 'Kt_diss_ac', 'U_hat', 'V_hat', 'RHO_bar']
        for key, item in gv.globals_dict['data_run'].iteritems():
            if key in keys_for_out:
                data_out[key] = np.moveaxis(item, 0, -1)
        update.rm_all('data_run')
        update.up_entry('data_run', data_out)
        del data_out


        print('min {} and max {}').format(
            np.min(gv.globals_dict['data_run']['t_diss']),
            np.max(gv.globals_dict['data_run']['t_diss'])
        )
        # the question remains IF those are really entropy production rates...
        print('globally averaged entropy production rate: {}').format(
            np.mean(gv.globals_dict['data_run']['t_diss'])
        )

        # there may yet be a more concise and effective way to do this.
        # technically there is no need to read the whole data set.
        # just write to file whatever we needed...

        data = read_data(kwargs, i)
        # define file output name
        kwargs['files'][i] = kwargs['files'][i][:-3]+'_refined_{}.nc'.format(kwargs['num_rings'])

        U_hat = xr.DataArray(
            gv.globals_dict['data_run']['U_hat'],
            dims = ['time', 'lev', 'ncells']
        )
        U_hat.attrs['grid_type'] = "unstructured"
        U_hat.attrs['long_name'] = "U density weighted coarse"
        update.rm_entry('data_run', ['U_hat'])

        V_hat = xr.DataArray(
            gv.globals_dict['data_run']['V_hat'],
            dims = ['time', 'lev', 'ncells']
        )
        V_hat.attrs['grid_type'] = "unstructured"
        V_hat.attrs['long_name'] = "U density weighted coarse"
        update.rm_entry('data_run', ['V_hat'])
        RHO_bar = xr.DataArray(
            gv.globals_dict['data_run']['RHO_bar'],
            dims = ['time', 'lev', 'ncells']
        )
        RHO_bar.attrs['grid_type'] = "unstructured"
        RHO_bar.attrs['long_name'] = "RHO coarse"
        update.rm_entry('data_run', ['RHO_bar'])

        t_diss = xr.DataArray(
            gv.globals_dict['data_run']['t_diss'],
            dims = ['time', 'lev', 'ncells']
        )
        t_diss.attrs['grid_type'] = "unstructured"
        t_diss.attrs['long_name'] = "turbulent dissipaion"
        update.rm_entry('data_run', ['t_diss'])

        t_diss_ac = xr.DataArray(
            gv.globals_dict['data_run']['t_diss_ac'],
            dims = ['time', 'lev', 'ncells']
        )
        t_diss_ac.attrs['grid_type'] = "unstructured"
        t_diss_ac.attrs['long_name'] = "turbulent dissipation active"
        update.rm_entry('data_run', ['t_diss_ac'])

        K = xr.DataArray(
            gv.globals_dict['data_run']['Kt_diss'],
            dims = ['time', 'lev', 'ncells']
        )
        K.attrs['grid_type'] = "unstructured"
        K.attrs['long_name'] = "K value"
        update.rm_entry('data_run', ['Kt_diss'])

        K_ac = xr.DataArray(
            gv.globals_dict['data_run']['Kt_diss_ac'],
            dims = ['time', 'lev', 'ncells']
        )
        K_ac.attrs['grid_type'] = "unstructured"
        K_ac.attrs['long_name'] = "K value active"
        update.rm_entry('data_run', ['Kt_diss_ac'])

        T = xr.DataArray(
            gv.globals_dict['data_run']['T'],
            dims = ['time', 'lev', 'ncells']
        )
        T.attrs['grid_type'] = "unstructured"
        T.attrs['long_name'] = "Temperature"
        update.rm_entry('data_run', ['T'])

        data = data.assign(U_hat = U_hat)
        del U_hat
        data = data.assign(V_hat = V_hat)
        del V_hat
        data = data.assign(RHO_bar = RHO_bar)
        del RHO_bar
        data = data.assign(t_diss = t_diss)
        del t_diss
        data = data.assign(t_diss_ac = t_diss_ac)
        del t_diss_ac
        data = data.assign(T = T)
        del T
        data = data.assign(K = K)
        del K
        data = data.assign(K_ac = K_ac)
        del K_ac

        cio.write_netcdf(kwargs['files'][i], data)

        update.rm_all('data_run')


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


