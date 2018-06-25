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
        'vars'      :[['U'], ['V'], ['T'], ['RHO']],
        'vector'    :['U', 'V']
        }
    mo.area_avg('bar', var)
    # this must become more involved for mp

    # compute \hat{U} and \hat{V}
    print('  ------------')
    print('  computing hat averages')
    var = {
        'vars'      :[['U'], ['V'], ['T']],
        'vector'    :['U', 'V']
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
    update = up.Updater()

    update.up_entry('data_run', {'gradient':  mo.gradient()})

    print('mean velocity gradient u_x: {}').format(np.mean(gv.globals_dict['data_run']['gradient'][0, 0]))
    print('mean velocity gradient u_y: {}').format(np.mean(gv.globals_dict['data_run']['gradient'][0, 1]))
    print('mean velocity gradient v_x: {}').format(np.mean(gv.globals_dict['data_run']['gradient'][1, 0]))
    print('mean velocity gradient v_y: {}').format(np.mean(gv.globals_dict['data_run']['gradient'][1, 1]))
    return None

# changed according to new array structure - check
def do_the_dyads():
    '''computes the dyadic product of uv and rho plus averaging'''
    # check if everything is there
    needs = ['U', 'U_bar', 'V', 'V_bar', 'RHO', 'RHO_bar']
    if not all(need in data for need in needs):
        this = [need for need in needs if need not in data]
        sys.exit('ERROR: do_the_dyads I miss quantities to do the computing {}'.format(this))
      # define kwargs for computation
    kwargs = {
        'vars'    : ['U', 'V', 'RHO'],
        'UV'      : {'variables'   : ['U', 'V'],
                     'kind'   : 'vec'
                    },
        'dyad'    : {'variables'   : ['U_f', 'V_f']
                    }
        }

      #start cellwise iteration
    doprint = 5000
    if not('dyad' in data):
        l_vec = len(kwargs['UV']['vars'])
        # in case of first call build output file
        # output slot, outgoing info
        print('creating array values["dyad"]')
        out = np.zeros([
            grid_nfo['ncells'],
            l_vec,
            l_vec,
            grid_nfo['ntim'],
            grid_nfo['nlev']
        ])

    for i in range(grid_nfo['ncells']):
        if i == doprint:
            print('cell {} of {}').format(i, grid_nfo['ncells'])
            doprint = doprint + 5000
        # get area members
        values = do.get_members('data_run', i, kwargs['vars'])
        # add lat lon coordinates to values, and areas
        values.update(do.get_members('grid_nfo', i, ['lat', 'lon', 'cell_area']))
        # get coarse values
        for var in kwargs['vars'][:2]:
            values[var+'_hat'] = data[var+'_hat'][i, :, :]
        # compute fluctuations
        values = mo.compute_flucts(values, **kwargs['UV'])
        out[i, :, :, :, :] = mo.compute_dyads(values, i, **kwargs['dyad'])

    return data

# changed according to new array structure - check
def perform(kwargs):
    update = up.Updater()
    '''performs the compuations neccessary'''
    # compute u and v hat and bar
    print('--------------')
    print('averaging variable fields')
    t2 = count_time()
    print(time.time())
    do_the_average() #is ok maybe...
    t2 = count_time(t2)
    print(time.time())
    print(t2)
    # '_bar' and '_hat' contain averages
    # compute gradient
    print('--------------')
    print('computing the gradients')
    t2 = count_time()
    print(time.time())

    do_the_gradients()
    t2 = count_time(t2)
    print(time.time())
   # print('Speedup: {}').format(t2-t1)
    # 'gradient' [ncells, 0:1, 0:1, ntim, nlev]
    # 0:1 : d/dx d/dy; 0:1 : u, v
    # compute and average the dyads plus comute their primes
    print('--------------')
    print('computing the dyads')
    do_the_dyads()
    # 'dyad' [ncells, ntim, nlev, 0:1, 0:1]
    for key in gv.globals_dict['data_run'].iterkeys():
        print key
    print('--------------')
    print('computing the turbulent friction')
    update.up_entry(
        'data_run',
        {
            'turb_fric' : np.zeros([
                gv.globals_dict['grid_nfo']['ncells'],
                gv.globals_dict['grid_nfo']['ntim'],
                gv.globals_dict['grid_nfo']['nlev']
            ])
        }
    )

    update.up_entry(
        'data_run',
        {
            'turb_fric' : np.einsum(
                'klmij,klmij->klm',
                gv.globals_dict['data_run']['dyad'],
                gv.globals_dict['data_run']['gradient']
            )
        }
    )

    update.up_entry(
        'data_run',
        {'turb_diss' : -1 * np.divide(
            gv.globals_dict['data_run']['turb_fric'],
            gv.globals_dict['data_run']['RHO_bar']
        )}
    )

    update.up_entry(
        'data_run',
        {'turb_fric' : -1 * np.divide(
            gv.globals_dict['data_run']['turb_fric'],
            gv.globals_dict['data_run']['T_bar']
        )}
    )
    return None

if __name__ == '__main__':
   # kwargs = user()
    update = up.Updater()
    kwargs = {
        'experiment' : 'BCW_CG_15_days',
        'file_name' : 'time_slice',
        'num_rings' : 3,
        'num_files' : 9
    }
    kwargs['filep'] = u'/home1/kd031/projects/icon/experiments/'+kwargs['experiment']+'/'
    kwargs['files'] = [
        n for n in
        glob.glob(kwargs['filep']+kwargs['file_name']+'_*.nc') if
        os.path.isfile(n)]

    print kwargs['files']
    kwargs['variables'] = ['U', 'V', 'RHO', 'THETA_V', 'EXNER']

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

        # define file output name
        kwargs['files'][i] = kwargs['files'][i][:-3]+'_refined_{}.nc'.format(kwargs['num_rings'])

        # update the grid_nfo
        grid_nfo = {
            'i_cell' : 0,
            'ntim'                : data.dims['time'],
            'nlev'                : data.dims['lev_2'],
           # 'ncells'              : data.dims['cell'],
            'ncells'              : 1000,
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
        #delete data this doubles the read operation but goes easy on memory
        del data # just to cleanup memory
        update.complete('data_run', data_run)
        del data_run # keep that thing clean!

        # preliminary work towards multiprocessing, not yet properly implemented
        num_procs = 5
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
        update.up_entry(
            'data_run',
            {'T' : po.potT_to_T_exner(
                gv.globals_dict['data_run']['THETA_V'],
                gv.globals_dict['data_run']['EXNER']
            )}
        )
        update.rm_entry(
            'data_run',
            ['EXNER', 'THETA_V']
        )

        perform(kwargs) #gibt ( dann) keinen output mehr, weil alles in global steht...

        # move cells axis back to original position.
        data_out = {}
        for key, item in gv.globals_dict.iteritems():
            data_out[key] = np.moveaxis(item, 0, -1)

        update.up_entry('data_run', data_out)
        del data_out


        print('min {} and max {}').format(
            np.min(gv.globals_dict['turb_fric']),
            np.max(gv.globals_dict['turb_fric'])
        )
        # the question remains IF those are really entropy production rates...
        print('globally averaged entropy production rate: {}').format(
            np.mean(gv.globals_dict['turb_fric'])
        )

        # there may yet be a more concise and effective way to do this.
        # technically there is no need to read the whole data set.
        # just write to file whatever we needed...

        data = read_data(kwargs, i)

        t_fric = xr.DataArray(
            gv.globals_dict['turb_fric'],
            dims = ['time', 'lev_2', 'cell']
        )

        t_diss = xr.DataArray(
            gv.globals_dict['turb_diss'],
            dims = ['time', 'lev_2', 'cell']
        )

        T = xr.DataArray(
            gv.globals_dict['T'],
            dims = ['time', 'lev_2', 'cell']
        )

        data = data.assign(t_fric = t_fric)
        data = data.assign(t_diss = t_diss)
        data = data.assign(T = T)

        cio.write_netcdf(kwargs['files'][i], data)


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


