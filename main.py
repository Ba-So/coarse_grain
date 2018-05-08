'''The main of the Coarse-Graining Post Processing Program
    The actual computation of thetrubulent diffusion is done here.'''
import os
import sys
import glob
import numpy as np
import custom_io as cio
import data_op as dop
import math_op as mo
import phys_op as po
import xarray as xr

# the main file
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
    doprint = 1000
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
            doprint = doprint + 1000
        # get area members
        values = dop.get_members(grid_nfo, data, i, kwargs['vars'])
        # add lat lon coordinates to values
        values.update(dop.get_members(grid_nfo, grid_nfo, i, ['lat', 'lon']))
        # get individual areas
        values.update(dop.get_members(grid_nfo, grid_nfo, i, ['cell_area']))
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
    data = do_the_gradients(data, grid_nfo, gradient_nfo, kwargs)
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
    data['turb_fric'] = -1 * np.divide(data['turb_fric'], data['T_bar'])
    return data

if __name__ == '__main__':
   # kwargs = user()
    kwargs = {
        'experiment' : 'HS_FT_6000_days',
        'num_rings' : 3,
        'num_files' : 1
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
        t_fric = xr.DataArray(
            data_run['turb_fric'],
            dims = ['time', 'lev', 'ncells']
        )

        T = xr.DataArray(
            data_run['T'],
            dims = ['time', 'lev', 'ncells']
        )

        data = data.assign(t_fric = t_fric)
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

