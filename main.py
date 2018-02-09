import os
import glob

# the main file. 
def user():
  kwargs = {}
  print 'enter the experiment you wish to postprocess:'
  kwargs['experiment'] = raw_input()
  print 'how many rings of hexagons shall be averaged over?'
  while True:
    helper = int(input())
    if type(helper) == int:
      kwargs['num_rings'] = helper
      break
  print 'how many files shall be processed?'
  while True:
    helper = int(input())
    if type(helper) == int:
      kwargs['num_files'] = helper
      break
  return kwargs 

def read_grid(kwargs):
  switch = True
  for n in kwargs['grid']:
    if 'refined_{}'.format(kwargs['num_rings']) in n:
      switch = False
      kwargs['grid'] = [n]
  if switch:
    for n in kwargs['grid']:
      if not 'refined' in n:
        kwargs['grid'] = [n]
  if switch:
    quarks  = {
        'num_rings': kwargs['num_rings'],
        'path' : kwargs['']
        }
    return cio.read_netcdfs(kwargs['grid'], 'time', quarks, func= lambda
        ds, quarks: cio.read_grid(ds, **quarks))
  else:
    return cio.read_netcdfs(kwargs['grid'], 'time')

def read_data(kwargs, i):
  quarks['variables']  = kwargs['variables'] 
  return cio.read_netcdfs([kwargs['file'][i]], 'time', quarks, func= lambda ds, quarks:cio.
    extract_variables(ds, **quarks)) 

def do_the_average(data, grid_nfo, kwargs):
  '''computes the u and v hat values'''
  # fairly simple
  var={
      vars      :['U','V'],
      vec       :['U','V']
      }
  # compute \bar{U} and \bar{V}
  data = area_avg('bar', grid_nfo, data, var, True)
  # compute \hat{U} and \hat{V}
  data = area_avg('hat', grid_nfo, data, var, True)

  return data

def do_the_gradients(data, grid_nfo, kwargs):
  '''computes the gradients of u an v'''

  return data

def do_the_dyads(data, grid_nfo):
  '''computes the dyadic product of uv and rho plus averaging'''
  # check if everything is there
  needs = ['U','U_bar','V','V_bar','RHO', 'RHO_bar', 'lat', 'lon']
  if not all(need in data for need in needs):
    this = [need for need in needs if need not in data]
    sys.exit('ERROR: do_the_dyads I miss quantities to do the computing
        {}'.format(this))
  # define kwargs for computation
  kwargs    = {
      'vars'    : ['U', 'V', 'RHO']
      'UV'      : {'vars'   : ['U', 'V'],
                   'kind'   : 'vec'
                   }
      'dyad'    : {'vars'   : ['U_f', 'V_f']
                   }
      }

  # start cellwise iteration 
  for i in range(grid['ncells']):
    # get area members
    values = dop.get_members(grid_nfo, data, i, kwargs['vars'])
    # add lat lon coordinates to values
    values.update(dop.get_members(grid_nfo, data, i, ['lat', 'lon']))
    # get individual areas
    values.update(dop.get_members(grid_nfo, grid_nfnfoo, i, ['cell_area']) )
    # get coarse values
    for var in kwargs['vars'][:2]:
      values[var+'_bar']    = data[var+'_bar'][:,:,i]
    values = mo.compute_flucts(values, grid_nfo, grid_nfo['num_area_hex'][i], **kwargs['UV'])
    values = mo.compute_dyads(values, grid_nfo, i, **kwargs['dyad'])
  return data

def perform(data, grid_nfo, kwargs):
  # compute u and v hat and bar
  data = do_the_average(data, grid_nfo, kwargs)
  # compute gradient
  data = do_the_gradients(data, grid_nfo, kwargs)
  # compute and average the dyads plus comute their primes
  data = do_the_dyads(data, grid_nfo, kwargs)
  return data 







if __name__== '__main__':
  kwargs= user()
  kwargs['filep'] =  u'/home1/kd031/projects/icon/experiments/'+kwargs['experiment']+'/'
  kwargs['files'] = [
      n for n in
      glob.glob(kwargs['filep]'+kwargs['experiment']+'_*.nc') if
      os.path.isfile(n)]
  kwargs['grid'] = [
      n for n in
      glob.glob(kwargs['filep']+'*grid*.nc') if
      os.path.isfile(n)]
  kwargs['variables'] = ['U', 'V', 'RHO']
  if is_empty(kwargs['files']) or is_empty(kwargs['grid']:
     sys.exit('Error:missing gridfiles or datafiles')
  
  grid = read_grid(kwargs['grid'])
  grid_nfo={
      'area_num_hex'        : grid['area_num_hex'].values,
      'area_neighbor_idx'   : grid['area_neighbor_idx'].values,
      'coarse_area'         : grid['coarse_area'].values,
      'cell_area'           : grid['cell_area'].values,
      'i_cell'              : 0 
      }
  
  if kwargs['num_files'] > len(kwargs['files']):
    fin = len(kwargs['files'])
  else:
    fin = len(kwargs['num_files'])

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

    data = perform(data_run, grid_nfo, kwargs)
    write_netcdf(kwargs['files'][i]+'refined_{}'.format(kwargs['num_rings']),
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


