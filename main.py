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

def do_the_hat(data, grid, kwargs):
  '''computes the u and v hat values'''
  for var in kwargs['variables']:
    #....

  return data

def do_the_gradients(data, grid, kwargs):
  '''computes the gradients of u an v'''
  return data

def do_the_dyads(data, grid):
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
      'RHO'     : {'vars'   : ['RHO'],
                   'kind'   : 'scalar'
                   }
      'dyad'    : {'vars'   : ['U_f', 'V_f']
                   }
      }

  # start cellwise iteration 
  for i in range(grid['ncells']):
    # get area members
    values = dop.get_members(grid_dic, dat_dic, i, kwargs['vars'])
    # add lat lon coordinates to values
    values.update(dop.get_members(grid_dic, dat_dic, i, ['lat', 'lon']))
    # get individual areas
    values.update(dop.get_members(grid_dic, grid_dic, i, ['area']) )
    # get coarse values
    for var in kwargs['vars']:
      values[var+'_bar']    = data[var+'_bar'].values[:,:,i]
    values = mo.compute_flucts(values, grid_dic, grid_dic['num_area_hex'][i], **kwargs['UV'])
    values = mo.compute_flucts(values, grid_dic, i, **kwargs['RHO'])
    values = mo.compute_dyads(values, grid_dic, i, **kwargs['dyad'])
    values = values / grid_dic[''] 
  return data

def perform(data, grid, kwargs):
  # compute u and v hat
  data = do_the_hat(data, grid, kwargs)
  # compute gradient
  data = do_the_gradients(data, grid, kwargs)
  # compute and average the dyads plus comute their primes
  data = do_the_dyads(data, grid, kwargs)
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
  
  if kwargs['num_files'] > len(kwargs['files']):
    fin = len(kwargs['files'])
  else:
    fin = len(kwargs['num_files'])

  for i in range(fin):
    print ('file {} of {}').format(i, fin)
    data = read_data(kwargs, i)
    data_run = {}
    for var in kwargs['variables']:
      data_run[var] = data[var].values

    data = perform(data_run, grid, kwargs)
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


