# all math Operations neccessary for computations

# I want to represent each of these as matrices. 
# what is more effective?
# memory wise: 
#           computing smaller matrices on the fly
#           I'd need gridsize times 3 matrices,
#           1 - rhov''v''
#           2 - delv
#           3 - v (to compute del v) so this is a throw away matrix
# MISSING:
# *  Routine for projection of vectors (see ICON code)
# *  Routine for conversion of lat-lon to carthesian coordinates and back (for
#   projection)
# *  
# TODO:
# *  Find out about Python parallelization to speed this up
# *  
import os
import xarray as xr
import numpy as np
import data_op as dop

def coarse_area(grid):
  """Sums the coarse_area from the areas of its members"""
  co_ar = np.array(
            [0.0 for i in range(0,grid.dims['ncells'])]
            )
  cell_a= grid['cell_area'].values
  a_nei_idx = grid['area_neighbor_idx'].values
  a_num_hex = grid['area_num_hex'].values

  for i in range(0, grid.dims['ncells']):
    for j in range(0, a_num_hex[i]):
        ij = a_nei_idx[j,i]
        co_ar[i] += cell_a[ij]
  
  coarse_area = xr.DataArray(
            co_ar,
            dims = ['ncells']
            )
  kwargs = {'coarse_area' : coarse_area}
  grid = grid.assign(**kwargs)
  
  return grid

def area_avg(kind, grid_nfo, data, var, vec = False):
  '''computes the bar average over all vars (multiplied)'''
  helper = {}
  name   = [] 
  dat_dic  = {}
  print var
  kwargs = {}
  
  if not vec:
    len_vec= 1
    name.append('')
    for va in var['vars']:
      name[0]  = name[0]+va
    name[0] = name[0]+'_'+kind
  else:
    len_vec= len(var['vector']) 
    name = var['vector'][:]
    for i in range(len(name)):
      name[i] = name[i]+'_'+kind
     
  print var
    #for speedup get info before loop starts.

  if kind == 'hat':

    func= lambda val, fac, kwargs: avg_hat(val, fac, **kwargs)
    kwargs = {
              'ntim': grid_nfo['ntim'],
              'nlev': grid_nfo['nlev']
        }
    if not('RHO' in data):
      sys.exit('ERROR: area_avg kind = "hat": missing "RHO"')
      print 'getting RHO'
    if not 'RHO_bar' in data.data_vars:
      print 'computing Rho_bar'
      data = area_avg('bar', grid_nfo, data, {'vars':['RHO']})
    print 'initialize..'
    factor = {}
    factor['RHO_bar']=data['RHO_bar']
    factor['coarse_area']= grid_nfo['coarse_area']
    factor=  mult(factor)

  elif kind == 'bar':
    print 'in bar'
    func= lambda val, fac, kwargs: avg_bar(val, fac, **kwargs)
    factor= grid_nfo['coarse_area']

  else:
    print 'ERROR area_avg: unknown averaging type'
    return None

  if vec:
    if not(all(key in grid_nfo for key in ['lat', 'lon'])):
      sys.exit('ERROR: avg_area kind kind = vec, missing "lat" or "lon"')
    kwargs = {
        'grid_nfo' : grid_nfo,
        'vec'      : var['vector'], 
        'func'     : func,
        'kwargs'   : kwargs
              }
    func= lambda val, fac, kwargs: avg_vec(val, fac, **kwargs)

  dims   = [len_vec,grid_nfo('ntim'),grid_nfo('nlev'),grid_nfo('ncells')]
  stack  = np.empty(dims, order='F')

  #create kwargs for get_members:
  for i in range(0, grid_nfo['ncells'] ):
    values=  dop.get_members(grid_nfo, data, i, var['vars'])  
    values.update(dop.get_members(grid_nfo, data, i, ['cell_area']))
    # divide by factor (area or rho_bar) 
    kwargs['i'] = i
    values= func(values, factor, kwargs) 
    stack[:,:,:,i] = values
  # put into xArray
  for i in range(len_vec):
    data[name[i]] = stack[i,:,:,:] 
  return data

# call functions for area_avg
# -----
def avg_bar(values, factor, i):
  # multiply var area values (weighting) 
  values=  mult(values)
  # Sum Rho*var(i) up
  values=  np.sum(values,0)
  return values/factor[i]

def avg_hat(values, factor, i, ntim, nlev):
  # multiply var area values (weighting) 
  values=  mult(values)
  # Sum Rho*var(i) up
  values=  np.sum(values,0)
  for k in range(ntim):
    for j in range(nlev):
      values[k,j] = values[k,j]/factor[k,j,i]
  return values
# -----

def avg_vec(values, factor, i, grid_nfo, dat_dic, vec, func, kwargs):
  kwargs['i'] = i
  coords =  dop.get_members(grid_nfo, dat_dic, i, ['lat','lon'])
  # rotating vectors.
  rot_vec = np.empty([len(vec),grid_nfo['area_num_hex'][i], grid_nfo['ntim'], grid_nfo['nlev']])
  rot_vec.fill(0)
  rot_vec[:,:,:,:]  = rotate_vec(
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
  for j in range(len(vec)):
    values_vec.append({vec[j] : rot_vec[j,:,:,:]})
    values_vec[j].update(help_dic)

  # computing averages
  helper = np.empty([len(vec),grid_nfo['ntim'],grid_nfo['nlev']]) 
  for j in range(len(vec)):
    # func is either avg_hat or avg_bar
    helper[j,:,:]  = func(values_vec[j], factor, kwargs)
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
  rot_vec[:,:,:,:]  = rotate_vec(
                   values['lon'], values['lat'],
                   values[vars[0]][:,:,:], values[vars[1]][:,:,:]
                   )
  result    = np.empty([num_hex, grid_dic['ntim'], grid_dic['nlev']])
  for i in range(len(vars) ):
    result.fill(0)
    for j in range(num_hex): # a bit ad hoc
      result[j,:,:]     = rot_vec[i, j, :, :] - values[vars[i]+'_bar'][:,:]
    values[vars[i]+'_f'] = result

   return values

def compute_dyads(values, grid_dic, vars):
  '''computes the dyadic products of v'''
  if not(all(key in values for key in ['RHO', 'cell_area'])):
    sys.exit('ERROR: compute_dyads, "RHO" or "cell_area" missing in values')
  l_vec   = len(vars)
  product = np.empty([l_vec,l_vec,grid_dic['area_num_hex'],grid_dic['ntim'],grid_dic['nlev']])
  product.fill(0)
  constituents = []
  for var in vars:
    constituents.append(values[var])
  for i in range(l_vec):
    for j in range(l_vec):
      product[i+j,:,:,:] = constituents[i] * constituents[j] * values['RHO']
  out = {}
  out['product'] = product
  # This is not working. due to i, j
  #use mult here
  out['cell_area'] = values['cell_area']
  out = avg_bar(out, grid_dic['coarse_area'], grid_dic['i_cell']  )
  return values

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

def rotate_latlon(lat, lon, plat, plon, x, y):
  ''' rotates lat and lon for better bilinear interpolation
      taken from ICON 
      may be disfunctional!'''

  rotlat    = np.asin(
                np.sin(lat)*np.sin(plat) +
                np.cos(lat)*np.cos(plat)*cos(lon-plon)
                )
  rotlon    = np.atan2(
                np.cos(lat)*np.sin(lon-plon),
                (np.cos(lat)*np.sin(plat)*np.cos(lon-plon)-np.sin(lat)*np.cos(plat))
                )

  return rotlat, rotlon

def rotate_vec(lon, lat, x,y):
  ''' rotates vectors using rotat_latlon_vec'''
  len_x = x.shape[0] 
  plon = lon[0]
  plat = lat[0]
  x_tnd= np.empty(x.shape)
  y_tnd= np.empty(y.shape)
  x_tnd.fill(0)
  y_tnd.fill(0)
  x_tnd[0,:,:]  = x[0,:,:] 
  y_tnd[0,:,:]  = y[0,:,:] 
  # do only for all except center value...
  for i in range(1,len_x):
    sin_d, cos_d = rotate_latlon_vec(lon[i],lat[i],plon,plat)
    x_tnd[i,:,:] = x[i,:,:]*cos_d -y[i,:,:]*sin_d
    y_tnd[i,:,:] = x[i,:,:]*sin_d +y[i,:,:]*cos_d

  return x_tnd, y_tnd

  
def rotate_latlon_vec(lon, lat, plon, plat):
  '''gives entries of rotation matrix for vector rotation
     taken from ICON code '''

  z_lamdiff = plon - lon
  z_a       = np.cos(plat)*np.sin(z_lamdiff)
  z_b       = np.cos(lat)*np.sin(plat)-np.sin(lat)*np.cos(plat)*np.cos(z_lamdiff)
  z_sq      = np.sqrt(z_a*z_a + z_b*z_b)
  sin_d     = z_a/z_sq
  cos_d     = z_b/z_sq
  
  return sin_d, cos_d

