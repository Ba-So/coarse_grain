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
import sys
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

  if kind == 'hat':

    func= lambda val, fac, kwargs: avg_hat(val, fac, **kwargs)
    kwargs = {
              'ntim': grid_nfo['ntim'],
              'nlev': grid_nfo['nlev']
        }
    # fatal: RHO missing
    if not('RHO' in data):
      sys.exit('ERROR: area_avg kind = "hat": missing "RHO"')

    if not('RHO' in var['vars']):
      var['vars'].append('RHO')
    
    # semi fatal: RHO_bar missing
    if not 'RHO_bar' in data:
      data = area_avg('bar', grid_nfo, data, {'vars':['RHO']})

    factor = {}
    factor['RHO_bar']=data['RHO_bar']
    factor['coarse_area']= grid_nfo['coarse_area']
    factor=  mult(factor)

  elif kind == 'bar':
    func= lambda val, fac, kwargs: avg_bar(val, fac, **kwargs)
    factor= grid_nfo['coarse_area']

  else:
    print 'ERROR area_avg: unknown averaging type'
    return None

  if vec:
    if not(all(key in grid_nfo for key in ['lat', 'lon'])):
      sys.exit('ERROR: avg_area kind kind = vec, missing "lat" or "lon"')
    kwargs = {
        'i_cell'   : 0,
        'grid_nfo' : grid_nfo,
        'vec'      : var['vector'], 
        'func'     : func,
        'kwargs'   : kwargs
              }
    func= lambda val, fac, kwargs: avg_vec(val, fac, **kwargs)
  dims   = [len_vec,grid_nfo['ntim'],grid_nfo['nlev'],grid_nfo['ncells']]
  stack  = np.empty(dims, order='F')

  #create kwargs for get_members:
  for i in range(0, grid_nfo['ncells'] ):
    values=  dop.get_members(grid_nfo, data, i, var['vars'])  
    values.update(dop.get_members(grid_nfo, grid_nfo, i, ['cell_area']))
    # divide by factor (area or rho_bar) 
    kwargs['i_cell'] = i
    values= func(values, factor, kwargs) 
    stack[:,:,:,i] = values
  # put into xArray
  for i in range(len_vec):
    data[name[i]] = stack[i,:,:,:] 
  return data

# call functions for area_avg
# -----
def avg_bar(values, factor, i_cell):
  # multiply var area values (weighting) 
  values=  mult(values)
  # Sum Rho*var(i) up
  values=  np.sum(values,0)
  return values/factor[i_cell]

def avg_hat(values, factor, i_cell, ntim, nlev):
  # multiply var area values (weighting) 
  values=  mult(values)
  # Sum Rho*var(i) up
  values=  np.sum(values,0)
  for k in range(ntim):
    for j in range(nlev):
      values[k,j] = values[k,j]/factor[k,j,i_cell]
  return values
# -----

def avg_vec(values, factor, i_cell, grid_nfo, vec, func, kwargs):
  kwargs['i_cell'] = i_cell
  coords =  dop.get_members(grid_nfo, grid_nfo, i_cell, ['lat','lon'])
  # rotating vectors.
  rot_vec = np.empty([len(vec),
                      grid_nfo['area_num_hex'][i_cell], 
                      grid_nfo['ntim'], 
                      grid_nfo['nlev']
                      ])
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

def compute_dyads(values, grid_nfo, i_cell, vars):
  '''computes the dyadic products of v'''
  if not(all(key in values for key in ['RHO'])):
    sys.exit('ERROR: compute_dyads, "RHO" missing in values')
  if not(all(key in grid_nfo for key in ['cell_area'])):
    sys.exit('ERROR: compute_dyads, "cell_area" missing in grid_nfo')

  # dimension of dyad
  l_vec   = len(vars)

  if not('dyad' in values):
    # in case of first call build output file
    # output slot, outgoing info
    values['dyad']        = np.empty([
      l_vec,
      l_vec,
      grid_nfo['ntim'],
      grid_nfo['nlev'],
      grid_nfo['ncells']
     ])
  # the product of dyadic multiplication (probably term dyadic is misused here)
  product = np.empty([
                    l_vec,
                    l_vec,
                    grid_nfo['area_num_hex'][i_cell],
                    grid_nfo['ntim'],
                    grid_nfo['nlev']
                    ])

  product.fill(0)
  # helper, containing the constituents for computation 
  constituents = []
  for var in vars:
    constituents.append(values[var])
  # helper as handle for avg_bar, values in it are multiplied and averaged over
  helper                = {}
  helper['cell_area']   = values['cell_area']
  helper['RHO']         = values['RHO']

  for i in range(l_vec):
    for j in range(l_vec):
      product[i,j,:,:,:] = constituents[i] * constituents[j]
  for i in range(l_vec):
    for j in range(l_vec):
      helper['product']    = product[i,j,:,:,:]
      values['dyad'][i,j,:,:,i_cell] = avg_bar(helper,
                                                    grid_nfo['coarse_area'],
                                                    i_cell
                                                    )
  return values

def gradient(data, grid_nfo, var):

  for i in range(grid_nfo['ncells']):
    # chekc for correct key
      # define distance
      # assume circle pi r^2 => d= 2*sqrt(A/pi)
    area    = grid_nfo['hex_area'][i]
    lonlat  = [grid_nfo['lon'][i], grid_nfo['lat'][i]]
    coords  = central_coords(lonlat, area)
    # check for correct keys
    area    = grid_nfo['cell_area'][i]
    circ_dist_avg(data, grid_nfo, coords, area, var)
  # find/interpolate value at distance d in x/y direction
  #     -> use geodesics / x/y along longditudes/latitudes
  # turn values at distance (with rot_vec)
  # compute d/dz use fassinterpol x_i-1+2x_i+x_i+1/dx including grid value?
  return data

def central_diff(xl, x, xr, d):
  return (xl-2*x+xr)/d**2

def radius(area):
  '''returns radius of circle on sphere in radians'''
  re        = 6371000
  r         = np.sqrt(area/np.pi)/ re
  return r

def central_coords(lonlat, area):
  # get radius
  r = radius(area)
  # coords[0,:] latitude values
  # coords[1,:] longditude values
  # u is merdional wind, along longditudes
  #     -> dx along longditude
  # v is zonal wind, along latitudes
  #     -> dy along latitude
  # coords[:,:2] values for dx
  # coords[:,:2] values for dy
  coords = np.array([lonlat for j in range(4)])
  dlon  = np.arcsin(np.sin(r)/np.cos(lonlat[0]))
  dlat  = r 
  coords[0,0] = coords[0,0] - dlon
  coords[1,0] = coords[1,0] + dlon
  coords[2,1] = coords[2,1] - dlat
  coords[3,1] = coords[3,1] + dlat
  return coords

def circ_dist_avg(data, grid_nfo, coords, area, var):
  values = {name: np.empty([4]) for name in var['values']}
  for name in var['values']:
      values[name].fill(0)
  #how large is check radius?
  candidates = [[],[],[],[]]
  member_idx = [[],[],[],[]]
  member_rad = [[],[],[],[]]
  members    = [[],[],[],[]]
  check_r    = radius(area)
  for j in range(4):
    maxcoords = central_coords(coords[j,:],area)
    # find candidates for members of circle at target coords
    for i_cell in range(grid_nfo['ncells']):
      check = (
         (maxcoords[0,0] <= grid_nfo['lat'][i] <= maxcoords[1,0])
         and
         (maxcoords[2,1] <= grid_nfo['lon'][i] <= maxcoords[3,1])
         )
      if check:
        candidates[j].append[i_cell]
    # verify candidates as members 
    for k in candidates[j]:
      r =  arc_len(coords[j,:], [grid_nfo['lon'][k], grid_nfo['lat'][k]])
      if r <= check_r:
        member_idx[j].append(k)
        member_rad[j].append(r)
    # now compute distance weighted average of area weighted members:
      members[j]    = do.get_members_idx(data, member_idx[j], var['vars'])
        
      # if you suffer from boredom: This setup gives a lot of flexebility.
      #     may be transferred to other parts of this program.
      if 'vec' in var.iterkeys():
          #turn vectors
          vec   = vars['vec']
          rot_vec = np.empty([len(vec),
                              grid_nfo['area_num_hex'][i_cell], 
                              grid_nfo['ntim'], 
                              grid_nfo['nlev']
                              ])
          rot_vec.fill(0)
          rot_vec[:,:,:,:]  = rotate_vec(
                           coords[j,0], coords[j,1],
                       members[j][vec[0]][:,:,:], members[vec[1]][:,:,:]
                       )
          members[j][vec[0]]   = rot_vec[0,:,:,:]
          members[j][vec[1]]   = rot_vec[1,:,:,:]

      helper         = dist_avg(members[j], member_idx[j], member_rad[j], grid_nfo, var['vars'])
      for name in var['vars']:
          values[name][j]   = helper[name]

  return values


def dist_avg(members, idxcs, radii, grid_nfo, vars):
  len_i  = len(idxcs)
  # define weights.
  weight = np.empty([len_i])
  weight.fill(0)
  factor = 0
  for k in range(len_i):
    weight[k] = grid_nfo['cell_area'][idxcs[k]]* radii[k]
    factor    = factor + weight[k]
  # compute distance averages.
  sum    = {name : 0 for name in vars}
  for k in vars:
    for i in range(len_i):
      sum[k]    = sum[k] + members[k][i]*weight[i]
      sum[k]    = sum[k]/factor 

  return sum 




def arc_len(p_x, p_y):
  '''computes length of geodesic arc on a sphere (in rad)'''
  r = np.arccos(
        np.sin(p_x[1])* np.sin(p_y[1])
       +np.cos(p_x[1])* np.cos(p_y[1]) * np.cos(p_y[0]-p_x[0])
       )
  return r

      

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

