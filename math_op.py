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
import xarray as xr
import numpy as np
import data_op as dop

def coarse_area(grid):
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

def area_avg(kind, grid, data, var):
  helper = {}
  values = []
  names  = []
  if kind=='hat':
    if not 'RHO_bar' in data.data_vars:
      area_avg('bar', grid, data, 'RHO')

    name= var+'_hat'
    names =  [var, 'RHO', 'area']
    factor= data['RHO_bar'].values

  if kind=='bar':
    name= var+'_bar'
    names =  [var, 'area']
    factor= grid['coarse_area'].values

  helper[name] = np.array([])
  for i in range(0, grid.dims['ncells']):
    values=  dop.get_members(grid, data, i, names)  
    values=  mult(values)
    values=  sum(values)
    values=  values/factor[i]
    helper[name]  = np.append(data, values)
  return data.assign(helper)

def area_avg_vec(kind, grid, data, variables):
  helper = {}
  values = []
  names  = [var for var in variables]
  name   = variables
  if kind=='hat':
    if not 'RHO_bar' in data.data_vars:
      area_avg('bar', grid, data, 'RHO')

    for i in range(len(name)):
      name[i]= name[i]+'_hat'
    names = np.append(names,['RHO', 'area'])
    factor= data['RHO_bar'].values

  if kind=='bar':
    name= var+'_bar'
    names =  np.append(var, ['area'])
    factor= grid['coarse_area'].values

  helper[name] = np.array([])
  for i in range(0, grid.dims['ncells']):
    values  =  dop.get_members(grid, data, i, names)  
    lat, lon=  dop.get_members(grid, data, i, ['lat','lon'])
    values[0:2] = rotate_vec(lon, lat, values[0:2])
    valuesa = np.array([values[0], values[2:]])
    valuesb = np.array([values[1], values[2:]])
    
    valuesa  =  mult(valuesa)
    valuesb  =  mult(valuesb)
    valuesa =  sum(valuesa)
    valuesb =  sum(valuesb)
    valuesa =  valuesa/factor[i]
    valuesb =  valuesb/factor[i]
    helper[name[0]]  = np.append(data, valuesa)
    helper[name[1]]  = np.append(data, valuesb)

  return data.assign(helper)


def mult(dataset):
  helper = 1
  for data in dataset:
    helper = np.multiply(helper, data)  
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

def rotate_vecs(lon, lat, x,y, plon=lon[0],plat=lat[0]):

  for i in range(len(x)):
    sin_d, cos_d = rotate_latlon_vec(lon[i],lat[i],plon,plat)
    x_tnd[i] = x[i]*cos_d -y[i]*sin_d
    y_tnd[i] = x[i]*sin_d +y[i]*cos_d

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

