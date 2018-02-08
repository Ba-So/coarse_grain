import math_op as mo
import numpy as np
import custom_io as cio
import data_op as dop
import xarray as xr 


def test_coarse_area(files):
  #will no longer work
  grid = cio.read_netcdfs(files, 'time')
  grid = mo.coarse_area(grid)
  if isinstance(grid,xr.Dataset):
    return True
  else:
    print type(grid)
    return False

def test_area_avg_bar(files, grid):
  #will no longer work
  kwargs = {'variables' : ['U','RHO']} 
  data= cio.read_netcdfs(files, 'time', kwargs, func= lambda ds, kwargs:cio.
      extract_variables(ds, **kwargs)) 
  kind= 'bar'
  var = {
        'vars' : ['U']
      } 
  data= mo.area_avg(kind, grid, data, var)
  return data 

def test_area_avg_hat(files, grid):
  # will no longer work.
  kwargs = {'variables' : ['U','RHO']} 
  data= cio.read_netcdfs(files, 'time', kwargs, func= lambda ds, kwargs:cio.
      extract_variables(ds, **kwargs)) 
  kind= 'hat'
  var = {
        'vars' : ['U']
      } 
  data= mo.area_avg(kind, grid, data, var)
  return data 

def test_vec_avg_hat(files, grid):
  kwargs = {'variables' : ['U','V','RHO']} 
  data= cio.read_netcdfs(files, 'time', kwargs, func= lambda ds, kwargs:cio.
      extract_variables(ds, **kwargs)) 
  kind= 'hat'
  var = {
        'vars' : ['U','V'],
        'vector':['U','V']
      } 
  data= mo.area_avg(kind, grid, data, var, True)
  return data 

def test_rotate_latlon_vec(grid):
  data_dic = {}
  dat_dic ={      
          'lat': grid['lat'].values,
          'lon': grid['lon'].values
          }
  grid_nfo= {
            'area_num_hex' : grid['area_num_hex'].values,
            'area_neighbor_idx' : grid['area_neighbor_idx'].values,
            'ntim'              : data.dims['time'],
            'nlev'              : data.dims['lev']
            }
  i = 3
  dat_dic = dop.get_members(grid_nfo, dat_dic, i, ['lat','lon'])
  sin_cos = np.empty([2, grid_nfo['area_num_hex'][i]])
  sin_cos.fill(0)
  plon = dat_dic['lon'][0]
  plat = dat_dic['lat'][0]
  print plon, plat
  for j in range(1,grid_nfo['area_num_hex'][i]):
    sin_cos[:,j] = mo.rotate_latlon_vec(dat_dic['lon'][j], dat_dic['lat'][j], plon,
        plat)
  return None

def test_compute_flucts(data, grid):
  vars = ['U', 'V']  
  i_cell = 5
  kind   = 'vec'
  dat_dic   = {}
  for var in vars:
    dat_dic[var]    = data[var].values
  dat_dic['lat']    = grid['lat'].values
  dat_dic['lon']    = grid['lon'].values
  grid_nfo= {
            'area_num_hex' : grid['area_num_hex'].values,
            'area_neighbor_idx' : grid['area_neighbor_idx'].values,
            'ntim'              : data.dims['time'],
            'nlev'              : data.dims['lev']
            }

  values =  dop.get_members(grid_nfo, dat_dic, i_cell, vars)

  for var in vars:
    values[var+'_bar']  = values[var][0,:, :]

  values.update(dop.get_members(grid_nfo, dat_dic, i_cell, ['lat', 'lon']))
  
  values =  mo.compute_flucts(values, grid_nfo, grid_nfo['area_num_hex'][i_cell], vars, kind)
  kind = 'scalar'
  values =  mo.compute_flucts(values, grid_nfo, grid_nfo['area_num_hex'][i_cell], vars, kind)

  return values 

def test_compute_dyads(data, grid):
  vars = ['U', 'V', 'RHO']  
  i_cell = 5
  kind   = 'vec'
  dat_dic   = {}
  for var in vars:
    dat_dic[var]    = data[var].values
  dat_dic['lat']    = grid['lat'].values
  dat_dic['lon']    = grid['lon'].values
  grid_nfo= {
            'area_num_hex' : grid['area_num_hex'].values,
            'area_neighbor_idx' : grid['area_neighbor_idx'].values,
            'ntim'              : data.dims['time'],
            'nlev'              : data.dims['lev']
            }
  
  values =  dop.get_members(grid_nfo, dat_dic, i_cell, vars)

  vars = ['U', 'V']  
  grid_nfo['area_num_hex'] = grid['area_num_hex'].values[i_cell]
  values =  mo.compute_dyads(values, grid_nfo, vars)

  return values
  

if __name__== '__main__':
  filep = u'/home1/kd031/projects/icon/experiments/HS_FT_3000_days/'
  filen = [
          filep + u'HS_FT_3000_days_R2B05L26_0042.nc',
          filep + u'HS_FT_3000_days_R2B05L26_0043.nc'
          ]
  gridf = u'iconR2B05-grid_refined_3.nc'
  grid = cio.read_netcdfs([filep+gridf], 'time')
  grid = mo.coarse_area(grid)
  kwargs = {'variables' : ['U','V','RHO']} 
  data= cio.read_netcdfs([filen[0]], 'time', kwargs, func= lambda ds, kwargs:cio.
      extract_variables(ds, **kwargs)) 
 # data  = test_area_avg_bar([filen[0]], grid)
 # data  = test_area_avg_hat([filen[0]], grid)
 # data = test_vec_avg_hat([filen[0]], grid)
 # i = test_rotate_latlon_vec(grid)
 # data = test_compute_flucts(data, grid)
 # data = test_compute_dyads(data, grid)


