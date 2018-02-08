import data_op as do
import custom_io as cio
import xarray as xr
import numpy as np
from collections import Counter

def test_account_for_pentagons(gridp, kwargs):
  grid = cio.read_netcdfs([filep+gridf], 'time', kwargs, func= lambda ds, kwargs:
      cio.read_grid(ds, **kwargs))
  grid = do.account_for_pentagons(grid)
  if isinstance(grid,xr.Dataset):
    return True
  else:
    print type(grid)
    return False

def test_define_hex_area(grid):

  # test if -1 is occuring together with 36
  a = [(grid['area_neighbor_idx'].values[36,i] == -1 and
    grid['area_num_hex'][i].values == 36) for i in range(40962)]

  b = ([(grid['area_neighbor_idx'].values[36,i] != -1 and
    grid['area_num_hex'][i].values == 37) for i in range(40962)])

  Test = [(a[i] or b[i]) for i in range(40962)]
  
  for i in range(40962):
    if Test[i] == False:
      print i
      print grid['area_num_hex'][i].values
      print grid['area_neighbor_idx'].values[:,i]

  if isinstance(grid,xr.Dataset):
    Test.append( True)
  else:
    print type(grid)
    Test.append( False)
  if all( i == True for i in Test):
    return True
  else:
    return False

def test_hex_area(grid):
  grid_nfo= {
            'area_num_hex' : grid['area_num_hex'].values,
            'area_neighbor_idx' : grid['area_neighbor_idx'].values,
            'ncells'              : grid.dims['ncells'],
            }

  for i in range(grid_nfo['ncells']):
    check = [count < 2 for item, count in Counter(grid['area_neighbor_idx'].values[:,i]).iteritems()]
    if not all(ch == True for ch in check):
      return False

  return True


def test_get_members(grid, files):
  kwargs = {'variables' : ['U','RHO']} 
  data= cio.read_netcdfs([files], 'time', kwargs, func= lambda ds, kwargs:cio.
      extract_variables(ds, **kwargs)) 
  grid_nfo= {
            'area_num_hex' : grid['area_num_hex'].values,
            'area_neighbor_idx' : grid['area_neighbor_idx'].values,
            'ntim'              : data.dims['time'],
            'nlev'              : data.dims['lev']
            }
  dat_dict  = {}
  for var in kwargs['variables']:
    dat_dict[var] = data[var].values

  for i in range(40962):
    out = do.get_members(grid_nfo, dat_dict, i, **kwargs)
    if not(len(out['U']) == grid['area_num_hex'][i].values):
      return False
  return True

def test_get_members_coords(grid, files):
  kwargs = {'variables' : ['lat','lon']} 
  data= cio.read_netcdfs([files], 'time', kwargs, func= lambda ds, kwargs:cio.
      extract_variables(ds, **kwargs)) 
  grid_nfo= {
            'area_num_hex' : grid['area_num_hex'].values,
            'area_neighbor_idx' : grid['area_neighbor_idx'].values,
            'ntim'              : data.dims['time'],
            'nlev'              : data.dims['lev']
            }
  dat_dict  = {}
  for var in kwargs['variables']:
    dat_dict[var] = grid[var].values

  for i in range(40962):
    out = do.get_members(grid_nfo, dat_dict, i, **kwargs)
    if not all(i == True for i in [count < 2 for item, count in
      Counter(out).iteritems()]):
      print i
      return False
  return True


if __name__== '__main__':
  filep = u'/home1/kd031/projects/icon/experiments/HS_FT_3000_days/'
  filen = [
          filep + u'HS_FT_3000_days_R2B05L26_0042.nc',
          filep + u'HS_FT_3000_days_R2B05L26_0043.nc'
          ]
  # re-prepare a grid for testing
  gridf = u'iconR2B05-grid.nc'
  kwargs = {'num_rings' : 3, 'path' : filep+'iconR2B05-grid'}
  #grid = cio.read_netcdfs([filep+gridf], 'time', kwargs, func= lambda ds, kwargs:
  #    cio.read_grid(ds, **kwargs))
  gridfr = u'iconR2B05-grid_refined_3.nc'
  grid = cio.read_netcdfs([filep+gridfr], 'time')

 #print test_define_hex_area(grid)
 #print test_get_members(grid, filen[0])
 #print test_get_members(grid, filen[0])
  print test_hex_area(grid)
 #a = debug_hex_area(grid)
  kwargs = {'variables' : ['U','RHO']} 
  data= cio.read_netcdfs([filen[0]], 'time', kwargs, func= lambda ds, kwargs:cio.
      extract_variables(ds, **kwargs)) 

  #grid = cio.read_netcdfs([filep+gridf], 'time', kwargs, func= lambda ds, kwargs:
  #    cio.read_grid(ds, **kwargs))
  #grid = do.account_for_pentagons(grid)



