import numpy as np
import math_op as mo
import custom_io as cio
import data_op as dop
import phys_op as po
import xarray as xr

def test_potT_to_T_pressure(data_s):

    data_s['T'] = po.potT_to_T_pressure(data_s['THETA_V'], data_s['P'], 'dry')
    return data_s

def test_potT_to_T_exner(data_s):
    data_s['T'] = po.potT_to_T_exner(data_s['THETA_V'], data_s['EXNER'])
    return data_s

if __name__== '__main__':
  filep = u'/home1/kd031/projects/icon/experiments/HS_FT_6000_days/'
  filen = [
          filep + u'HS_FT_6000_days_R2B05L26_0042.nc',
          filep + u'HS_FT_6000_days_R2B05L26_0043.nc'
          ]
  gridf = u'iconR2B05-grid_refined_3.nc'
  grid = cio.read_netcdfs([filep+gridf], 'time')
  grid = mo.coarse_area(grid)
  kwargs = {'variables' : ['U','V','RHO', 'THETA_V', 'EXNER']}
  data= cio.read_netcdfs([filen[0]], 'time', kwargs, func= lambda ds, kwargs:cio.
      extract_variables(ds, **kwargs))
  grid_nfo={
      'area_num_hex'        : grid['area_num_hex'].values,
      'area_neighbor_idx'   : grid['area_neighbor_idx'].values,
      'coarse_area'         : grid['coarse_area'].values,
      'cell_area'           : grid['cell_area'].values,
      'ntim'                : data.dims['time'],
      'nlev'                : data.dims['lev'],
      'ncells'              : data.dims['ncells'],
      'lat'                 : data['clat'].values,
      'lon'                 : data['clon'].values,
      'i_cell'              : 0
      }
  data_s = {}
  for var in kwargs['variables']:
    data_s[var] = data[var].values
  data_s = test_potT_to_T_exner(data_s)
  #data_s = test_potT_to_T_pressure(data_s)
