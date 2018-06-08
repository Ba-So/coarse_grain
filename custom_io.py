import xarray as xr
import data_op as dop
import math_op as mop

# Need this to read a file timestep wise, to minimize the amount of data
# carried around.

def read_netcdfs(path, dim, kwargs=None, func=None):
    """
        reads data from .nc files:
        path: Path to file
        qty : qty to be read
        """
    ds = xr.open_dataset(path)

    if func is not None:
        ds = func(ds, kwargs)

    ds.load()

    return ds

def extract_variables(ds, variables):

  #variables = ['U', 'V']
  ds_o = ds[variables]
  # for convienience.
  check = ['vlon', 'vlat']
  ds_keys = [i for i in ds_o.variables.iterkeys()]
  if 'cell2' in ds_o.dims.iterkeys():
      ds_o = ds_o.rename({'cell2' : 'ncells'})
  if all(x in ds_keys for x in check):
      ds_o = ds_o.rename({
        'vlon' : 'clon',
        'vlat' : 'clat'
        })

  return ds_o

def read_grid(ds, path, num_rings):
  ''' subcontractor for reading ICON grid files '''
  # What:
  #     reassign
  #     rename
  #     sort pentagon stuff

  grid  = xr.Dataset({'cell_idx'            : ds['vertex_index'],
                      'cell_neighbor_idx'   : ds['vertices_of_vertex'],
                      'cell_area'           : ds['dual_area_p']
                      })

  grid  = grid.rename({'vertex' : 'ncells',
                       'vlon' : 'lon', 'vlat' : 'lat',
                       })

  grid['cell_idx'].attrs['long_name']           = 'cell index'
  grid['cell_neighbor_idx'].attrs['long_name']  = 'cell neighbor index'
  grid['cell_area'].attrs['long_name']          = 'cell area'

  print '--------------'
  print 'accounting for pentagons'
  grid = dop.account_for_pentagons(grid)
  print '--------------'
  print 'defining hex area members'
  grid = dop.define_hex_area(grid, num_rings)
  print '--------------'
  print 'computing total hex area'
  grid = mop.coarse_area(grid)
  print '--------------'
  print 'computing gradient_nfo'
  grid = dop.get_gradient_nfo(grid)
  print '--------------'
  print 'writing file as {}_refined_{}.nc'.format(path[:-3],num_rings)
  write_netcdf(path[:-3] + '_refined_{}.nc'.format(num_rings), grid)

  return grid

def write_netcdf(path, ds):
    """writes Information to .nc file"""
    print ('writing to {}').format(path)
    ds.to_netcdf(path, mode='w')


