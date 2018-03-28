import xarray as xr
import data_op as dop
import math_op as mop

# Need this to read a file timestep wise, to minimize the amount of data
# carried around.

def read_netcdfs(files, dim, kwargs=None, func=None):
  """
        reads data from .nc files:
        path: Path to file
        qty : qty to be read
        """
  def process_one_path(path):
    # use a context manager, to ensure file gets closed after read
    with xr.open_dataset(path) as ds:
      #transform_func should do some sort of selection
      # or aggregation
      if func is not None:
        ds = func(ds, kwargs)
      # Load all data from the transformed dataset, to ensure we
      # use it after closing each original file
      ds.load()
      return ds

#  paths   = sorted(glob(files))
  paths = files
  datasets = [process_one_path(p) for p in paths]
  combined = datasets[0]
  if len(datasets) > 1:
    combined = xr.concat(datasets, dim)
  return combined

def extract_variables(ds, variables):

  def take_one_variable(var, ds):
    return ds[var]

  #variables = ['U', 'V']
  data = {}
  for var in variables:
      data[var] = take_one_variable(var, ds)

  ds_o = xr.Dataset(data)

  # for convienience.
  ds_o = ds_o.rename({
          'cell2' : 'ncells',
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
  print 'writing file as {}_refined_{}.nc'.format(path,num_rings)
  write_netcdf(path+'_refined_{}.nc'.format(num_rings), grid)
  return grid

def write_netcdf(path, ds):
    """writes Information to .nc file"""
    ds.to_netcdf(path, mode='w')


