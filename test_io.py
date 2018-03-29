import custom_io as cio
import xarray as xr

def test_read_netcdfs(files, kwargs):

  grid = cio.read_netcdfs(files, 'time', kwargs)
  if isinstance(grid,xr.Dataset):
    return True
  else:
    print type(grid)
    return False

def test_read_netcdfs_extr(files, kwargs):
  grid = cio.read_netcdfs(files, 'time', kwargs, func= lambda ds, kwargs:
      cio.extract_variables(ds, **kwargs))
  if isinstance(grid,xr.Dataset):
    return True
  else:
    print type(grid)
    return False

def test_read_netcdfs_grid(files, kwargs):
  grid = cio.read_netcdfs(files, 'time', kwargs, func= lambda ds, kwargs:
      cio.read_grid(ds, **kwargs))
  if isinstance(grid,xr.Dataset):
    return True
  else:
    print type(grid)
    return False

if __name__== '__main__':
  filep = u'/home1/kd031/projects/icon/experiments/HS_FT_3000_days/'
  filen = [
          filep + u'HS_FT_3000_days_R2B05L26_0042.nc',
          filep + u'HS_FT_3000_days_R2B05L26_0043.nc'
          ]
  kwargs = {'variables' : ['U','V']}
  # print test_read_netcdfs(filen, kwargs)
  # print test_read_netcdfs_extr(filen, kwargs)
  kwargs = {'num_rings' : 3, 'path' : filep+'iconR2B05-grid'}
  print test_read_netcdfs_grid([filep+'iconR2B05-grid.nc'], kwargs)

