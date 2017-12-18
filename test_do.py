import data_op as do
import custom_io as cio
import xarray as xr
import numpy as np

def test_account_for_pentagons(gridp):
  grid = cio.read_netcdfs(gridp, 'time', func= lambda ds:
      cio.read_grid(ds))
  grid = do.account_for_pentagons(grid)
  if isinstance(grid,xr.Dataset):
    return True
  else:
    print type(grid)
    return False

def test_find_neighbors(gridp):
  grid = cio.read_netcdfs(gridp, 'time', func= lambda ds:
      cio.read_grid(ds))
  grid = do.account_for_pentagons(grid)
  grid = do.find_neighbors(grid)
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
  gridf = u'iconR2B05-grid.nc'
  # print test_account_for_pentagons([filep+gridf])

  grid = cio.read_netcdfs([filep+gridf], 'time', func= lambda ds:
      cio.read_grid(ds))
  grid = do.account_for_pentagons(grid)


