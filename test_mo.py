import math_op as mo
import custom_io as cio
import xarray as xr 


def test_coarse_area(files):
  grid = cio.read_netcdfs(files, 'time')
  grid = mo.coarse_area(grid)
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
  gridf = u'icon_R2B05-grid_refined.nc'

  grid = cio.read_netcdfs([filep+gridf], 'time')
  grid = mo.coarse_area(grid)

