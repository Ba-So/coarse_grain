import numpy as np
import math_op as mo
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

def test_area_avg_bar(data, grid_nfo):
  kind= 'bar'
  var = {
        'vars' : ['U']
      }
  data = mo.area_avg(kind, grid_nfo, data, var)
  return data

def test_area_avg_hat(data, grid_nfo):
  kind= 'hat'
  var = {
        'vars' : ['U']
      }
  data = mo.area_avg(kind, grid_nfo, data, var)
  return data

def test_vec_avg_hat(data, grid_nfo):
  kind= 'hat'
  var = {
        'vars' : ['U','V'],
        'vector':['U','V']
      }
  data= mo.area_avg(kind, grid_nfo, data, var, True)

  data['bar_sq']    = np.sqrt(data['U_hat']**2 + data['V_hat']**2)
  data['sq']        = np.sqrt(data['U']**2 + data['V']**2)
  data['mean_diff'] = np.mean(data['U_hat']) - np.mean(data['U'])

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

def test_compute_flucts(data, grid_nfo):
  vars = ['U', 'V']
  i_cell = 5
  grid_nfo['i_cell'] = i_cell
  kind   = 'vec'

  values =  dop.get_members(grid_nfo, data , i_cell, vars)

  for var in vars:
    values[var+'_bar']  = values[var][0,:, :]

  values.update(dop.get_members(grid_nfo, grid_nfo, i_cell, ['lat', 'lon']))

  values =  mo.compute_flucts(values, grid_nfo, grid_nfo['area_num_hex'][i_cell], vars, kind)
  kind = 'scalar'
  values =  mo.compute_flucts(values, grid_nfo, grid_nfo['area_num_hex'][i_cell], vars, kind)

  return values

def test_compute_dyads(data, grid):
  vars = ['U', 'V', 'RHO']
  i_cell = 5
  kind   = 'vec'
  dat_dic   = {}
  grid_nfo['i_cell'] = i_cell
  values =  dop.get_members(grid_nfo, data, i_cell, vars)
  values.update(dop.get_members(grid_nfo, grid_nfo, i_cell, ['cell_area']))
  vars = ['U', 'V']
  values =  mo.compute_dyads(values, grid_nfo, vars)

  return values

def test_radius(radius):
  area  = radius**2*np.pi
  t_rad = mo.radius(area)
  return t_rad == radius/6371

def test_max_min_bounds(data, grid_nfo):
  i_cell    = 3
  for i_cell in range(3, 100):
    lonlat    = [grid_nfo['lon'][i_cell],grid_nfo['lat'][i_cell]]
    area      = grid_nfo['coarse_area'][i_cell]
    bounds    = mo.max_min_bounds(lonlat, area)
    print bounds
  print('{}').format(np.shape(bounds))
  return None

def test_gradient_coordinates(grid_nfo):
  for i_cell in range(3,100):
    area = grid_nfo['cell_area'][i_cell]
    lonlat = [grid_nfo['lon'][i_cell], grid_nfo['lat'][i_cell]]
    coords = mo.gradient_coordinates(lonlat, area)
    print coords
  return None

def test_arc_len(grid_nfo):
  i_one = 3
  i_two = 4
  p_one = [grid_nfo['lon'][i_one], grid_nfo['lat'][i_one]]
  p_two = [grid_nfo['lon'][i_two], grid_nfo['lat'][i_two]]
  distance  = mo.arc_len(p_one, p_two)
  print distance
  return None

def test_circ_dist_avg(data, grid_nfo):
  i_cell    = 6
  lonlat    = [grid_nfo['lon'][i_cell],grid_nfo['lat'][i_cell]]
  print lonlat
  area      = grid_nfo['coarse_area'][i_cell]
  coords    = mo.gradient_coordinates(lonlat, area)
  print coords[1,:]
  print coords
  # test scalar only
#  area      = grid_nfo['cell_area'][i_cell]
#  var       = {
#        'vars': ['RHO'],
#      }
#  values    = mo.circ_dist_avg(data, grid_nfo, coords, area, var)
#  print values
  # test with vector
  var       = {
        'vars': ['U', 'V', 'RHO'],
        'vector' : ['U', 'V']
      }
  values    = mo.circ_dist_avg(data, grid_nfo, coords, area, var)

  return values


def test_turn_spherical(grid_nfo):
  lonlat = [grid_nfo['lon'][300], grid_nfo['lat'][300]]
  print lonlat
  new_co = np.array(mo.rotate_latlon(lonlat, grid_nfo))
  print new_co[:,3]

  test = True
  return new_co

def test_rotate_vec_to_global(grid_nfo, data_s):
    i_cell = 3
    plon, plat = mo.get_polar(
        grid_nfo['lon'][i_cell],
        grid_nfo['lat'][i_cell])
    U, V = mo.rotate_vec_to_local(
        plon, plat,
        grid_nfo['lon'][i_cell],
        grid_nfo['lat'][i_cell],
        data_s['U'][i_cell, :, :],
        data_s['V'][i_cell, :, :])
    print('U: {}, V: {}').format(U[5, 6], V[5, 6])
    U, V = mo.rotate_vec_to_global(
        plon, plat,
        grid_nfo['lon'][i_cell],
        grid_nfo['lat'][i_cell],
        U, V)
    print('U: {}, V: {}').format(U[5, 6], V[5, 6])
    diff_U = U - data_s['U'][i_cell, :, :]
    diff_V = V - data_s['V'][i_cell, :, :]
    print('{} , {}').format(diff_U, diff_V)
    U, V = mo.rotate_ca_members_to_local(
        grid_nfo['lon'][3:10],
        grid_nfo['lat'][3:10],
        data_s['U'][3:10, :, :],
        data_s['V'][3:10, :, :])
    U, V = mo.rotate_ca_members_to_global(
        grid_nfo['lon'][3:10],
        grid_nfo['lat'][3:10],
        U[:, :, :],
        V[:, :, :])
    diff_U = U - data_s['U'][3:10, :, :]
    diff_V = V - data_s['V'][3:10, :, :]
    print('{} , {}').format(diff_U, diff_V)

    return None

def test_get_members_idx(data_s):
    idxcs = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 53, 98,
             101, 102, 103, 104, 105, 106, 107, 108, 112, 113, 114, 115, 117,
             2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153,
             2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164,
             2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175,
             2176, 2177, 2179, 2184, 2224, 2225, 2226, 2227, 2228, 2229, 2230,
             2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241,
             2242, 2243, 2244, 2249, 4222, 4223, 4224, 4225, 4226, 4227, 4228,
             4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239,
             4240, 4241, 4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249, 4250,
             4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4263, 4265, 4303,
             4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314,
             4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4328, 6301,
             6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312,
             6313, 6314, 6315, 6316, 6317, 6318, 6319, 6320, 6321, 6322, 6323,
             6324, 6325, 6326, 6327, 6328, 6329, 6330, 6331, 6332, 6333, 6334,
             6335, 6336, 6337, 6342, 6344, 6382, 6383, 6384, 6385, 6386, 6387,
             6388, 6389, 6390, 6391, 6392, 6393, 6394, 6395, 6396, 6397, 6398,
             6399, 6400, 6401, 6402, 6407, 8380, 8381, 8382, 8383, 8384, 8385,
             8386, 8387, 8388, 8389, 8390, 8391, 8392, 8393, 8394, 8395, 8396,
             8397, 8398, 8399, 8400, 8401, 8402, 8403, 8404, 8405, 8406, 8407,
             8408, 8409, 8414, 8416, 8454, 8455, 8456, 8457, 8458, 8459, 8460,
             8461, 8462, 8463, 8464, 8465, 8466, 8467, 8468, 8469, 8470, 8471,
             8472 ]
    vars = ['U', 'V', 'RHO']
    data_s = dop.get_members_idx(data_s, idxcs, vars)

    return data_s

def test_gradient(data_s, grid_nfo):
    var = {
        'vars': ['U', 'V'],
        'vector': ['U', 'V']
       }
    return mo.gradient(data_s, grid_nfo, var)

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
 # data_s   = test_area_avg_bar(data_s, grid_nfo)
 # data_s  = test_area_avg_hat(data_s, grid_nfo)
 # data_s = test_vec_avg_hat(data_s, grid_nfo)
 # i = test_rotate_latlon_vec(grid)
 # data = test_compute_flucts(data, grid_nfo)
 # data_s = test_compute_dyads(data_s, grid_nfo)
 # data_s = test_radius(3.0)
 # data_s = test_get_members_idx(data_s)
 # print data_s
 # data_s = test_max_min_bounds(data, grid_nfo)
 # data_s = test_arc_len(grid_nfo)
  data_s = test_circ_dist_avg(data_s, grid_nfo)
 # data_s = test_turn_spherical(grid_nfo)
 # data_s = test_gradient_coordinates(grid_nfo)
 # data_s = test_rotate_vec_to_global(grid_nfo, data_s)
 # data_s = test_gradient(data_s, grid_nfo)
