#!/usr/bin/env python
# coding=utf-8
import numpy as np
import modules.cio as cio
import modules.math_mod as math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.basemap import Basemap, cm

def plot_grid_pts(IO):
    coarse_grain = IO.load_from('grid', 'grad_idx')


    lats = IO.load_from('grid', 'vlat')
    lons = IO.load_from('grid', 'vlon')
    flags = []
    for i in range(len(lats)):
        cg = []
        broke = False
        for j in range(4):
            cg.append(np.extract(np.greater(coarse_grain[i,j,:], -1), coarse_grain[i,j,:]))
        for j in range(4):
            for k, cgx in enumerate(cg[j]):
                for l in range(j+1,4):
                    if np.any(np.equal(cg[l], cgx)):
                        flags.append(i)
                        broke = True
                        break
                if broke:
                    break
            if broke:
                print('idx: {}'.format(i))
                print('gradient idx: \n {}'.format(coarse_grain[i, :, :]))
                break

    idx = flags
  # for i, idxi in enumerate(idx):
  #     print idxi
  #     print('center lon lat: {}'.format([lons[idxi], lats[idxi]]))
    #   for i in range(4):
    #       cg = np.where(coarse_grain[0,i,:]>-1)[0]
    #       print('lonsnlats \n {}'.format([[lons[cgi], lats[cgi]] for cgi in cg]))
    print('not affected: {}'.format(np.where(idx != np.arange(len(lats)))[0]))

    xlat = []
    ylon = []
    cxlat = lats[idx[0]]
    cylat = lons[idx[0]]
    id_c = idx[0]
    xlat.append(lats[id_c])
    ylon.append(lons[id_c])
    for cg in coarse_grain[id_c, :, :].flatten():
        if cg >= 0 :
            xlat.append(lats[cg])
            ylon.append(lons[cg])
    xlns = []
    ylts = []
    for i in range(4):
        xlns.append([])
        ylts.append([])
        for cg in coarse_grain[id_c, i, :]:
            if cg >=0 :
                xlns[i].append(lons[cg])
                ylts[i].append(lats[cg])

    dbound = 0.157
    lat_max = np.max(xlat) + dbound
    lat_min = np.min(xlat) - dbound
    lon_max = np.max(ylon) + dbound
    lon_min = np.min(ylon) - dbound
    lon_0 = cxlat
    lat_0 = cylat
    m = Basemap(projection='stere', lon_0=lon_0, lat_0=lat_0, llcrnrlat=lat_min, llcrnrlon=lon_min, urcrnrlat=lat_max, urcrnrlon=lon_max)
    x, y = m(ylon, xlat)
    xt = []
    yt = []
    for i in range(4):
        xd, yd = m(xlns[i], ylts[i])
        xt.append(xd)
        yt.append(yd)

    m.drawmapboundary(fill_color='#00ffff')
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')
    m.scatter(x[0],y[0],marker='o',color='r')
    colors = ['k','g','b','y']
    for i in range(4):
        m.scatter(xt[i],yt[i],2,marker='o',color=colors[i])
    plt.show()
    plt.savefig('grid_vis.png')

def excentricity_areas(IO):
    r_e = 6.37111*10**3
    coarse_grain = IO.load_from('grid', 'area_member_idx')
    coarse_area = IO.load_from('grid', 'coarse_area')
    lats = IO.load_from('grid', 'vlat')
    lons = IO.load_from('grid', 'vlon')
    print('lats: {}'.format(np.shape(lats)))
    mean_quad_dev = np.zeros(np.shape(coarse_area))
    for idx_set, area_i in zip(coarse_grain, coarse_area):
        ra = math.radius(area_i) * r_e
        rs = []
        idxj = [x for x in idx_set if x > -1]
        clat = lats[idxj[0]]
        clon = lons[idxj[0]]
        for idx in idxj[1 :]:
            rs.append(math.arc_len([clon, clat], [lons[idx], lats[idx]]) * r_e)
        abw = 0
        for r in rs:
            abw = abw + (r - ra)**2
        mean_quad_dev[idx_set[0]] = np.sqrt(abw / len(rs))

    mean_mean = np.mean(mean_quad_dev)
    print('mean {}'.format(np.mean(mean_quad_dev)) )
    print(mean_quad_dev)

    print('std: {}'.format(np.std(mean_quad_dev)) )
    print('var {}'.format(np.var(mean_quad_dev)) )
    # interpolation
    # -----
    ngridx = 1152
    ngridy = 576
    npts = ngridx * ngridy
    xi = np.linspace(-np.pi, np.pi, ngridx)
    yi = np.linspace(-np.pi/2, np.pi/2, ngridy)
    triang = tri.Triangulation(lons, lats)
    interpolator = tri.LinearTriInterpolator(triang, mean_quad_dev)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    # ----

    lat_max = np.max(lats)
    lat_min = np.min(lats)
    lon_max = np.max(lons)
    lon_min = np.min(lons)
    lon_0 = 0
    lat_0 = 0

    m = Basemap(
        projection='stere',
        lon_0=lon_0, lat_0=lat_0,
        llcrnrlat=lat_min, llcrnrlon=lon_min,
        urcrnrlat=lat_max, urcrnrlon=lon_max
    )

    x, y = m(Xi, Yi) # compute map proj coordinates.
    # draw filled contours.
    clevs = np.linspace(mean_mean-20, mean_mean+20, 40)
    cs = m.contourf(x ,y ,zi ,clevs ,cmap=plt.get_cmap('seismic'))
    # add colorbar.
    cbar = m.colorbar(cs,location='bottom',pad="5%")
    plt.show()
    plt.savefig('grid_quad_dev.png')


if __name__ == '__main__':
    path = '/home1/kd031/projects/icon/experiments/BCWgeofric'
    grid = r'iconR\dB\d{2}-grid_refined_3.nc'
    data = r'BCWgeofric_slice.nc'
    IO = cio.IOcontroller(path, grid, data)
  #  excentricity_areas(IO)
    plot_grid_pts(IO)
    print('done.')



