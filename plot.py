#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from modules.new_grad_mod import convert_rad_m

def plot_coords(coord_list, center=None, names=None):
    fig = plt.figure(figsize=(8,8))
    if not center:
        print('no center')
        center = [0.0, 0.0]
    center = np.rad2deg(center)
    coord_list = np.rad2deg(coord_list)
    m = Basemap(projection='moll', resolution=None, width=12E6, height=12E6, lat_0=center[1], lon_0=center[0],)
    m.etopo(scale=0.5, alpha=0.5)
    if not names:
        names = ['N', 'S', 'E', 'W']
    for i, coord in enumerate(coord_list):
        x, y = m(coord[0], coord[1])
        plt.plot(x, y, 'ok', markersize = 5)
        plt.text(x, y, names[i], fontsize=12)
    plt.show()

def plot_bounds(bounds, center, radius):
    coord_list = bounds2coords(bounds, center)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    center = np.rad2deg(center)
    coord_list = np.rad2deg(coord_list)
    m = Basemap(projection='moll', resolution=None, width=12E6, height=12E6, lat_0=center[1], lon_0=center[0],)
    m.etopo(scale=0.5, alpha=0.5)
    for i, coord in enumerate(coord_list):
        x, y = m(coord[0], coord[1])
        plt.plot(x, y, 'ok', markersize = 5)
        #plt.text(x, y, names[i], fontsize=12)
    circle1 = plt.Circle(xy=m(*center), radius=convert_rad_m(radius), edgecolor="blue", zorder=10)
    ax.add_artist(circle1)
    plt.show()

def bounds2coords(bounds, center):
    lons = bounds[0]
    lats = bounds[1]
    coords = []
    for i, lon_pair in enumerate(lons):
        if lon_pair[0] > -4 :
            coords.append([lon_pair[0], center[1]])
            coords.append([lon_pair[1], center[1]])
    for lat in lats:
        coords.append([center[0], lat])
    return coords


if __name__ == '__main__':
    center = [np.pi, np.pi/4]
    names = ['myass']
    coord_list = [[np.pi, np.pi/4]]
    plot_coords(coord_list, center=center, names=names)
