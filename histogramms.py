#!/usr/bin/env python
# coding=utf-8

import os
import sys
import glob
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import argparse

import custom_io as cio
import data_op as dop
import math_op as mo
import phys_op as po
import main as ma

def plot_histogram(data, name):
    mu = np.mean(data)
    median = np.median(data)
    sigma = np.std(data)
    d = np.abs(data - median)
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    m = 2
    data = data[s<m]

    fig, ax = plt.subplots()
    density, bins= np.histogram(data, normed = True, density = True, bins = 50 )
    unity_density = density/density.sum()
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + unity_density


    legend_text = '\n'.join((
        r'$\mu={:.2E}$'.format(mu, ),
        r'$\mathrm{median}'+r'={:.2E}$'.format(median, ),
        r'$\sigma= {:.2E}$'.format(sigma, )))
    props = {
        'boxstyle' : 'round',
        'facecolor' : 'wheat',
        'alpha' : 0.5,
    }
    ax.text(
        0.05, 0.95,
        legend_text,
        transform= ax.transAxes,
        fontsize = 14,
        verticalalignment = 'top',
        bbox= props
    )


    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    barpath = path.Path.make_compound_path_from_polys(XY)

    patch = patches.PathPatch(barpath)
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    ax.set_yscale('log')
    ax.set_ylim(bottom.min(), top.max())
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig('bilder/histogramm_{}'.format(name),dpi=73)
   # plt.show()

    plt.close()

def plot_timeline(data, name):
    data = np.average(data, (1,2))
    plt.figure(figsize = (8,6), dpi=80)
    plt.subplot(111)
    X = np.linspace(0, 14, len(data), endpoint = True)
    plt.plot(X, data, linewidth=2.5, linestyle='-')
    plt.xlim(0, 14)
    plt.xticks([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_title(r'time series of global average of {}'.format(name))

    plt.savefig('bilder/timeline_{}'.format(name), dpi=80)
    plt.show()

def plot_slice(data, name):
    None

def collect_files(path):
    files = [
        n for n in
        sorted(glob.glob(path + '/*_refined*.nc')) if
        os.path.isfile(n)
        and 'grid' not in n
        and 'disc' not in n
    ]
    grid = [
        n for n in
        glob.glob(path + '/*grid*refined*.nc') if
        os.path.isfile(n)
    ]

    return files, grid

def run(args):
    if args.path:
        paths, grid_path = collect_files(args.path)
        print(paths)
        print(grid_path)
    elif args.file:
        paths = [args.file]
        print(paths)
    else:
        sys.exit("Error: neither path nor file given")
    # grid = ma.read_grid(grid_path)

    for var in args.vars:
        data = cio.read_netcdfs(paths[0])
        for key in data.iterkeys():
            print key

        data_var = data[var].values
        for i, path in enumerate(paths[1:]):
            data = cio.read_netcdfs(path)
            temp = data[var].values
            del data
            data_var = np.concatenate((data_var, temp), axis=0)
            del temp
        plot_histogram(data_var, var)
        plot_timeline(data_var, var)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Data from Coarse Grained.')
    parser.add_argument(
        '-p',
        dest = 'path',
        type = str,
        help='a string specifying the path to experiment'
    )
    parser.add_argument(
        '-f',
        dest = 'file',
        type = str,
        help='a string specifying the path to experiment'
    )
    parser.add_argument(
        '-nr',
        dest = 'num_rings',
        type = int,
        help = 'an integer specifying the number of rings to be coarse-grained over.'
    )
    parser.add_argument(
        '-v',
        dest = 'vars',
        type = str,
        nargs = '+',
        help = 'names of variables to be analysed and plotted'
    )
    args = parser.parse_args()
    print(
        'analysing and plotting for variables: {}'.format(args.vars)
    )
    print(args)
    run(args)
