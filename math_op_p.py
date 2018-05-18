#!/usr/bin/env python
# coding=utf-8


import os
import sys
import xarray as xr
import numpy as np
import data_op as dop
import math_op as mo

def gradient_mp(chunk):
    # now what you get is:
    # an array 40962/n_proc * [[4,2,21,26],[1]]
   # data, grid_nfo, gradient_nfo, var):

    re        = 6.37111*10**6

    ncells = len(chunk)
    numvars, ntim, nlev = np.shape(chunk[0][0])[1:]


    data_out  = np.empty([
        ncells,
        2, numvars,
        ntim, nlev
        ])
    data_out.fill(0)

    for i in range(ncells):
        # chekc for correct key
        # define distance
        # assume circle pi r^2 => d= 2*sqrt(A/pi)


        area    = chunk[i][2]
        d       = 2 * mo.radius(area) * re
        data = chunk[i][1]
        neighs = chunk[i][0]

        for var in range(numvars):
            data_out[i, 0, var, :, :]   = mo.central_diff(
                neighs[0, var], data[var], neighs[1, var],
                d
                )
            data_out[i, 1, var, :, :]   = mo.central_diff(
                neighs[2, var], data[var], neighs[3, var],
                d
                )

    # find/interpolate value at distance d in x/y direction
    #     -> use geodesics / x/y along longditudes/latitudes
    # turn values at distance (with rot_vec)
    # compute d/dz use fassinterpol x_i-1+2x_i+x_i+1/dx including grid value?
    return data_out

def circ_dist_avg_vec(chunks):
    # wrapper for circ_dist_avg in case of vector
    out_values = []
    for i, chunk in enumerate(chunks):
        member_rad = chunk[-1][0]
        coords = chunk[-1][1]
        latlons = chunk[-1][2]
        values = chunk[:-1]

        out_values.append(circ_dist_avg(member_rad, coords, latlons, values))

    return np.array(out_values)

def circ_dist_avg(member_rad_in, coords_in, latlons_in, values_in):
    '''does its thing only in case of vector. not suited for scalars, for now'''

    out_values = []

    for j in range(4):

        coords = coords_in[j]
        member_rad = member_rad_in[j]
        members = []
        for x, member in enumerate(values_in):
            members.append(member[j])
            #turn vectors
        lats = latlons_in[j][0]
        lons = latlons_in[j][1]
        cell_area = latlons_in[j][2]
        plon, plat = mo.get_polar(coords[0], coords[1])
        rot_vec  =  mo.rotate_members_to_local(
            lons, lats, plon, plat,
            members[0][:,:,:], members[1][:,:,:])
        members[0]   = rot_vec[0]
        members[1]   = rot_vec[1]
            # compute average
        helper         = dist_avg(members, cell_area, member_rad)
            # turn vectors back
        rot_vec = mo.rotate_single_to_global(
            coords[0], coords[1],
            helper[0][:,:], helper[1][:,:])
        helper[0]   = rot_vec[0]
        helper[1]   = rot_vec[1]

        out_values.append(helper)

    return out_values


def dist_avg(members, radii, cell_area):
    len_i  = len(radii)
    # define weights.
    weight = np.empty([len(radii)])
    weight.fill(0)
    factor = 0
    for i in range(len_i):
        weight[i] = cell_area[i] * radii[i]
        factor    = factor + weight[i]
    # compute distance averages.
    sum    = []
    for j,member in enumerate(members):
        sum.append(np.empty(member[0].shape))
        sum[j].fill(0)
        for i in range(len_i):
            sum[j]    = np.add(sum[j], np.multiply(member[i],weight[i]))
        sum[j]    = np.divide(sum[j],factor)

    return sum
