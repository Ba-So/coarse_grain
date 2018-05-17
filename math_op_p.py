#!/usr/bin/env python
# coding=utf-8


import os
import sys
import xarray as xr
import numpy as np
import data_op as dop

def gradient(chunk):
   # data, grid_nfo, gradient_nfo, var):

    re        = 6.37111*10**6

    data['gradient']  = np.empty([
        2, 2,
        grid_nfo['ntim'], grid_nfo['nlev'], grid_nfo['ncells']
        ])
    data['gradient'].fill(0)

    info_bnd = 5000
    for i in range(grid_nfo['ncells']):
        # chekc for correct key
        # define distance
        # assume circle pi r^2 => d= 2*sqrt(A/pi)
        if i == info_bnd:
            print('progress: {} cells of {}').format(info_bnd, grid_nfo['ncells'])
            info_bnd = info_bnd + 5000

        # check for correct keys
        neighs  = circ_dist_avg(data, grid_nfo, gradient_nfo, i, var)
        area    = grid_nfo['cell_area'][i]
        d       = 2 * radius(area) * re
        data['gradient'][0,0,:,:,i]   = central_diff(
            neighs['U_hat'][0], data['U_hat'][:, :, i], neighs['U_hat'][1],
            d
            )
        data['gradient'][0,1,:,:,i]  = central_diff(
            neighs['V_hat'][0], data['V_hat'][:, :, i], neighs['V_hat'][1],
            d
            )
        data['gradient'][1,0,:,:,i]   = central_diff(
            neighs['U_hat'][2], data['U_hat'][:, :, i], neighs['U_hat'][3],
            d
            )
        data['gradient'][1,1,:,:,i]   = central_diff(
            neighs['V_hat'][2], data['V_hat'][:, :, i], neighs['V_hat'][3],
            d
            )

    # find/interpolate value at distance d in x/y direction
    #     -> use geodesics / x/y along longditudes/latitudes
    # turn values at distance (with rot_vec)
    # compute d/dz use fassinterpol x_i-1+2x_i+x_i+1/dx including grid value?
    return data

def vector_circ_dist_avg():
    # wrapper for circ_dist_avg in case of vector

    return

def circ_dist_avg(data, grid_nfo, gradient_nfo, i_cell, var):
    values = {
        name: np.empty(
            [4,grid_nfo['ntim'],grid_nfo['nlev']])
            for name in var['vars']}
    for name in var['vars']:
        values[name].fill(0)
    #how large is check radius?
    for j in range(4):
        coords = gradient_nfo['coords'][i_cell, j, :]
        member_idx = gradient_nfo['member_idx'][i_cell][j]
        member_rad = gradient_nfo['member_rad'][i_cell][j]
    # compute distance weighted average of area weighted members:
        members    = dop.get_members_idx(data, member_idx, var['vars'])
        # if you suffer from boredom: This setup gives a lot of flexebility.
        #     may be transferred to other parts of this program.
        if 'vector' in var.iterkeys():
            #turn vectors
            lats = np.array([grid_nfo['lat'][i] for i in member_idx])
            lons = np.array([grid_nfo['lon'][i] for i in member_idx])
            vec   = var['vector']
            rot_vec = np.empty([
                len(vec),
                len(member_idx),
                grid_nfo['ntim'],
                grid_nfo['nlev']])
            rot_vec.fill(0)
            plon, plat = get_polar(coords[0], coords[1])
            rot_vec[:,:,:,:]  =  rotate_members_to_local(
                lons, lats, plon, plat,
                members[vec[0]][:,:,:], members[vec[1]][:,:,:])
            members[vec[0]]   = rot_vec[0,:,:,:]
            members[vec[1]]   = rot_vec[1,:,:,:]
        helper         = dist_avg(members, member_idx, member_rad, grid_nfo, var['vars'])
        if 'vector' in var.iterkeys():
            rot_vec =rotate_single_to_global(
                coords[0], coords[1],
                helper[vec[0]][:,:], helper[vec[1]][:,:])
            helper[vec[0]]   = rot_vec[0]
            helper[vec[1]]   = rot_vec[1]

        for name in var['vars']:
            values[name][j]   = helper[name]

    return values

