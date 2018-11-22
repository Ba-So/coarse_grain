#!/usr/bin/env python
# coding=utf-8
import numpy as np
from debugdecorators import TimeThis, PrintArgs
from paralleldecorators import Mp, ParallelNpArray
from functiondecorators import requires
#pass data as full, but give c_area in slices.
#race conditions?
mp = Mp(8, False) # standart be 8 cores, and deactivated

#@TimeThis
@requires()
@ParallelNpArray(mp)
def func(ina, inb, ret):
    """function sceleton"""
    foo = ina * inb
    ret = foo
#--------------------
#@TimeThis
@requires({
    'full' : ['data'],
    'slice' : ['c_area', 'ret']
})
@ParallelNpArray(mp)
def avg_bar(data, c_area, ret):
    """computes the bar average of data
        data - needs to be full stack
        c_area - can be part"""
    vals = np.sum(data, 0)
    ret[:] = np.divide(vals, coarse_area)
#--------------------
#@TimeThis
@requires({
    'full' : ['data', 'rho'],
    'slice' : ['rho_bar', 'c_area', 'ret']
})
@ParallelNpArray(mp)
def avg_hat(data, rho, rho_bar, c_area, ret):
    """computes the hat average of data,
    requires the average of rho"""
    vals = np.sum(np.multiply(data, rho, 0))
    ret[:] = np.divide(vals, (c_area, rho_bar))
#--------------------
#@TimeThis
@requires({
    'full' : [],
    'slice' : ['ret', 'data', 'data_avg']
})
@ParallelNpArray(mp)
def fluctsof(data, data_avg, ret):
    """computes deviations from local mean"""
    ret[:] = np.subtract(data_avg, data)
#--------------------
#@TimeThis
@requires({
    'full' : ['data'],
    'slice' : ['c_area', 'ret']
})
@ParallelNpArray(mp)
def uv_2D_gradient():
    """computes the gradient of velocity components in u,v using
    the information in gradient_nfo
    NEEDS:
        full: vectorfield
        partial: coarse_area, member_idx, member_rad

    """

# TODO: reformulate to do only search only through one variable
#       -> less confusing code!
#@TimeThis
def circ_dist_avg(i_cell, var):

    values = np.zeros(ntim, nlev, 4)




