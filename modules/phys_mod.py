#!/usr/bin/env python
# coding=utf-8
import numpy as np
import itertools

from decorators.debugdecorators import TimeThis, PrintArgs
from decorators.paralleldecorators import gmp, ParallelNpArray, shared_np_array
from decorators.functiondecorators import requires
import modules.math_mod as math
'''contains all functions related to physical operations/terms'''

@TimeThis
@requires({
    'full': ['x_vals', 'y_vals', 'rho'],
    'slice': ['x_avg_list', 'y_avg_list', 'c_mem_idx', 'coarse_area', 'ret']
})
@ParallelNpArray(gmp)
def compute_dyad(x_vals, y_vals, rho, x_avg_list, y_avg_list, c_mem_idx, coarse_area, ret):
    '''computes the dyadic product of avg(rho X'X')'''
    out = []
    for idx_set, c_area, x_avg, y_avg, reti in itertools.izip(c_mem_idx, coarse_area, x_avg_list, y_avg_list, ret):
        x_flucts, y_flucts = math.vec_flucts(
            [x_vals[j] for j in idx_set],
            [y_vals[j] for j in idx_set],
            x_avg, y_avg
        )
        constituents = np.array([[x[0] for x in x_flucts], [y[0] for y in y_flucts]])
        rho_set = np.array([rho[j] for j in idx_set])
        cell_area = np.array([x[1] for x in x_flucts])
        # set up uv matrix
        product = np.einsum('ilmk,jlmk->ijlmk', constituents, constituents)
        # average over coarse_area
        dyad = np.einsum(
            'ijlmk,l,lmk->ijmk',
            product,
            cell_area,
            rho_set
        )
        #normalize:
        reti[:,] = np.divide(dyad, c_area)

@TimeThis
@requires({
    'full' : [],
    'slice': ['rhoxy', 'gradxy', 'tfric']
})
@ParallelNpArray(gmp)
def turb_fric(rhoxy, gradxy, tfric):
    tfric[:, ] = np.einsum(
        'kijlm,kijlm->klm',
        rhoxy,
        gradxy
    )

@TimeThis
def turb_fric_erich(tfric):
    '''computes not the entropy but enthalpy production through enthalpy'''
    rd = 287.04
    c_p = 1004.64
    c_v = c_p - rd
    s_to_d = 60*60*24 # to convert from 1/s to 1/d
    return tfric * s_to_d / c_p


@TimeThis
@requires({
    'full' : [],
    'slice': ['gradxy', 'rhobar', 'tfric', 'imagk']
})
@ParallelNpArray(gmp)
def friction_coefficient(gradxy, rhobar, tfric, imagk):
    E_sq = np.square(
        np.subtract(
            gradxy[:, 0, 0, :,],
            gradxy[:, 1, 1, :,],
        )
    )
    F_sq = np.square(
        np.subtract(
            gradxy[:, 0, 1, :,],
            gradxy[:, 1, 0, :,],
        )
    )
    rhoEF = np.multiply(
        rhobar,
        np.add(E_sq, F_sq)
    )
    imagk[:,] = np.divide(tfric, rhoEF)




