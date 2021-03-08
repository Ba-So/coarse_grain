#!/usr/bin/env python
# coding=utf-8
import numpy as np

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
def compute_dyad_Rey(x_vals, y_vals, rho, x_avg_list, y_avg_list, c_mem_idx, coarse_area, ret):
    '''computes the dyadic product of avg(rho X'X') assuming the Reynolds postulates to be accurate'''
    out = []
    for idx_set, c_area, x_avg, y_avg, reti in zip(c_mem_idx, coarse_area, x_avg_list, y_avg_list, ret):
        idx_i = np.extract(np.greater(idx_set, -1), idx_set)
        x_flucts, y_flucts = math.vec_flucts(
            [x_vals[j] for j in idx_i],
            [y_vals[j] for j in idx_i],
            x_avg, y_avg
        )
        constituents = np.array([[x[0] for x in x_flucts], [y[0] for y in y_flucts]])
        rho_set = np.array([rho[j] for j in idx_i])
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
        #normalize: out_tens = np.multiply(out_tens, (-1))
        out_tens = np.divide(dyad, c_area)
        #subtract anisotropic part
       # trace = 0.5 * np.add(out_tens[0,0,:], out_tens[1,1,:])
       # for i in range(2):
       #     out_tens[i,i,:] = np.subtract(out_tens[i,i,:], trace)
        out_tens = np.multiply(out_tens, (-1))
        reti[:,] = out_tens[:]


@TimeThis
@requires({
    'full': ['x_vals', 'y_vals', 'rho'],
    'slice': ['x_avg_list', 'y_avg_list', 'rho_avg', 'c_mem_idx', 'coarse_area', 'ret']
})
@ParallelNpArray(gmp)
def compute_dyad(x_vals, y_vals, rho, x_avg_list, y_avg_list, rho_avg, c_mem_idx, coarse_area, ret):
    '''computes the dyadic product of avg(rho X'X') without relying on the Reynolds postulates'''
    out = []
    for idx_set, c_area, x_avg, y_avg, rhoi, reti in zip(c_mem_idx, coarse_area, x_avg_list, y_avg_list, rho_avg, ret):
        idx_i = np.extract(np.greater(idx_set, -1), idx_set)
        x_set = [x_vals[j] for j in idx_i]
        y_set = [y_vals[j] for j in idx_i]
        rho_set = np.array([rho[j] for j in idx_i])
        plon, plat = math.get_polar(x_set[0][2][0], x_set[0][2][1])
        x_tnd, y_tnd = math.rotate_multiple_to_local(
            plon, plat,
            x_set, y_set
        )
        constituents = np.array([[x[0] for x in x_tnd], [y[0] for y in y_tnd]])
        cell_area = np.array([x[1] for x in x_set])
        # set up uv matrix
            # i, j refers to the first and second components of the vector
            # l refers to the corase-cell members, m and k are level and time
        fluc_tens = np.einsum('ilmk,jlmk->ijlmk', constituents, constituents)
        # average over coarse_area
        fluc_tens = np.einsum(
            'ijlmk,l,lmk->ijmk',
            fluc_tens,
            cell_area,
            rho_set
        )
        #normalize:
        fluc_tens = np.divide(fluc_tens, c_area)
        avg_vec = np.array([x_avg, y_avg])
        avg_tens = np.einsum('imk,jmk,mk->ijmk', avg_vec, avg_vec, rhoi)
        out_tens = np.subtract(fluc_tens, avg_tens)
        trace = 0.5 * np.add(out_tens[0,0,:], out_tens[1,1,:])
        for i in range(2):
            out_tens[i,i,:] = np.subtract(out_tens[i,i,:], trace)

        out_tens = np.multiply(out_tens, (-1))

        reti[:,] = out_tens[:]


@TimeThis
@requires({
    'full': ['x_vals', 'y_vals', 'rho', 'scalar'],
    'slice': ['x_avg_list', 'y_avg_list', 'scalar_avg', 'c_mem_idx', 'coarse_area', 'ret']
})
@ParallelNpArray(gmp)
def compute_dyad_scalar(x_vals, y_vals, rho, scalar, x_avg_list, y_avg_list, scalar_avg, c_mem_idx, coarse_area, ret):
    '''computes the dyadic product of avg(rho X'X')'''
    out = []
    for idx_set, c_area, x_avg, y_avg, s_avg, reti in zip(c_mem_idx, coarse_area, x_avg_list, y_avg_list, scalar_avg, ret):
        idx_i = np.extract(np.greater(idx_set, -1), idx_set)
        x_flucts, y_flucts = math.vec_flucts(
            [x_vals[j] for j in idx_i],
            [y_vals[j] for j in idx_i],
            x_avg, y_avg
        )
        scale_flucts = math.scalar_flucts([scalar[j] for j in idx_i], s_avg)
        constituents = np.array([[x[0] for x in x_flucts], [y[0] for y in y_flucts]])
        scales = np.array([x[0] for x in scale_flucts])
        rho_set = np.array([rho[j] for j in idx_i])
        cell_area = np.array([x[1] for x in x_flucts])
        # set up uv matrix
        product = np.einsum('ilmk,lmk->ilmk', constituents, scales)
        # average over coarse_area
        dyad = np.einsum(
            'ilmk,l,lmk->imk',
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
    tfric[:, ] = (-1) * np.einsum(
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
    return tfric * s_to_d / c_v

def shear_tensor_2D(UV_grad):
    shape = list(np.shape(UV_grad))
    S_tens = np.zeros(shape)
    sumthree = (UV_grad[:, 0, 0, :, :] + UV_grad[:, 1, 1, :, :]) / 2
    for i in range(2):
        for j in range(2):
            if i != j:
                S_tens[:, i, j, :, :] = (UV_grad[:, i, j, :, :] + UV_grad[:, j, i, :, :])/2
            else:
                S_tens[:, i, j, :, :] = np.subtract(UV_grad[:, i, j, :, :], sumthree[:,:,:])
    return S_tens


def norm2_2d_tensor(S_tens):
    shape = list(np.shape(S_tens))
    shape.pop(1)
    shape.pop(1)
    S_norm = np.zeros(shape)
    for i in range(2):
        for j in range(2):
            S_norm = np.add(
                S_norm,
                np.sqrt(
                    np.power(
                        S_tens[:, i, j, :, :],
                        2),
                ))
    return S_norm


@TimeThis
@requires({
    'full': ['d_filt'],
    'slice': ['coarse_sm_tens', 'uv_grad', 'T', 'smag_fric']
})
@ParallelNpArray(gmp)
def smag_fric(d_filt, coarse_sm_tens, uv_grad, T, smag_fric):
    c_s = 0.2
    smag_fric[:,] = np.divide(
        (-2) * c_s * np.einsum(
            'ijklm,ijklm->ilm',
            coarse_sm_tens,
            uv_grad
        ),
        T
    )


@TimeThis
@requires({
    'full' : [],
    'slice' : ['rhoxy', 'gradex', 't_presc']
})
@ParallelNpArray(gmp)
def turb_pres(rhoxy, gradex, t_presc):
    '''computes the turbulent conversion of kinetic to internal energy, through pressure gradient'''
    c_p = 1004.64
    t_presc[:, ] = (-1) * np.einsum(
        'kilm,kilm->klm',
        rhoxy,
        gradex
    ) * (c_p)

@TimeThis
@requires({
    'full' : [],
    'slice': ['rhoxy', 'gradT', 'T', 't_heat']
})
@ParallelNpArray(gmp)
def turb_heat(rhoxy, gradT, T, t_heat):
    '''computes the turbulent heat flux'''
    c_p = 1004.64
#   gradT_T = np.divide(gradT, T)
    t_heat[:, ] = np.divide((-1) * np.einsum(
        'kilm,kilm->klm',
        rhoxy,
        gradT
    ) * (c_p), T)

@TimeThis
@requires({
    'full' : [],
    'slice': ['rhoxw', 'zstar_grad', 't_nstpy']
})
@ParallelNpArray(gmp)
def turb_enstrophy(rhoxw, zstar_grad, t_nstpy):
    '''computes the turbulent enstrophy flux'''
    t_nstpy[:, ] = (-1) * np.einsum(
        'kilm, kilm->klm',
        rhoxw,
        zstar_grad)

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

def compute_f(lat):
    '''computes the coreolis parameter f for given latitude'''
    Omega = 2*np.pi / 86400 # 1/s
    return np.multiply(np.sin(lat), (2 * Omega))

@TimeThis
@requires({
    'full' : [],
    'slice': ['exner', 'pressure']
})
@ParallelNpArray(gmp)
def exner_to_pressure(exner, pressure):
    '''computes the pressure from the exner pressure and the surface pressure'''
    c_p = 1004.64
    R_d = 287.64
    p_zero = 100000 #Pa
    pressure[:,] = p_zero * np.power(exner[:,], (c_p / R_d))

@TimeThis
@requires({
    'full' : [],
    'slice': ['gradp', 'rho', 'lat', 'u_g', 'v_g']
})
@ParallelNpArray(gmp)
def geostrophic_wind(gradp, rho, lat, u_g, v_g):
    ''' dh p, rho, latlon'''
    f = compute_f(lat)
    rho_f = np.zeros(np.shape(rho))
    for i, f_i in enumerate(f):
        rho_f[i,] = np.multiply(rho[i,], f_i)

    bound = 10 * np.pi / 180
    #pick cells where lat is not too close to equator
    latcrit = np.all([np.greater_equal(lat, bound), np.less_equal(lat, -bound)], 0)
    for crit, iu, iv, irho_f, ip in zip(latcrit, u_g, v_g, rho_f, gradp):
        iu[:,] = np.divide(ip[0, :,], irho_f[:, ]) * (-1)
        iv[:,] = np.divide(ip[1, :,], irho_f[:, ])

@TimeThis
@requires({
    'full' : [],
    'slice': ['theta', 'exner', 'T']
})
@ParallelNpArray(gmp)
def thet_ex_to_T(theta, exner, T):
    ''' converts theta and exner to T'''
    T[:,] = np.einsum('ijk,ijk->ijk',
                      theta,
                      exner)

@TimeThis
@requires({
    'full' : [],
    'slice': ['theta', 'grad_ex', 'lat', 'u_g', 'v_g']
})
#@ParallelNpArray(gmp)
def geostrophic_wind_2(theta, grad_ex, lat, u_g, v_g):
    ''' dh p, rho, latlon'''

    f = compute_f(lat)

    bound = 10 * np.pi / 180
    print(bound)
    c_p = 1004.64
    #pick cells where lat is not too close to equator
    latcrit = np.any([np.greater_equal(lat, bound), np.less_equal(lat, -bound)], 0)
    for crit, iu, iv, iex, ithet, fi in zip(latcrit, u_g, v_g, grad_ex, theta, f):
        if crit:
            c_f = c_p / fi
            iu[:,] = np.multiply(np.multiply(ithet[:,], iex[1,:, ]),((-1) * c_f))
            iv[:,] = np.multiply(np.multiply(ithet[:,], iex[0,:, ]), c_f)
         #   iu[:,] = c_p / fi * iex[1,:, ] * (-1)
         #   iv[:,] = c_p / fi * iex[0,:, ]

        else:
            pass
