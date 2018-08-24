#!/usr/bin/env python
# coding=utf-8
# This is the library of physics computations need in coarse_grain
import numpy as np
from multiprocessing import Pool
import global_vars as gv
import update as up


def potT_to_T_pressure(vpT, P, atmos):
    # so far only for the dry atmosphere!!
    '''
    Computes the Temperature form given virtual potential Temparature values
    defined by:
        vpT - \Theta_v, virtual potential Temperature
        P - pressure
        grid_nfo - contains information on grid
        atmos- control string dry/moist atmosphere model
        (mainly as reminder)
        \Theta_v = \Theta*(1+ 0.61r- r_l)
        r - is mixing ratio of wate
        rl - is mixing ratio of liquid water
        \Theta = T*(Pref/P)^(R/c_p)
        Pref - standard reference pressure (1000 mbar)
        R - gas constant of Air
        c_p - specific heat
    '''
    # as used in ICON-IAP
    Pref = 100000.0 #reference pressure
    rd = 287.04 #gas constant, dry air
    cpd = 1004.64 #specific heat at constant pressure
    R_o_cpd = rd / cpd #useful

    # heavy use of numpy functions for speedup
    P_o_Pref = np.divide(P, Pref)
    P_o_Pref = np.power(P_o_Pref,R_o_cpd)
    T = np.multiply(vpT, P)

    return T
def potT_to_T_exner():
    update = up.Updater()

    if gv.mp.get('mp'):
        pool = Pool(processes = gv.mp['n_procs'])
        result = (pool.map(potT_to_T_exner_sub, gv.mp['slices']))

        out = np.zeros(
            [
                gv.globals_dict['grid_nfo']['ncells'],
                gv.globals_dict['grid_nfo']['ntim'],
                gv.globals_dict['grid_nfo']['nlev']
            ]
        )
        for i in range(len(gv.mp['slices'])):
            out[gv.mp['slices'][i]] = result[i]
        pool.close()
        del result

    else:
        out = []
        for i in range(gv.globals_dict['grid_nfo']['ncells']):
            out.append(potT_to_T_exner_sub(i))

    update.up_entry('data_run', {'T' : np.array(out)})
    print(np.array(out).shape)
    del out
    return None

def potT_to_T_exner_sub(i_slice):
    '''
    Computes the Temperature from given virtual potential Temperature
    and Exner Pressure values
    vpT - virtual potential Temperature
    Exner - Exner Pressure
    Formula: T = Exner * vpT
    '''

    T = np.multiply(
        gv.globals_dict['data_run']['THETA_V'][i_slice],
        gv.globals_dict['data_run']['EXNER'][i_slice]
    )

    return T

def K():
    update = up.Updater()

    if gv.mp.get('mp'):
        pool = Pool(processes = gv.mp['n_procs'])
        result = (pool.map(K_sub, gv.mp['slices']))
        out = np.zeros(
            [
                gv.globals_dict['grid_nfo']['ncells'],
                gv.globals_dict['grid_nfo']['ntim'],
                gv.globals_dict['grid_nfo']['nlev']
            ]
        )
        for i in range(len(gv.mp['slices'])):
            out[gv.mp['slices'][i]] = result[i]
        pool.close()
        del result
    else:
        out = []
        for i in range(gv.globals_dict['grid_nfo']['ncells']):
            out.append(K_sub(i))

    update.up_entry('data_run', {'K' : out})

def K_sub(i_slice):
    # Todo: compute E^2 and F^2 and divide dyad ( rhov''v'') by rho(e^2+f^2)
    E_sq = np.square(np.subtract(
            gv.globals_dict['data_run']['gradient'][i_slice,0,0,:,], #dxu
            gv.globals_dict['data_run']['gradient'][i_slice,1,1,:,]  #dyv
        )
    )

    F_sq = np.square(
        np.add(
            gv.globals_dict['data_run']['gradient'][i_slice,0,1,:], #dxv
            gv.globals_dict['data_run']['gradient'][i_slice,1,0,:]    #dyu
        )
    )

    rho_EF = np.multiply(
        gv.globals_dict['data_run']['RHO_bar'][i_slice],
        np.add(E_sq, F_sq)
    )

    return np.divide(
        gv.globals_dict['data_run']['turb_fric'][i_slice],
        rho_EF
    )

def turb_fric():
    update = up.Updater()

    if gv.mp.get('mp'):
        pool = Pool(processes = gv.mp['n_procs'])
        result = (pool.map(turb_fric_sub, gv.mp['slices']))
        out = np.zeros(
            [
                gv.globals_dict['grid_nfo']['ncells'],
                gv.globals_dict['grid_nfo']['ntim'],
                gv.globals_dict['grid_nfo']['nlev']
            ]
        )
        for i in range(len(gv.mp['slices'])):
            out[gv.mp['slices'][i]] = result[i]
        pool.close()
        del result
    else:
        out = []
        for i in range(gv.globals_dict['grid_nfo']['ncells']):
            out.append(turb_fric_sub(i))

    update.up_entry('data_run', {'turb_fric' : -1 * np.array(out)})
    del out
    return None


def turb_fric_sub(i_slice):
    ''' computes \overline{rho v'' v''} ** nabla \hat{v}'''
    t_fric = np.einsum(
        'kijlm,kijlm->klm',
        gv.globals_dict['data_run']['dyad'][i_slice],
        gv.globals_dict['data_run']['gradient'][i_slice]
    )

    return t_fric

