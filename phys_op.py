# This is the library of physics computations need in coarse_grain

import numpy as np

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

def potT_to_T_exner(vpT, Exner):
    '''
    Computes the Temperature from given virtual potential Temperature
    and Exner Pressure values
    vpT - virtual potential Temperature
    Exner - Exner Pressure
    Formula: T = Exner * vpT
    '''
    T = np.multiply(vpT, Exner)

    return T


