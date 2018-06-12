#!/usr/bin/env python
# coding=utf-8

#update functions for global
# update for dicts??
import global_var as gv
globals_dict = {
    'grid_nfo' : gv.grid_nfo,
    'gradient_nfo' : gv.gradient_nfo,
    'data_run' : gv.data_run
}

functions ={
    'update_all' : update_all,
    'update_dict': update_dict,
    'update_array': update_array
}

def update(varn, values, function = None, indices = None):
    if indices:
        global indices = indices
    if function:
        functions[function](varn, values, )
    else:
        update_all(varn, values)
    return None

def update_all(varn, values):
    global globals_dict
    globals_dict[varn] = values
    return None

def update_dict(varn, values):
    global globals_dict
    global_dict[varn].update(values)

def update_array(varn, values):
    global indices
    global globals_dict
    globals_dict[varn][indices] = values



