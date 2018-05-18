#!/usr/bin/env python
# coding=utf-8
import numpy as np


def reorder(x):
    n_procs = np.shape(x)[0]
    out_shape = list(np.shape(x[0]))
    out_shape[0] = 0
    for i in range(n_procs):
        out_shape[0] = out_shape[0]+np.shape(x[i])[0]

    out = np.empty(out_shape)
    out.fill(0)
    for i,element in enumerate(x):
        for j,k in enumerate(element):
            out[i+j*n_procs] = k

    return out

