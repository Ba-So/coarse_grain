#!/usr/bin/env python
# coding=utf-8
import numpy as np
import global_vars as gv
import update as up


def reorder(x):
    ''' stub '''
    dims = [gv.globals_dict['grid_nfo']['ncells']]
    out = np.zeros(dims)
    del dims

    chunks = gv.mp['chunks']
    for i, chunk in enumerate(chunks):
        out[chunk] = x[i]


    return out

def out_to_global(x):
    '''
        molds the output array from pool computing into data_run
        expected input: list of length n_procs
        uses the information on gv.mp to slice subinfos into data_run
    '''
    update = up.Updater()
    for i, elem in enumerate(x):
        update.part('data_run', elem, gv.mp['slices'][i])
    return None



def prepare_mp(num_procs):
    '''
        prepares the multiprocessing for using pool
        chunks: n arrays containing indices of ncells to walk over
        slices: containg slicers to slice chunks
        n_procs: number of parallel processes
        chunk_len and last_chunk : probably superflous
    '''

    update = up.Updater()
    ncells = gv.globals_dict['grid_nfo']['ncells']

    chunk_len = ncells // num_procs

    last_chunk = chunk_len + ncells % num_procs

    # for convenience
    chunks = []

    for j in range(0, num_procs - 1):
        chunks.append([i + j * chunk_len for i in range(chunk_len)])

    chunks.append([(i + (num_procs-1) * chunk_len) for i in range(last_chunk)])

    # for reconsolidating using Updater().part(), this very much assumes that
    # ncells is the first index of the arrays - otherwise all goes boom
    slices = []
    for i in range(num_procs - 1):
        slices.append(slice(i * chunk_len, (i+1) * chunk_len))

    slices.append(slice((num_procs-1) * chunk_len, ncells))

    iterator = [i for i in range(ncells)]

    update.up_mp(
        {
            'chunks' : chunks,
            'slices' : slices,
            'n_procs' : num_procs,
            'chunk_len' : chunk_len,
            'last_chunk': last_chunk,
            'iterator' : iterator
        }
    )

    return None


