#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys as sys
from decorators.paralleldecorators import gmp, ParallelNpArray, shared_np_array
from decorators.debugdecorators import TimeThis, PrintArgs, PrintReturn
import modules.math_mod as math
import modules.phys_mod as phys
import modules.cio as cio


class Operations(object):
    def func():
        #wrappers deciding how much information a part of the code recieves.
        #either whole or only slices.
        return func()

class CoarseGrain(Operations):

    def __init__(self, path, grid, data):
        self._ready = True
        self.IO = cio.IOcontroller(path, grid, data)
        self.prepare()

    def set_mp(self, switch, num_procs):
        Mp.toggle_switch(switch)
        Mp.change_num_procs(num_procs)

    def set_ready(self, state):
        self._ready = state

    def prepare(self):
        xlat = self.IO.load_from('grid', 'lat')
        xlon = self.IO.load_from('grid', 'lon')
        cell_area = self.IO.load_from('grid', 'cell_area')
        self.gridnfo = [[cell_area[i], [xlon[i], xlat[i]]] for i in range(len(xlat))]
        self.c_mem_idx = self.IO.load_from('grid', 'cell_neighbor_idx')
        self.c_area = self.IO.load_from('grid', 'coarse_area')

    def create_array(self, shape):
        data_list = np.zeros(shape)
        shrd_list = shared_np_array(np.shape(data_list))
        shrd_list[:] = data_list[:]
        return shrd_list

    def nfo_merge(self, data):
        data_merge = [[data[i]] + nfo for i,nfo in enumerate(self.gridnfo)]
        return data_merge

    def execute(self):
        U = self.IO.load_from('data', 'U')
        varshape = np.shape(U)
        print(np.mean(U))
        U = self.nfo_merge(U)
        U_bar = self.create_array(varshape)
        gmp.set_parallel_proc(True)
        gmp.set_num_procs(8)
        math.bar_scalar(U, self.c_area, self.c_mem_idx, U_bar)
        print(np.mean(U_bar))
        print(U_bar[30,1,1])
        print(U[30][0][1,1])

        #]stuff


if __name__ == '__main__':
    path = '/home1/kd031/projects/icon/experiments/BCWcold'
    gridfile = r'iconR\dB\d{2}-grid_refined_\d{1}.nc'
    datafile = r'BCWcold_R2B07_slice.nc'
    gmp.set_parallel_proc(True)
    gmp.set_num_procs(8)
    cg = CoarseGrain(path, gridfile, datafile)
    print cg._ready
    cg.execute()


