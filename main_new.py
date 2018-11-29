#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys as sys
from decorators.paralleldecorators import Mp, ParallelNpArray, shared_np_array
from decorators.debugdecorators import TimeThis, PrintArgs, PrintReturn
import modules.cio as cio


mp = Mp(False, 2)
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
        self.gridnfo = [[i, cell_area[i], [xlon[i], xlat[i]]] for i in range(len(xlat))]

    def nfo_merge(self, data):
        return [[data[i]] + nfo for i,nfo in enumerate(self.gridnfo)]

    def execute(self):
        U = self.IO.load_from('data', 'U')
        U = self.nfo_merge(U)
        print(len(U))
        #stuff


if __name__ == '__main__':

    path = '/home1/kd031/projects/icon/experiments/BCWcold'
    gridfile = r'iconR\dB\d{2}-grid_refined_\d{1}.nc'
    datafile = r'BCWcold_R2B07_slice.nc'
    cg = CoarseGrain(path, gridfile, datafile)
    print cg._ready
    cg.execute()


