#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys as sys
import itertools
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

    def bar_avg_2Dvec(self, xname, yname):
        rho = self.IO.load_from('data', 'RHO')
        print(gmp.switch)
        varshape = np.shape(rho)
        rho = self.nfo_merge(rho)
        rho_bar = self.create_array(varshape)
        print('computing the coarse density values ...')
        math.bar_scalar(rho, self.c_area, self.c_mem_idx, rho_bar)

        X = self.IO.load_from('data', xname)
        Y = self.IO.load_from('data', yname)
        X = self.nfo_merge(X)
        Y = self.nfo_merge(Y)
        X_hat = self.create_array(varshape)
        Y_hat = self.create_array(varshape)
        print('computing the deninsity weighted coarse vector components ...')
        math.hat_2Dvector(X, Y, rho, rho_bar, self.c_area, self.c_mem_idx, X_hat, Y_hat)
        print(np.mean(X_hat), np.mean(Y_hat))
        return X_hat, Y_hat

    def xy_hat_gradients(self, xdata, ydata):
        varshape = list(np.shape(xdata))
        varshape.insert(1, 4)
        varshape.insert(2, 2)
        xy_gradient = self.create_array(varshape)
        X = self.nfo_merge(xdata)
        Y = self.nfo_merge(ydata)
        math.xy_2D_gradient(
            X, Y,
            self.g_mem_idx, self.g_coords_rads,
            self.c_area, xy_gradient)
        return xy_gradient

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
        self.c_mem_idx = self.IO.load_from('grid', 'area_member_idx').astype(int)
        self.c_area = self.IO.load_from('grid', 'coarse_area')
        self.g_mem_idx = self.IO.load_from('grid', 'member_idx')
        g_coords = np.moveaxis(self.IO.load_from('grid', 'coords'), 0, -1)
        # Hack due to wrong output of grid prepare
        g_rads = np.moveaxis(self.IO.load_from('grid', 'member_rad'), 0, -1)
        print(np.shape(g_coords))
        self.g_coords_rads = [
            [[gci, gri] for gci, gri in itertools.izip(gc, gr)] for gc, gr in itertools.izip(
                g_coords, g_rads
            )]
        print(np.shape(self.g_coords_rads))

    def create_array(self, shape):
        data_list = np.zeros(shape)
        shrd_list = shared_np_array(np.shape(data_list))
        shrd_list[:] = data_list[:]
        return shrd_list

    def nfo_merge(self, data):
        data_merge = [[data[i]] + nfo for i,nfo in enumerate(self.gridnfo)]
        return data_merge

    def execute(self):
        print('computing U and V hat')
        U_hat, V_hat = self.bar_avg_2Dvec('U', 'V')
        print('computing U and V grad')
        UV_gradients = self.xy_hat_gradients(U_hat, V_hat)

    def testing(self):
        xname = 'U'
        yname = 'V'
        X = self.IO.load_from('data', xname)
        Y = self.IO.load_from('data', yname)
        varshape = list(np.shape(X))
        X = self.nfo_merge(X)
        Y = self.nfo_merge(Y)
        varshape.insert(1, 4)
        varshape.insert(2, 2)
        xy_gradient = self.create_array(varshape)
        math.xy_2D_gradient(
            X, Y,
            self.g_mem_idx, self.g_coords_rads,
            self.c_area, xy_gradient)
        return xy_gradient




if __name__ == '__main__':
    path = '/home1/kd031/projects/icon/experiments/BCWcold'
    gridfile = r'iconR\dB\d{2}-grid_refined_\d{1}.nc'
    datafile = r'BCWcold_R2B07_slice.nc'
    gmp.set_parallel_proc(True)
    gmp.set_num_procs(16)
    cg = CoarseGrain(path, gridfile, datafile)
    print cg._ready
    cg.testing()
    #cg.execute()


