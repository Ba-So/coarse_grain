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
        print('computing the density weighted coarse vector components ...')
        math.hat_2Dvector(X, Y, rho, rho_bar, self.c_area, self.c_mem_idx, X_hat, Y_hat)
        return X_hat, Y_hat

    def xy_hat_gradients(self, xdata, ydata):
        varshape = list(np.shape(xdata))
        varshape.insert(1, 4)
        varshape.insert(2, 2)
        xy_gradient = self.create_array(varshape)
        X = self.nfo_merge(xdata)
        Y = self.nfo_merge(ydata)
        print('computing the xy gradients....')
        math.xy_2D_gradient(
            X, Y,
            self.g_mem_idx, self.g_coords_rads,
            self.c_area, xy_gradient)
        return xy_gradient

    def rhoxy_averages(self, xname, yname, xavg, yavg):
        varshape = list(np.shape(xavg))
        varshape.insert(1, 2)
        varshape.insert(1, 2)
        rhoxy = self.create_array(varshape)
        x = self.IO.load_from('data', xname)
        y = self.IO.load_from('data', yname)
        rho = self.IO.load_from('data', 'RHO')
        x = self.nfo_merge(x)
        y = self.nfo_merge(y)
        print('computing the rhoxy values ...')
        phys.compute_dyad(
            x, y, rho,
            xavg, yavg,
            self.c_mem_idx, self.c_area,
            rhoxy
        )
        return rhoxy

    def turbulent_friction(self, rhoxy, gradxy):
        print('computing the turbulent friction values ...')
        varshape = list(np.shape(rhoxy))
        varshape.pop(1)
        varshape.pop(1)
        t_fric = self.create_array(varshape)
        phys.friction_coefficient(rhoxy, gradxy, t_fric)
        return t_fric

    def friction_coefficient(self, gradxy, t_fric):
        rho = self.IO.load_from('data', 'RHO')
        varshape = np.shape(rho)
        rho = self.nfo_merge(rho)
        rho_bar = self.create_array(varshape)
        print('computing the coarse density values ...')
        math.bar_scalar(rho, self.c_area, self.c_mem_idx, rho_bar)

        print('computing the K values ...')
        varshape = list(np.shape(t_fric))
        kimag = self.create_array(varshape)
        phys.friction_coefficient(gradxy, rho_bar, t_fric, kimag)
        return kimag



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
        # Hack due to wrong output of grid prepare
        g_coords = np.moveaxis(self.IO.load_from('grid', 'coords'), 0, -1)
        g_rads = np.moveaxis(self.IO.load_from('grid', 'member_rad'), 0, -1)
        self.g_coords_rads = [
            [[gci, gri] for gci, gri in itertools.izip(gc, gr)] for gc, gr in itertools.izip(
                g_coords, g_rads
            )]

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
        rhouv_flucts = self.rhoxy_averages('U', 'V', U_hat, V_hat)
        self.IO.write_to('data', U_hat, name='U_HAT', attrs={'long name': 'density weighted coarse zonal wind'})
        self.IO.write_to('data', V_hat, name='V_HAT', attrs={'long name': 'density weighted coarse meridional wind'})
        del U_hat, V_hat
        turbfric = self.turbulent_friction(rhouv_flucts, UV_gradients)
        del rhouv_flucts
        self.IO.write_to('data', turbfric, name='T_FRIC', attrs={'long name' : 'turbulent_friction'})
        K = self.friction_coefficient(UV_gradients, turbfric)
        self.IO.write_to('data', K, name='K_TURB', attrs={'long name' : 'turbulent dissipation coefficient'})

    def testing(self):
        xname = 'U'
        yname = 'V'
        X = self.IO.load_from('data', xname)
        Y = self.IO.load_from('data', yname)
        rhoxy = self.rhoxy_averages('U', 'V', X, Y)




if __name__ == '__main__':
    path = '/home1/kd031/projects/icon/experiments/BCWcold'
    gridfile = r'iconR\dB\d{2}-grid_refined_\d{1}.nc'
    datafile = r'BCWcold_R2B07_slice.nc'
    gmp.set_parallel_proc(True)
    gmp.set_num_procs(16)
    cg = CoarseGrain(path, gridfile, datafile)
    print cg._ready
    #cg.testing()
    cg.execute()


