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

    def bar_avg_2Dvec(self, xname, yname, numfile=0):
        rho = self.IO.load_from('data', 'RHO', numfile)
        print(gmp.switch)
        varshape = np.shape(rho)
        rho = self.nfo_merge(rho)
        rho_bar = self.create_array(varshape)
        print('computing the coarse density values ...')
        math.bar_scalar(rho, self.c_area, self.c_mem_idx, rho_bar)

        X = self.IO.load_from('data', xname, numfile)
        Y = self.IO.load_from('data', yname, numfile)
        X = self.nfo_merge(X)
        Y = self.nfo_merge(Y)
        X_hat = self.create_array(varshape)
        Y_hat = self.create_array(varshape)
        print('computing the density weighted coarse vector components ...')
        math.hat_2Dvector(X, Y, rho, rho_bar, self.c_area, self.c_mem_idx, X_hat, Y_hat)
        return X_hat, Y_hat

    def xy_hat_gradients(self, xdata, ydata):
        varshape = list(np.shape(xdata))
        varshape.insert(1, 2)
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

    def xy_local_gradients(self, xdata, ydata=None):
        lg_mem_idx = self.IO.load_from('grid', 'local_grad_idx')
        lg_coords = self.IO.load_from('grid', 'local_grad_coords')
        lg_rads = self.IO.load_from('grid', 'local_grad_dist')
        self.lg_coords_rads = [
            [[gci, gri] for gci, gri in zip(gc, gr)] for gc, gr in zip(
                lg_coords, lg_rads
            )]

        varshape = list(np.shape(xdata))
        varshape.insert(1, 2)
        X = self.nfo_merge(xdata)
        print('computing the xy gradients....')
        if ydata:
            varshape.insert(2,2)
            gradient = self.create_array(varshape)
            Y = self.nfo_merge(ydata)
            math.xy_2D_gradient(
                X, Y,
                lg_mem_idx, lg_coords_rads,
                lc_area, gradient)
        else:
            gradient = self.create_array(varshape)
            math.x_2D_gradient(
                X,
                lg_mem_idx, lg_coords_rads,
                lc_area, gradient)

        return gradient

    def rhoxy_averages(self, xname, yname, xavg, yavg, numfile=0):
        varshape = list(np.shape(xavg))
        varshape.insert(1, 2)
        varshape.insert(1, 2)
        rhoxy = self.create_array(varshape)
        x = self.IO.load_from('data', xname, numfile)
        y = self.IO.load_from('data', yname, numfile)
        x_avg = self.IO.load_from('data', xavg, numfile)
        y_avg = self.IO.load_from('data', yavg, numfile)
        rho = self.IO.load_from('data', 'RHO', numfile)
        x = self.nfo_merge(x)
        y = self.nfo_merge(y)
        print('computing the rhoxy values ...')
        phys.compute_dyad(
            x, y, rho,
            x_avg, y_avg,
            self.c_mem_idx, self.c_area,
            rhoxy
        )
        return rhoxy

    def turbulent_shear_prod(self, rhoxy, gradxy, filenum=0):
        print('computing the turbulent shear production values ...')
        varshape = list(np.shape(rhoxy))
        varshape.pop(1)
        varshape.pop(1)
        t_fric = self.create_array(varshape)
        phys.turb_fric(rhoxy, gradxy, t_fric)
        print('saving to file ...')
        self.IO.write_to('data', turbfric, name='T_FRIC',
                        attrs={
                            'long_name' : 'turbulent_friction',
                            'coordinates': 'clat clon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        return t_fric

    def friction_coefficient(self, gradxy, t_fric, numfile=0):
        rho = self.IO.load_from('data', 'RHO', numfile)
        varshape = np.shape(rho)
        rho = self.nfo_merge(rho)
        rho_bar = self.create_array(varshape)
        print('computing the coarse density values ...')
        math.bar_scalar(rho, self.c_area, self.c_mem_idx, rho_bar)

        print('computing the K values ...')
        varshape = list(np.shape(t_fric))
        kimag = self.create_array(varshape)
        phys.friction_coefficient(gradxy, rho_bar, t_fric, kimag)
        print('saving to file ...')
        self.IO.write_to('data', K, name='K_TURB',
                        attrs={
                            'long_name' : 'turbulent dissipation coefficiet',
                            'coordinates': 'clat clon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=numfile
                        )
        return kimag

    def turb_fric_Kd(self, numfile):
        t_fric = self.IO.load_from('data', 'T_FRIC', numfile)
        turberich = phys.turb_fric_erich(t_fric)
        self.IO.write_to('data', turberich, name='T_ERICH',
                         attrs={
                             'long_name' : 'turbulent_shear_erich',
                             'coordinates': 'clat clon',
                             '_FillValue' : float('nan'),
                             'grid_type' : 'unstructured',
                             'units' : 'K/d'
                             }, filenum=numfile
                         )

    def geostrophic_winds(self, zanumfile=0):
        rho = self.IO.load_from('data', 'RHO', numfile)
        exner = self.IO.load_from('data', 'EXNER', numfile)

        varshape = list(np.shape(exner))
        pressure = self.create_array(varshape)
        phys.exner_to_pressure(exner, pressure)
        del exner
        grad_p = xy_local_gradients(pressure)
        del pressure
        xlat = self.IO.load_from('grid', 'vlat')
        varshape = list(np.shape(exner))
        u_g = self.create_array(varshape)
        v_g = self.create_array(varshape)
        phys.geostrophic_wind(grad_p, rho, xlat, u_g, v_g)
        print('writing to file...')
        self.IO.write_to('data', u_g, name='U_GEO',
                        attrs={
                            'long_name': 'geostrophically balanced zonal wind',
                            'coordinates': 'clat clon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=numfile
                        )
        self.IO.write_to('data', v_g, name='V_GEO',
                        attrs={
                            'long_name': 'geostrophically balanced meridional wind',
                            'coordinates': 'clat clon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=numfile
                        )

        return u_g, v_g

    def prepare_winds(self, filenum=0):
        print('processing file number {}'.format(filenum + 1))
        print('computing U and V hat')
        U_hat, V_hat = self.bar_avg_2Dvec('U', 'V', filenum)
        print('saving to file ...')
        self.IO.write_to('data', U_hat, name='U_HAT',
                        attrs={
                            'long_name': 'density weighted coarse zonal wind',
                            'coordinates': 'clat clon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        self.IO.write_to('data', V_hat, name='V_HAT',
                        attrs={
                            'long_name': 'density weighted coarse meridional wind',
                            'coordinates': 'clat clon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        print('computing U and V grad')
        UV_gradients = self.xy_hat_gradients(U_hat, V_hat)
        if debug:
            self.IO.write_to('data', UV_gradients[:,0,0,:,:], name='DUX',
                            attrs={
                                'long_name': 'density weighted coarse meridional wind',
                                'coordinates': 'clat clon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
            self.IO.write_to('data',  UV_gradients[:,0,1,:,:], name='DVX',
                            attrs={
                                'long_name': 'density weighted coarse meridional wind',
                                'coordinates': 'clat clon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
            self.IO.write_to('data',  UV_gradients[:,1,0,:,:], name='DUY',
                            attrs={
                                'long_name': 'density weighted coarse meridional wind',
                                'coordinates': 'clat clon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
            self.IO.write_to('data',  UV_gradients[:,1,1,:,:], name='DVY',
                            attrs={
                                'long_name': 'density weighted coarse meridional wind',
                                'coordinates': 'clat clon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )

    def load_grads(self, filenum=0):
        varshape = list(np.shape(self.IO.load_from('data', 'DUX', filenum)))
        varshape.insert(1, 2)
        varshape.insert(2, 2)
        UV_gradients = np.empty(varshape)
        UV_gradients[:, 0, 0, :, :] = self.IO.load_from('data', 'DUX', filenum)
        UV_gradients[:, 0, 1, :, :] = self.IO.load_from('data', 'DVX', filenum)
        UV_gradients[:, 1, 0, :, :] = self.IO.load_from('data', 'DUY', filenum)
        UV_gradients[:, 1, 1, :, :] = self.IO.load_from('data', 'DVY', filenum)

        return UV_gradients

