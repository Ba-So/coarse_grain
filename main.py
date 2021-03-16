#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys as sys
from os import path
import argparse
from decorators.paralleldecorators import gmp, shared_np_array
import modules.math_mod as math
import modules.phys_mod as phys
import modules.cio as cio
import modules.new_grad_mod as grad

class Operations(object):
    def func():
        #wrappers deciding how much information a part of the code recieves.
        #either whole or only slices.
        return None

    def bar_avg_2Dvec(self, xname, yname, numfile=0):
        rho = self.IO.load_from('data', 'RHO', numfile)
        varshape = np.shape(rho)
        rho = self.nfo_merge(rho)
        if not self.IO.isin('newdata', 'RHO_BAR'):
            rho_bar = self.create_array(varshape)
            print('computing the coarse density values ...')
            math.bar_scalar(rho, self.c_area, self.c_mem_idx, rho_bar)
            print('saving to file ...')
            self.IO.write_to('data', rho_bar, name='RHO_BAR',
                            attrs={
                                'long_name': 'area weighted coarse mean of rho',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=numfile
                            )
        else:
            rho_bar = self.IO.load_from('newdata', 'RHO_BAR', numfile)


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
        idxlist = np.arange(np.shape(xdata)[0])
        print('computing the xy gradients....')
        grad.vector_gradient(
            X, Y,
            idxlist,
            self.g_coords, self.g_dist,
            self.int_idx, self.int_dist,
            xy_gradient)
        return xy_gradient

    def xy_fine_gradients(self, xdata, ydata):
        varshape = list(np.shape(xdata))
        varshape.insert(1, 2)
        varshape.insert(2, 2)
        xy_gradient = self.create_array(varshape)
        X = self.nfo_merge(xdata)
        Y = self.nfo_merge(ydata)
        idxlist = np.arange(np.shape(xdata)[0])
        print('computing the xy gradients....')
        int_idx = self.IO.load_from('grid', 'f_int_idx')
        int_dist = self.IO.load_from('grid', 'f_int_dist')
        g_coords = self.IO.load_from('grid', 'f_grad_coords')
        g_dist = self.IO.load_from('grid', 'f_grad_dist')
        grad.vector_gradient(
            X, Y,
            idxlist,
            g_coords, g_dist,
            int_idx, int_dist,
            xy_gradient)
        return xy_gradient

    def x_hat_gradients(self, xdata):
        varshape = list(np.shape(xdata))
        varshape.insert(1,2)
        xy_gradient = self.create_array(varshape)
        X = self.nfo_merge(xdata)
        print('computing the xy gradients....')
        grad.scalar_gradient(
            X,
            self.g_coords, self.g_dist,
            self.int_idx, self.int_dist,
            xy_gradient
        )
        return xy_gradient


    def hat_avg_scalar(self, data, numfile=0):
        rho = self.IO.load_from('data', 'RHO', numfile)
        print(gmp.switch)
        varshape = np.shape(rho)
        rho = self.nfo_merge(rho)
        rho_bar = self.IO.load_from('newdata', 'RHO_BAR', numfile)
        X = self.IO.load_from('data', data, numfile)
        X = self.nfo_merge(X)
        X_hat = self.create_array(varshape)
        math.hat_scalar(X, rho, rho_bar, self.c_area, self.c_mem_idx, X_hat)
        return X_hat


    def rhoxy_averages(self, xname, yname, xavg, yavg, numfile=0):
        '''(compute tau(full), without reynolds assumption)'''
        debug = False
        x = self.IO.load_from('data', xname, numfile)
        y = self.IO.load_from('data', yname, numfile)
        x_avg = self.IO.load_from('newdata', xavg, numfile)
        y_avg = self.IO.load_from('newdata', yavg, numfile)
        rho = self.IO.load_from('data', 'RHO', numfile)
        rho_bar = self.IO.load_from('newdata', 'RHO_BAR', numfile)
        x = self.nfo_merge(x)
        y = self.nfo_merge(y)
        varshape = list(np.shape(x_avg))
        varshape.insert(1, 2)
        varshape.insert(1, 2)
        rhoxy = self.create_array(varshape)
        print('computing the rhoxy values ...')
        phys.compute_dyad(
            x, y, rho,
            x_avg, y_avg, rho_bar,
            self.c_mem_idx, self.c_area,
            rhoxy
        )
        if debug:
            self.IO.write_to('data', rhoxy[:,0,0,:,:], name='TAU11',
                            attrs={
                                'long_name': 'tau 11',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=numfile
                            )
            self.IO.write_to('data', rhoxy[:,0,1,:,:], name='TAU12',
                            attrs={
                                'long_name': 'tau 12',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=numfile
                            )
            self.IO.write_to('data', rhoxy[:,1,0,:,:], name='TAU21',
                            attrs={
                                'long_name': 'tau 21',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=numfile
                            )
            self.IO.write_to('data', rhoxy[:,1,1,:,:], name='TAU22',
                            attrs={
                                'long_name': 'tau 22',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=numfile
                            )
        return rhoxy

    def rhoxy_averages_re(self, xname, yname, xavg, yavg, numfile=0):
        '''Reynolds Stress Tensor computation'''
        debug = False
        x = self.IO.load_from('data', xname, numfile)
        y = self.IO.load_from('data', yname, numfile)
        x_avg = self.IO.load_from('newdata', xavg, numfile)
        y_avg = self.IO.load_from('newdata', yavg, numfile)
        rho = self.IO.load_from('data', 'RHO', numfile)
        rho_bar = self.IO.load_from('newdata', 'RHO_BAR', numfile)
        x = self.nfo_merge(x)
        y = self.nfo_merge(y)
        varshape = list(np.shape(x_avg))
        varshape.insert(1, 2)
        varshape.insert(1, 2)
        rhoxy = self.create_array(varshape)
        print('computing the rhoxy values ...')
        phys.compute_dyad_re(
            x, y, rho,
            x_avg, y_avg, rho_bar,
            self.c_mem_idx, self.c_area,
            rhoxy
        )
        if debug:
            self.IO.write_to('data', rhoxy[:,0,0,:,:], name='TAU_RE11',
                            attrs={
                                'long_name': 'horiz. Reynolds Stress Tensor 11',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=numfile
                            )
            self.IO.write_to('data', rhoxy[:,0,1,:,:], name='TAU_RE12',
                            attrs={
                                'long_name': 'horiz. Reynolds Stress Tensor 12',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=numfile
                            )
            self.IO.write_to('data', rhoxy[:,1,0,:,:], name='TAU_RE21',
                            attrs={
                                'long_name': 'horiz. Reynolds Stress Tensor 21',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=numfile
                            )
            self.IO.write_to('data', rhoxy[:,1,1,:,:], name='TAU_RE22',
                            attrs={
                                'long_name': 'horiz. Reynolds Stress Tensor 22',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=numfile
                            )
        return rhoxy

    def rhoxy_averages_scalar(self, xname, yname, sname, xavg, yavg, savg, numfile=0):
        x = self.IO.load_from('data', xname, numfile)
        y = self.IO.load_from('data', yname, numfile)
        if self.IO.isin('data', sname, numfile):
            s = self.IO.load_from('data', sname, numfile)
        elif self.IO.isin('newdata', sname, numfile):
            s = self.IO.load_from('newdata', sname, numfile)
        else:
            sys.exit('This was not supposed to happen.')
        x_avg = self.IO.load_from('newdata', xavg, numfile)
        y_avg = self.IO.load_from('newdata', yavg, numfile)
        s_avg = self.IO.load_from('newdata', savg)
        rho = self.IO.load_from('data', 'RHO', numfile)
        rho_bar = self.IO.load_from('newdata', 'RHO_BAR', numfile)
        x = self.nfo_merge(x)
        y = self.nfo_merge(y)
        s = self.nfo_merge(s)
        varshape = list(np.shape(x_avg))
        varshape.insert(1, 2)
        rhoxy = self.create_array(varshape)
        print('computing the rhoxy values ...')
        phys.compute_dyad_scalar(
            x, y, rho, s,
            x_avg, y_avg, rho_bar, s_avg,
            self.c_mem_idx, self.c_area,
            rhoxy
        )
        return rhoxy

    def rhoxy_averages_scalar_re(self, xname, yname, sname, xavg, yavg, savg, numfile=0):
        '''computing the rhoxy average, applying reynolds assumption'''
        x = self.IO.load_from('data', xname, numfile)
        y = self.IO.load_from('data', yname, numfile)
        if self.IO.isin('data', sname, numfile):
            s = self.IO.load_from('data', sname, numfile)
        elif self.IO.isin('newdata', sname, numfile):
            s = self.IO.load_from('newdata', sname, numfile)
        else:
            sys.exit('This was not supposed to happen.')
        x_avg = self.IO.load_from('newdata', xavg, numfile)
        y_avg = self.IO.load_from('newdata', yavg, numfile)
        s_avg = self.IO.load_from('newdata', savg)
        rho = self.IO.load_from('data', 'RHO', numfile)
        rho_bar = self.IO.load_from('newdata', 'RHO_BAR', numfile)
        x = self.nfo_merge(x)
        y = self.nfo_merge(y)
        s = self.nfo_merge(s)
        varshape = list(np.shape(x_avg))
        varshape.insert(1, 2)
        rhoxy = self.create_array(varshape)
        print('computing the rhoxy values ...')
        phys.compute_dyad_scalar_re(
            x, y, rho, s,
            x_avg, y_avg, rho_bar, s_avg,
            self.c_mem_idx, self.c_area,
            rhoxy
        )
        return rhoxy

    def turbulent_shear_prod(self, rhoxy, gradxy, filenum=0, outname='KIN_TRANS'):
        print('computing the turbulent shear production values ...')
        if not(self.IO.check_for([outname], filenum)[0]):
            varshape = list(np.shape(rhoxy))
            varshape.pop(1)
            varshape.pop(1)
            t_fric = self.create_array(varshape)
            phys.turb_fric(rhoxy, gradxy, t_fric)
            print('saving to file ...')
            self.IO.write_to('results', t_fric, name=outname,
                            attrs={
                                'long_name' : 'full kinetic energy transfer rates',
                                'units' : 'J/s',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
        return

    def turbulent_shear_prod_full(self, rhoxy, gradxy, filenum=0, outname='KITRA'):
        '''assumes the full shear tensor hand-in!
            computes both, the full and isotropic
            transfer rates.
        '''
        print('computing the turbulent shear production values ...')
        if not(self.IO.check_for([outname], filenum)[0]):
            varshape = list(np.shape(rhoxy))
            varshape.pop(1)
            varshape.pop(1)
            t_fric = self.create_array(varshape)
            phys.turb_fric(rhoxy, gradxy, t_fric)
            print('saving iso+aniso to file ...')
            self.IO.write_to('results', t_fric, name=outname,
                            attrs={
                                'long_name' : 'full kinetic energy transfer rates',
                                'units' : 'J/s',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
            del t_fric
            t_fric_iso = self.create_array(varshape)
            trace = 0.5* np.add(rhoxy[0,0,:], rhoxy[0,0,:])
            for i in range(2):
                rhoxy[i,i,:] = np.subtract(rhoxy[i,i,:], trace)
            phys.turb_fric(rhoxy, gradxy, t_fric_iso)
            self.IO.write_to('results', t_fric, name=outname+'_I',
                            attrs={
                                'long_name' : 'isotropic kinetic energy transfer rates',
                                'units' : 'J/s',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
        return

    def turbulent_molec_fric(self, filenum=0, outname='F_FRIC'):
        print('computing the turbulent shear production values ...')
        t_fric = self.IO.load_from('newdata', 'T_FRIC', filenum)
        sub_fric = self.hat_avg_scalar('TFRIC', filenum)
        t_fric = np.add(t_fric, sub_fric)
        print('saving to file ...')
        self.IO.write_to('results', t_fric, name=outname,
                        attrs={
                            'long_name' : 'sum molec and turbulent friction',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        return t_fric

    def turbulent_pres_flux(self, rhoxy, grad_ex, filenum=0, outname='T_PRES'):
        print('computing the turbulent pres flux...')
        varshape = list(np.shape(rhoxy))
        varshape.pop(1)
        t_pres = self.create_array(varshape)
        print('rhoxy: {}'.format(rhoxy[1, 1, :, 1]))
        print('grad_ex {}'.format(grad_ex[1, 1, :, 1]))
        phys.turb_pres(rhoxy, grad_ex, t_pres)
        print('saving to file ...')
        self.IO.write_to('results', t_pres, name=outname,
                        attrs={
                            'long_name' : 'turbulent_pres_flux',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        t_pres_kd = phys.turb_fric_erich(t_pres)
        self.IO.write_to('results', t_pres_kd, name=(outname+'_KD'),
                        attrs={
                            'long_name' : 'turbulent_pres_flux in kd',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        return t_pres

    def turbulent_heat_flux(self, rhoxy, grad_T, filenum=0, outname='INT_TRANS'):
        print('computing the turbulent heat flux...')
        varshape = list(np.shape(rhoxy))
        varshape.pop(1)
        t_heat = self.create_array(varshape)
        T = self.IO.load_from('newdata', 'T_HAT', filenum)
        print('rhoxy: {}'.format(rhoxy[1, 1, :, 1]))
        print('grad_T {}'.format(grad_T[1, 1, :, 1]))
        phys.turb_heat(rhoxy, grad_T, T, t_heat)
        print('saving to file ...')
        self.IO.write_to('results', t_heat, name=outname,
                        attrs={
                            'long_name' : 'Internal energy transfer rate',
                            'units' : 'J/s',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        return t_heat

    def turbulent_enstrophy_flux(self, rhoxw, zstar_grad, filenum=0, outname='T_NSTPY'):
        print('computing the turbulent enstrophy flux...')
        varshape = list(np.shape(rhoxw))
        varshape.pop(1)
        t_nstpy = self.create_array(varshape)
        phys.turb_enstrophy(rhoxw, zstar_grad, t_nstpy)
        print('saving to file ...')
        print('shape of t_nstpy: {}'.format(np.shape(t_nstpy)))
        self.IO.write_to('results', t_nstpy, name=outname,
                        attrs={
                            'long_name' : 'turbulent_enstrophy_flux',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
#       t_nstpy_kd = phys.turb_fric_erich(t_nstpy)
#       self.IO.write_to('results', t_nstpy_kd, name=(outname+'_KD'),
#                       attrs={
#                           'long_name' : 'turbulent_enstrophy_flux in kd',
#                           'coordinates': 'vlat vlon',
#                           '_FillValue' : float('nan'),
#                           'grid_type' : 'unstructured'
#                           }, filenum=filenum
#                       )
        return t_nstpy

    def friction_coefficient(self, gradxy, t_fric, numfile=0, outname='K_TURB'):
        rho = self.IO.load_from('data', 'RHO', numfile)
        varshape = np.shape(rho)
        rho = self.nfo_merge(rho)
        rho_bar = self.IO.load_from('newdata', 'RHO_BAR')

        print('computing the K values ...')
        varshape = list(np.shape(t_fric))
        kimag = self.create_array(varshape)
        phys.friction_coefficient(gradxy, rho_bar, t_fric, kimag)
        print('saving to file ...')
        self.IO.write_to('results', kimag, name='K_TURB',
                        attrs={
                            'long_name' : 'turbulent dissipation coefficiet',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=numfile
                        )
        return kimag

    def turb_fric_Kd(self, numfile, outname='T_ERICH'):
        # divide by RHO!!!
        print("I'm in main.turb_fric_Kd.")
        sys.exit("THIS CASE IS NOT HANDLED YET, and when I implemented it I didn't want to bother with it")
        t_fric = self.IO.load_from('results', 'T_FRIC', numfile)
        rho_bar = self.IO.load_from('newdata', 'RHO_BAR', numfile)
        t_fric = np.divide(t_fric, rho_bar)
        turberich = phys.turb_fric_erich(t_fric)
        self.IO.write_to('results', turberich, name=outname,
                         attrs={
                             'long_name' : 'turbulent_shear_erich',
                             'coordinates': 'vlat vlon',
                             '_FillValue' : float('nan'),
                             'grid_type' : 'unstructured',
                             'units' : 'K/d'
                             }, filenum=numfile
                         )

    def geostrophic_winds(self, numfile=0):
        UG_hat, VG_hat = self.bar_avg_2Dvec('UGEO', 'VGEO', numfile)
        self.IO.write_to('data', VG_hat, name='VG_HAT',
                        attrs={
                            'long_name': 'cg geostrophically balanced meridional wind',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=numfile
                        )
        self.IO.write_to('data', UG_hat, name='UG_HAT',
                        attrs={
                            'long_name': 'cg geostrophically balanced zonal wind',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=numfile
                        )

    def prepare_winds(self, filenum=0):
        debug = True
        print('processing file number {}'.format(filenum + 1))
        print('computing U and V hat')
        U_hat, V_hat = self.bar_avg_2Dvec('U', 'V', filenum)
        print('saving to file ...')
        self.IO.write_to('data', U_hat, name='U_HAT',
                        attrs={
                            'long_name': 'density weighted coarse zonal wind',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        self.IO.write_to('data', V_hat, name='V_HAT',
                        attrs={
                            'long_name': 'density weighted coarse meridional wind',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        if not(self.IO.isin('results', 'V_HAT', filenum)):
            self.IO.write_to('results', U_hat, name='U_HAT',
                            attrs={
                                'long_name': 'density weighted coarse zonal wind',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
            self.IO.write_to('results', V_hat, name='V_HAT',
                            attrs={
                                'long_name': 'density weighted coarse meridional wind',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )

        print('computing U and V grad')
        print('saving to file ...')
        UV_gradients = self.xy_hat_gradients(U_hat, V_hat)
        if debug:
            self.IO.write_to('data', UV_gradients[:,0,0,:,:], name='DUX',
                            attrs={
                                'long_name': 'density weighted coarse meridional wind',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
            self.IO.write_to('data',  UV_gradients[:,0,1,:,:], name='DVX',
                            attrs={
                                'long_name': 'density weighted coarse meridional wind',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
            self.IO.write_to('data',  UV_gradients[:,1,0,:,:], name='DUY',
                            attrs={
                                'long_name': 'density weighted coarse meridional wind',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )
            self.IO.write_to('data',  UV_gradients[:,1,1,:,:], name='DVY',
                            attrs={
                                'long_name': 'density weighted coarse meridional wind',
                                'coordinates': 'vlat vlon',
                                '_FillValue' : float('nan'),
                                'grid_type' : 'unstructured'
                                }, filenum=filenum
                            )

    def prepare_vorc(self, filenum=0):
        print('processing file number {}'.format(filenum + 1))
        print('computing vorc_hat')
        zstar_hat = self.hat_avg_scalar('ZSTAR', filenum)
        print('saving to file ...')
        self.IO.write_to('data', zstar_hat, name='ZSTAR_HAT',
                        attrs={
                            'long_name': 'density weighted mean star-vorticity',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        print('computing zstar_hat_grad')
        # to get out of precision range for computation
        zstar_hat = np.multiply(zstar_hat, 1000000)
        zstar_gradients = self.x_hat_gradients(zstar_hat)
        del zstar_hat
        zstar_gradients = np.divide(zstar_gradients, 1000000)
        self.IO.write_to('data',  zstar_gradients[:,0,:,:], name='ZSTARX',
                        attrs={
                            'long_name': 'zonal gradient of density weighted star Vorticity',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        self.IO.write_to('data',  zstar_gradients[:,1,:,:], name='ZSTARY',
                        attrs={
                            'long_name': 'meridional gradient of density weighted star Vorticity',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )

    def compute_T(self, filenum=0):
        rho = self.IO.load_from('data', 'RHO', filenum)
        varshape = list(np.shape(rho))
        rho = self.nfo_merge(rho)
        rho_bar = self.IO.load_from('newdata', 'RHO_BAR')

        theta = self.IO.load_from('data', 'THETA_V', filenum)
        exner = self.IO.load_from('data', 'EXNER', filenum)
        T = self.create_array(varshape)
        print('computing the true Temperature values ...')
        phys.thet_ex_to_T(theta, exner, T)
        self.IO.write_to('data', T, name='T',
                         attrs = {
                             'long_name': 'Temperature',
                             'coordinates': 'vlat vlon',
                             '_FillValue': float('nan'),
                             'grid_type' : 'unstructured'
                            }, filenum=filenum
                            )
        del theta, exner
        T = self.nfo_merge(T)
        T_hat = self.create_array(varshape)
        print('computing the density weighted coarse Temperature values ...')
        math.hat_scalar(
            T, rho, rho_bar,
            self.c_area, self.c_mem_idx,
            T_hat
        )
        self.IO.write_to('data', T_hat, name='T_HAT',
                        attrs={
                            'long_name': 'density weighted Temperature',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )

    def prepare_Tgrad(self, filenum=0):

        T_hat = self.IO.load_from('newdata', 'T_HAT')
        print('computing the coarse Temperature gradients')
        T_gradient = self.x_hat_gradients(T_hat)
        del T_hat
        self.IO.write_to('data', T_gradient[:,0,:, ], name='DTX',
                        attrs={
                            'long_name': 'x grad of density weighted Temperature ',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )

        self.IO.write_to('data', T_gradient[:,1,:, ], name='DTY',
                        attrs={
                            'long_name': 'y grad of density weighted Temperature ',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        del T_gradient

    def prepare_pres(self, filenum=0):
        rho = self.IO.load_from('data', 'RHO', filenum)
        varshape = list(np.shape(rho))
        rho = self.nfo_merge(rho)
        rho_bar = self.IO.load_from('newdata', 'RHO_BAR')

        theta = self.IO.load_from('data', 'THETA_V', filenum)
        theta = self.nfo_merge(theta)
        theta_hat = self.create_array(varshape)
        print('computing the density weighted coarse theta values ...')
        math.hat_scalar(
            theta, rho, rho_bar,
            self.c_area, self.c_mem_idx,
            theta_hat
        )
        self.IO.write_to('data', theta_hat, name='THETA_HAT',
                        attrs={
                            'long_name': 'density weighted theta_v',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        exner = self.IO.load_from('data', 'EXNER', filenum)
        exner = self.nfo_merge(exner)
        exner_hat = self.create_array(varshape)
        print('computing the density weighted coarse exner values ...')
        math.hat_scalar(
            exner, rho, rho_bar,
            self.c_area, self.c_mem_idx, exner_hat
        )
        del exner
        self.IO.write_to('data', exner_hat, name='EXNER_HAT',
                        attrs={
                            'long_name': 'density weighted exner pressure',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )

        print('computing the coarse exner gradients')
        exner_gradient = self.x_hat_gradients(exner_hat)
        del exner_hat
        self.IO.write_to('data', exner_gradient[:,0,:, ], name='DEXX',
                        attrs={
                            'long_name': 'x grad of density weighted exner pressure',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )

        self.IO.write_to('data', exner_gradient[:,1,:, ], name='DEXY',
                        attrs={
                            'long_name': 'y grad of density weighted exner pressure',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        del exner_gradient

    def load_grads(self, filenum=0):
        varshape = list(np.shape(self.IO.load_from('newdata', 'DUX', filenum)))
        varshape.insert(1, 2)
        varshape.insert(2, 2)
        UV_gradients = np.empty(varshape)
        UV_gradients[:, 0, 0, :, :] = self.IO.load_from('newdata', 'DUX', filenum)[:,]
        UV_gradients[:, 0, 1, :, :] = self.IO.load_from('newdata', 'DVX', filenum)[:,]
        UV_gradients[:, 1, 0, :, :] = self.IO.load_from('newdata', 'DUY', filenum)[:,]
        UV_gradients[:, 1, 1, :, :] = self.IO.load_from('newdata', 'DVY', filenum)[:,]
        return UV_gradients

    def load_ex_grad(self, filenum=0):
        varshape = list(np.shape(self.IO.load_from('newdata', 'DEXX', filenum)))
        varshape.insert(1, 2)
        EX_grad = np.empty(varshape)
        EX_grad[:, 0, :, :] = self.IO.load_from('newdata', 'DEXX', filenum)[:,]
        EX_grad[:, 1, :, :] = self.IO.load_from('newdata', 'DEXY', filenum)[:,]
        return EX_grad

    def load_T_grad(self, filenum=0):
        varshape = list(np.shape(self.IO.load_from('newdata', 'DTX', filenum)))
        varshape.insert(1, 2)
        T_grad = np.empty(varshape)
        T_grad[:, 0, :, :] = self.IO.load_from('newdata', 'DTX', filenum)[:,]
        T_grad[:, 1, :, :] = self.IO.load_from('newdata', 'DTY', filenum)[:,]
        return T_grad

    def load_zstar_grad(self, filenum=0):
        dummy = self.IO.load_from('newdata', 'ZSTARX', filenum)
        varshape = list(np.shape(dummy))
        varshape.insert(1, 2)
        zstar_grad = np.empty(varshape)
        zstar_grad[:, 0, :, :] = self.IO.load_from('newdata', 'ZSTARX', filenum)[:,]
        zstar_grad[:, 1, :, :] = self.IO.load_from('newdata', 'ZSTARY', filenum)[:,]
        return zstar_grad

    def coarse_smag_tens(self, UV_gradients, filenum=0):
        #if not(self.IO.check_for(['SMAG_FRIC'], filenum)[0]):
            T = self.IO.load_from('newdata', 'T_HAT', filenum)
            varshape = list(np.shape(UV_gradients))
            varshape.pop(1)
            varshape.pop(1)
            shear_tens = phys.shear_tensor_2D(UV_gradients)
            norm_shear = phys.norm2_2d_tensor(shear_tens)
            coarse_smag = self.create_array(varshape)
            #tensor product
            d_filt = math.radius_m(self.c_area[0])**2
            phys.smag_fric(d_filt, norm_shear, UV_gradients, T, coarse_smag)
            del d_filt, norm_shear, UV_gradients, T
            self.IO.write_to('results', coarse_smag, name='SMAG_FRIC',
                        attrs={
                            'long_name': 'Smagorinsky coarse fric',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )

class CoarseGrain(Operations):
    def __init__(self, path, grid, data, opt_name=None):
        self._ready = True
        self.IO = cio.IOcontroller(path, grid, data, opt_name=opt_name)
        self.prepare()

    def set_mp(self, switch, num_procs):
        Mp.toggle_switch(switch)
        Mp.change_num_procs(num_procs)

    def set_ready(self, state):
        self._ready = state

    def prepare(self):
        xlat = self.IO.load_from('grid', 'vlat')
        xlon = self.IO.load_from('grid', 'vlon')
        cell_area = self.IO.load_from('grid', 'dual_area_p')
        self.f_area = cell_area
        self.gridnfo = [[cell_area[i], [xlon[i], xlat[i]]] for i in range(len(xlat))]
        self.c_mem_idx = self.IO.load_from('grid', 'area_member_idx')
        self.c_area = self.IO.load_from('grid', 'coarse_area')
        self.int_idx = self.IO.load_from('grid', 'int_idx')
        self.int_dist = self.IO.load_from('grid', 'int_dist')
        self.g_coords = self.IO.load_from('grid', 'grad_coords')
        self.g_dist = self.IO.load_from('grid', 'grad_dist')

    def create_array(self, shape, dtype='float'):
        data_list = np.zeros(shape)
        shrd_list = shared_np_array(np.shape(data_list), dtype)
        shrd_list[:] = data_list[:]
        return shrd_list

    def nfo_merge(self, data):
        data_merge = [[data[i]] + nfo for i,nfo in enumerate(self.gridnfo)]
        return data_merge

    def exec_kine_transfer(self):
        print('computing turbulent transfer rates')
        print('***')
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}/{}'.format(filenum + 1, len(self.IO.datafiles)))
            if not(self.IO.check_for(['U_HAT', 'V_HAT', 'DVX', 'DUX', 'DVY', 'DVX'], filenum)[0]):
                print('preparing winds')
                self.prepare_winds(filenum)
                print('***')
            print('loading UVgrads now')
            UV_gradients = self.load_grads(filenum)
            print('preparing fluct dyad')
            rhouv_flucts = self.rhoxy_averages('U', 'V', 'U_HAT', 'V_HAT', filenum)
            print('computing the rates')
            self.turbulent_shear_prod(rhouv_flucts, UV_gradients, outname='KTRA')
            del rhouv_flucts
            print('done')

    def exec_kine_transfer_re(self):
        print('computing turbulent transfer rates, through Reynolds Stress only.')
        print('***')
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}/{}'.format(filenum + 1, len(self.IO.datafiles)))
            if not(self.IO.check_for(['U_HAT', 'V_HAT', 'DVX', 'DUX', 'DVY', 'DVX'], filenum)[0]):
                print('preparing winds')
                self.prepare_winds(filenum)
                print('***')
            print('loading UVgrads now')
            UV_gradients = self.load_grads(filenum)
            print('preparing Reynolds fluct dyad')
            rhouv_flucts = self.rhoxy_averages_re('U', 'V', 'U_HAT', 'V_HAT', filenum)
            print('computing the rates')
            self.turbulent_shear_prod_iso(rhouv_flucts, UV_gradients, outname='KTRA_RE')
            del rhouv_flucts
            print('done')

    def exec_heat_transfer(self):
        print('computing turbulent heat transfer rates')
        print('***')
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            if not(self.IO.check_for(['U_HAT', 'V_HAT', 'DVX', 'DUX', 'DVY', 'DVX'], filenum)[0]):
                print('preparing winds')
                self.prepare_winds(filenum)
                print('***')
            if not(self.IO.check_for(['T', 'T_HAT'], filenum)[0]):
                print('preparing Temperature')
                self.compute_T(filenum)
                print('***')
            if not(self.IO.check_for(['DTX', 'DTY'], filenum)[0]):
                print('preparing T Gradients')
                self.prepare_Tgrad(filenum)
                print('***')
            print('loading T Gradients')
            T_grad = self.load_T_grad(filenum)
            print('computing fluct dyad')
            rhouv_flucts = self.rhoxy_averages_scalar('U', 'V', 'T', 'U_HAT', 'V_HAT', 'T_HAT', filenum)
            print('computing the rates')
            turbfric = self.turbulent_heat_flux(rhouv_flucts, T_grad, outname='ITRA')
            del rhouv_flucts, T_grad, turbfric
            print('done')

    def exec_heat_transfer_re(self):
        print('computing turbulent heat transfer rates')
        print('***')
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            if not(self.IO.check_for(['U_HAT', 'V_HAT', 'DVX', 'DUX', 'DVY', 'DVX'], filenum)[0]):
                print('preparing winds')
                self.prepare_winds(filenum)
                print('***')
            if not(self.IO.check_for(['T', 'T_HAT'], filenum)[0]):
                print('preparing Temperature')
                self.compute_T(filenum)
                print('***')
            if not(self.IO.check_for(['DTX', 'DTY'], filenum)[0]):
                print('preparing T Gradients')
                self.prepare_Tgrad(filenum)
                print('***')
            print('loading T Gradients')
            T_grad = self.load_T_grad(filenum)
            print('computing fluct dyad')
            rhouv_flucts = self.rhoxy_averages_scalar_re('U', 'V', 'T', 'U_HAT', 'V_HAT', 'T_HAT', filenum)
            print('computing the rates')
            turbfric = self.turbulent_heat_flux(rhouv_flucts, T_grad, outname='ITRA_R')
            del rhouv_flucts, T_grad, turbfric
            print('done')

    def molecfric(self):
        for filenum, file in enumerate(self.IO.datafiles):
            self.turbulent_molec_fric(filenum)

    def heatflux(self):
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            if not(self.IO.check_for(['T', 'T_HAT'], filenum)[0]):
                self.compute_T(filenum)
            if not(self.IO.check_for(['DTX', 'DTY'], filenum)[0]):
                self.prepare_Tgrad(filenum)

            # load_T_grad norms by T as well!!
            T_grad = self.load_T_grad(filenum)

            rhouv_flucts = self.rhoxy_averages_scalar('U', 'V', 'T', 'U_HAT', 'V_HAT', 'T_HAT', filenum)
            print('computing turbulent friction...')
            self.turbulent_heat_flux(rhouv_flucts, T_grad)
            print('done.')

    def presflux(self):
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            if not(self.IO.check_for(['THETA_HAT', 'DEXX', 'DEXY'], filenum)[0]):
                self.prepare_pres(filenum)

            exner_grad = self.load_ex_grad(filenum)

            rhouv_flucts = self.rhoxy_averages_scalar('U', 'V', 'THETA_V', 'U_HAT', 'V_HAT', 'THETA_HAT', filenum)
            print('computing turbulent friction...')
            self.turbulent_pres_flux(rhouv_flucts, exner_grad)
            print('done.')

    def ageostrophic(self):
        #todo: the whole thing put the pieces together
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            if not(self.IO.check_for(['UGEO', 'VGEO'], filenum)[0]):
                sys.exit('geostrophic_winds not in file.. exiting')

            if not(self.IO.check_for(['V_HAT', 'DVX', 'DUX', 'DVY', 'DVX'], filenum)[0]):
                self.prepare_winds(filenum)

            if not(self.IO.check_for(['UG_HAT', 'VG_HAT'], filenum)[0]):
                print('computing the geostrophic winds...')
                self.geostrophic_winds(filenum)

            rhouv_flucts = self.rhoxy_averages('UGEO', 'VGEO', 'UG_HAT', 'VG_HAT', filenum)
            UV_gradients = self.load_grads(filenum)

            print('computing turbulent friction...')
            turbfric = self.turbulent_shear_prod(rhouv_flucts, UV_gradients, filenum, 'GEO_FRIC')
            del rhouv_flucts
            print('computing erichs way...')
            self.turb_fric_Kd(filenum, 'GEO_FR_KD')

            print('computing K...')
            self.friction_coefficient(UV_gradients, turbfric, filenum, 'GEO_K')
            del UV_gradients

    def enstrophyflux(self):
        '''computes the flux of enstrophy thorough the filter interface'''
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            if not(self.IO.check_for(['U_HAT', 'V_HAT'], filenum)[0]):
                self.prepare_winds(filenum)
            if not(self.IO.check_for(['ZSTARX', 'ZSTARY'], filenum)[0]):
                # written
                self.prepare_vorc(filenum)

            # check routine
            rhovw_flucts = self.rhoxy_averages_scalar('U', 'V', 'ZSTAR', 'U_HAT', 'V_HAT', 'ZSTAR_HAT', filenum)
            # written
            zstar_grad = self.load_zstar_grad(filenum)
            # write routine
            self.turbulent_enstrophy_flux(rhovw_flucts, zstar_grad)

    def r_r_comp(self):
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            winds = (self.IO.isin('data','U_HAT', filenum) and self.IO.isin('data','V_HAT', filenum))
            grads_v = (self.IO.isin('data','DVX', filenum) and self.IO.isin('data','DVY', filenum))
            grads_u = (self.IO.isin('data','DUX', filenum) and self.IO.isin('data','DUY', filenum))
            grads = grads_v and grads_u
            if not(grads and winds):
                self.prepare_winds(filenum)

            self.load_grads(filenum)

            self.rhoxy_averages('U', 'V', 'U_HAT', 'V_HAT', filenum)

    def smag(self):
        for numfile, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(numfile + 1))
            if not( self.IO.check_for('U_HAT', 'V_HAT', 'DVX', 'DVY', 'DUX', 'DUY', numfile)[0]):
                self.prepare_winds(numfile)
            coarse_UV_gradients = self.load_grads(numfile)
            #computation of smagorinski tensor
            self.coarse_smag_tens(coarse_UV_gradients, numfile)

    def testing(self):
        xname = 'U_HAT'
        yname = 'V_HAT'
        X = self.IO.load_from('data', xname)
        Y = self.IO.load_from('data', yname)
        X = self.nfo_merge(X)
        Y = self.nfo_merge(Y)
        g_idx = self.g_mem_idx[3,2]
        g_idx = g_idx[np.where(g_idx > -1)[0]]
        g_cr = self.g_coords_rads[3][2]
        x_set = [X[k] for k in g_idx]
        y_set = [Y[k] for k in g_idx]
        out = math.lst_sq_intp_vec(x_set, y_set, g_cr)[0]
        print('the values are {}'.format(out[3,3]))
        # compare lst sq with dist weighted avg.
        out_avg = math.dist_avg_vec(x_set, y_set, g_cr)[0]
        oo = out_avg - out
        print('the interpolated values are {}'.format(out_avg[3,3]))
        print(oo[3,3])
        print(np.average(oo))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coarse-Grain ICON Output files')
    parser.add_argument(
        'path_to_file',
        metavar = 'path',
        type = str,
        nargs = '+',
        help = 'a string specifying the path to the file'
    )
    parser.add_argument(
        'grid_file',
        metavar = 'gridf',
        type = str,
        nargs = '+',
        help='a string specifying the name of the gridfile'
    )
    parser.add_argument(
        'data_file',
        metavar = 'dataf',
        type = str,
        nargs = '+',
        help='a string specifying the name of the datafile'
    )
    parser.add_argument(
        'ptype',
        metavar = 'proc',
        type = str,
        nargs = '+',
        help='a string specifying procedure'
    )
    args = parser.parse_args()
    print(
        'coarse_graining the datafile {} using the gridfile {}.\n procedure {}'
        .format(
                path.join(args.path_to_file[0], args.data_file[0]),
                path.join(args.path_to_file[0], args.grid_file[0]),
                path.join(args.ptype[0])
                )
        )
    gmp.set_parallel_proc(True)
    gmp.set_num_procs(16)
    opt_name = 'BCW_Dual_{}'.format(args.ptype[0])
    cg = CoarseGrain(args.path_to_file[0], args.grid_file[0], args.data_file[0], opt_name=opt_name)
#    cg.gradient_debug()
    if args.ptype[0] == 'kine':
        cg.exec_kine_transfer()
    elif args.ptype[0] == 'kine_re':
        cg.exec_kine_transfer_re()
    elif args.ptype[0] == 'ine':
        cg.exec_heat_transfer()
    elif args.ptype[0] == 'ine_re':
        cg.exec_heat_transfer_re()
    else:
        print('invalid ptype')
