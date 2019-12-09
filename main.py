#!/usr/bin/env python
# coding=utf-8
import numpy as np
import sys as sys
from os import path
import itertools
import argparse
from decorators.paralleldecorators import gmp, ParallelNpArray, shared_np_array
from decorators.debugdecorators import TimeThis, PrintArgs, PrintReturn
import modules.math_mod as math
import modules.phys_mod as phys
import modules.cio as cio
import modules.new_grad_mod as grad
#import modules.op_mod as Operations

class Operations(object):
    def func():
        #wrappers deciding how much information a part of the code recieves.
        #either whole or only slices.
        return func()

    def bar_avg_2Dvec(self, xname, yname, numfile=0):
        rho = self.IO.load_from('data', 'RHO', numfile)
        varshape = np.shape(rho)
        rho = self.nfo_merge(rho)
        if not self.IO.isin('data', 'RHO_BAR'):
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
        x = self.nfo_merge(x)
        y = self.nfo_merge(y)
        s = self.nfo_merge(s)
        varshape = list(np.shape(x_avg))
        varshape.insert(1, 2)
        rhoxy = self.create_array(varshape)
        print('computing the rhoxy values ...')
        phys.compute_dyad_scalar(
            x, y, rho, s,
            x_avg, y_avg, s_avg,
            self.c_mem_idx, self.c_area,
            rhoxy
        )
        return rhoxy

    def turbulent_shear_prod(self, rhoxy, gradxy, filenum=0, outname='T_FRIC'):
        print('computing the turbulent shear production values ...')
        varshape = list(np.shape(rhoxy))
        print('varshape {}'.format(varshape))
        varshape.pop(1)
        varshape.pop(1)
        t_fric = self.create_array(varshape)
        phys.turb_fric(rhoxy, gradxy, t_fric)
        print('saving to file ...')
        self.IO.write_to('results', t_fric, name=outname,
                        attrs={
                            'long_name' : 'turbulent_friction',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        return t_fric

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

    def turbulent_heat_flux(self, rhoxy, grad_T, filenum=0, outname='T_HEAT'):
        print('computing the turbulent heat flux...')
        varshape = list(np.shape(rhoxy))
        varshape.pop(1)
        t_heat = self.create_array(varshape)
        T = self.IO.load_from('newdata', 'T', filenum)
        print('rhoxy: {}'.format(rhoxy[1, 1, :, 1]))
        print('grad_T {}'.format(grad_T[1, 1, :, 1]))
        phys.turb_heat(rhoxy, grad_T, T, t_heat)
        print('saving to file ...')
        self.IO.write_to('results', t_heat, name=outname,
                        attrs={
                            'long_name' : 'turbulent_heat_flux',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        t_heat_kd = phys.turb_fric_erich(t_heat)
        self.IO.write_to('results', t_heat_kd, name=(outname+'_KD'),
                        attrs={
                            'long_name' : 'turbulent_heat_flux in kd',
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
        print('rhoxw: {}'.format(rhoxw[1, 1, :, 1]))
        print('zstar_grad {}'.format(zstar_grad[1, 1, :, 1]))
        phys.turb_enstrophy(rhoxw, zstar_grad, t_nstpy)
        print('saving to file ...')
        print('shape of t_nstpy: {}').format(np.shape(t_nstpy))
        self.IO.write_to('results', t_nstpy, name=outname,
                        attrs={
                            'long_name' : 'turbulent_enstrophy_flux',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
        t_nstpy_kd = phys.turb_fric_erich(t_nstpy)
        self.IO.write_to('results', t_nstpy_kd, name=(outname+'_KD'),
                        attrs={
                            'long_name' : 'turbulent_enstrophy_flux in kd',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
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
        if not(self.IO.isin('results', 'V_HAT')):
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

    def prepare_heat(self, filenum=0):
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

    def smag_tens(self, UV_grad, filenum=0):
        print('UV_grad shape: {}'.format(np.shape(UV_grad)))
        shape = list(np.shape(UV_grad))
        S_tens = np.zeros(shape)
        shape.pop(1)
        shape.pop(1)
        S_norm = np.zeros(shape)
        sumthree = (UV_grad[:, 0, 0, :, :] + UV_grad[:, 1, 1, :, :]) / 2
        for i in range(2):
            for j in range(2):
                if i != j:
                    S_tens[:, i, j, :, :] = (UV_grad[:, i, j, :, :] + UV_grad[:, j, i, :, :])/2
                else:
                    S_tens[:, i, j, :, :] = np.subtract(UV_grad[:, i, j, :, :], sumthree[:,:,:])

                S_norm = np.add(
                    S_norm,
                    np.sqrt(
                        np.multiply(
                            np.power(
                                S_tens[:, i, j, :, :],
                                2),
                            2)
                    ))

        print('s_norm shape: {}'.format(np.shape(S_norm)))
        for i in range(2):
            for j in range(2):
                S_tens[:,i,j,:,:] = np.multiply(S_tens[:,i,j,:,:], S_norm)
        return S_tens


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

    def exec_heat_fric(self):
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            winds = (self.IO.isin('newdata','U_HAT', filenum) and self.IO.isin('newdata','V_HAT', filenum))
            grads_v = (self.IO.isin('newdata','DVX', filenum) and self.IO.isin('newdata','DVY', filenum))
            grads_u = (self.IO.isin('newdata','DUX', filenum) and self.IO.isin('newdata','DUY', filenum))
            grads = grads_u and grads_v
            if not(grads and winds):
                self.prepare_winds(filenum)
            print('loading UVgrads now')
            UV_gradients = self.load_grads(filenum)
            print('computing turbulent friction')
            rhouv_flucts = self.rhoxy_averages('U', 'V', 'U_HAT', 'V_HAT', filenum)
            turbfric = self.turbulent_shear_prod(rhouv_flucts, UV_gradients)
            del rhouv_flucts, turbfric
            print('computing coarse smagorinsky coefficient')
            coarse_smag_tens = self.smag_tens(UV_gradients, filenum)
            #tensor product
            c_s = 0.2
            d_filt = math.radius_m(self.c_area[0])
            coarse_smag = -2*c_s*d_filt*np.einsum('ijklm,ijklm->ilm', coarse_smag_tens, UV_gradients)
            del coarse_smag_tens, UV_gradients
            self.IO.write_to('data', coarse_smag, name='Smag_Fric',
                        attrs={
                            'long_name': 'Smagorinsky coarse fric',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=filenum
                        )
            del coarse_smag
            print('computing turbulent heating')
            isT = self.IO.isin('newdata','T_HAT', filenum)
            grads_T = (self.IO.isin('newdata','DTX', filenum) and self.IO.isin('newdata','DTY', filenum))
            if not(grads_T and isT):
                self.prepare_heat(filenum)

            # load_T_grad norms by T as well!!
            T_grad = self.load_T_grad(filenum)

            rhouv_flucts = self.rhoxy_averages_scalar('U', 'V', 'T', 'U_HAT', 'V_HAT', 'T_HAT', filenum)
            turbfric = self.turbulent_heat_flux(rhouv_flucts, T_grad)
            del rhouv_flucts, T_grad, turbfric
            print('done')

    def execute(self):
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            winds = (self.IO.isin('newdata','U_HAT', filenum) and self.IO.isin('newdata','V_HAT', filenum))
            grads_v = (self.IO.isin('newdata','DVX', filenum) and self.IO.isin('newdata','DVY', filenum))
            grads_u = (self.IO.isin('newdata','DUX', filenum) and self.IO.isin('newdata','DUY', filenum))
            grads = grads_u and grads_v
            if not(grads and winds):
                self.prepare_winds(filenum)
            print('loading UVgrads now')
            UV_gradients = self.load_grads(filenum)

            rhouv_flucts = self.rhoxy_averages('U', 'V', 'U_HAT', 'V_HAT', filenum)
            print('computing turbulent friction...')
            turbfric = self.turbulent_shear_prod(rhouv_flucts, UV_gradients)

            del rhouv_flucts
            del turbfric
            del UV_gradients
            self.turb_fric_Kd(filenum)

    def molecfric(self):
        for filenum, file in enumerate(self.IO.datafiles):
            self.turbulent_molec_fric(filenum)

    def heatflux(self):
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            isT = self.IO.isin('newdata','T_HAT', filenum)
            grads_T = (self.IO.isin('newdata','DTX', filenum) and self.IO.isin('newdata','DTY', filenum))
            if not(grads_T and isT):
                self.prepare_heat(filenum)

            # load_T_grad norms by T as well!!
            T_grad = self.load_T_grad(filenum)

            rhouv_flucts = self.rhoxy_averages_scalar('U', 'V', 'T', 'U_HAT', 'V_HAT', 'T_HAT', filenum)
            print('computing turbulent friction...')
            turbfric = self.turbulent_heat_flux(rhouv_flucts, T_grad)
            print('done.')

    def presflux(self):
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            thet = self.IO.isin('newdata','THETA_HAT', filenum)
            grads_ex = (self.IO.isin('newdata','DEXX', filenum) and self.IO.isin('newdata','DEXY', filenum))
            if not(grads_ex and thet):
                self.prepare_pres(filenum)

            exner_grad = self.load_ex_grad(filenum)

            rhouv_flucts = self.rhoxy_averages_scalar('U', 'V', 'THETA_V', 'U_HAT', 'V_HAT', 'THETA_HAT', filenum)
            print('computing turbulent friction...')
            turbfric = self.turbulent_pres_flux(rhouv_flucts, exner_grad)
            print('done.')

    def ageostrophic(self):
        #todo: the whole thing put the pieces together
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            grads_v = (self.IO.isin('newdata','DVX', filenum) and self.IO.isin('newdata','DVY', filenum))
            grads_u = (self.IO.isin('newdata','DUX', filenum) and self.IO.isin('newdata','DUY', filenum))
            grads = grads_v and grads_u
            if not(grads):
                self.prepare_winds(filenum)

            winds_geo = (self.IO.isin('data','UGEO', filenum) and self.IO.isin('newdata','VGEO', filenum))
            if not winds_geo:
                sys.exit('geostrophic_winds not in file.. exiting')

            winds_avg = (self.IO.isin('data','UG_HAT', filenum) and self.IO.isin('newdata','VG_HAT', filenum))
            if not winds_avg:
                print('computing the geostrophic winds...')
                self.geostrophic_winds(filenum)

            rhouv_flucts = self.rhoxy_averages('UGEO', 'VGEO', 'UG_HAT', 'VG_HAT', filenum)
            UV_gradients = self.load_grads(filenum)

            print('computing turbulent friction...')
            turbfric = self.turbulent_shear_prod(rhouv_flucts, UV_gradients, filenum, 'GEO_FRIC')
            del rhouv_flucts
            print('computing erichs way...')
            turberich = self.turb_fric_Kd(filenum, 'GEO_FR_KD')

            print('computing K...')
            K = self.friction_coefficient(UV_gradients, turbfric, filenum, 'GEO_K')
            del UV_gradients

    def enstrophyflux(self):
        '''computes the flux of enstrophy thorough the filter interface'''
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            winds = (self.IO.isin('newdata','U_HAT', filenum) and self.IO.isin('newdata','V_HAT', filenum))
            grads_vorc = (self.IO.isin('newdata', 'ZSTARX', filenum) and self.IO.isin('newdata', 'ZSTARY', filenum))
            if not(winds):
                self.prepare_winds(filenum)
            if not(grads_vorc):
                # written
                self.prepare_vorc(filenum)

            # check routine
            rhovw_flucts = self.rhoxy_averages_scalar('U', 'V', 'ZSTAR', 'U_HAT', 'V_HAT', 'ZSTAR_HAT', filenum)
            # written
            zstar_grad = self.load_zstar_grad(filenum)
            # write routine
            ens_flux = self.turbulent_enstrophy_flux(rhovw_flucts, zstar_grad)

    def r_r_comp(self):
        for filenum, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(filenum + 1))
            winds = (self.IO.isin('data','U_HAT', filenum) and self.IO.isin('data','V_HAT', filenum))
            grads_v = (self.IO.isin('data','DVX', filenum) and self.IO.isin('data','DVY', filenum))
            grads_u = (self.IO.isin('data','DUX', filenum) and self.IO.isin('data','DUY', filenum))
            grads = grads_v and grads_u
            if not(grads and winds):
                self.prepare_winds(filenum)

            UV_gradients = self.load_grads(filenum)

            rhouv_flucts = self.rhoxy_averages('U', 'V', 'U_HAT', 'V_HAT', filenum)

    def smag(self):
        for numfile, file in enumerate(self.IO.datafiles):
            print('processing file number {}'.format(numfile + 1))
            winds = (self.IO.isin('data','U_HAT', numfile) and self.IO.isin('data','V_HAT', numfile))
            grads_v = (self.IO.isin('data','DVX', numfile) and self.IO.isin('data','DVY', numfile))
            grads_u = (self.IO.isin('data','DUX', numfile) and self.IO.isin('data','DUY', numfile))
            grads = grads_v and grads_u
            if not(grads and winds):
                self.prepare_winds(numfile)
            coarse_UV_gradients = self.load_grads(numfile)
            #computation of smagorinski tensor
            coarse_smag_tens = self.smag_tens(coarse_UV_gradients, numfile)
            #tensor product
            c_s = 0.2
            d_filt = math.radius_m(self.c_area[0])
            coarse_smag = -2*c_s*d_filt*np.einsum('ijklm,ijklm->ilm', coarse_smag_tens, coarse_UV_gradients)
            del coarse_UV_gradients, coarse_smag_tens
            self.IO.write_to('data', coarse_smag, name='Smag_Fric',
                        attrs={
                            'long_name': 'Smagorinsky coarse fric',
                            'coordinates': 'vlat vlon',
                            '_FillValue' : float('nan'),
                            'grid_type' : 'unstructured'
                            }, filenum=numfile
                        )

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
#    path = '/home1/kd031/projects/icon/experiments/BCW07'
#    gridfile = r'iconR\dB\d{2}-grid_refined_\d.nc'
#    datafile = r'BCW_R2B05L70.nc'
  # cg.testing()
   #cg.convert_to_erich()
  #  cg.molecfric()
  # cg.ageostrophic()
  # cg.smag()
  # cg.heatflux()
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
    args = parser.parse_args()
    print(
        'coarse_graining the datafile {} using the gridfile {}.'
    ).format(path.join(args.path_to_file[0], args.data_file[0]), path.join(args.path_to_file[0], args.grid_file[0]))
    gmp.set_parallel_proc(True)
    gmp.set_num_procs(16)
    cg = CoarseGrain(args.path_to_file[0], args.grid_file[0], args.data_file[0])
    cg.enstrophyflux()
    cg.exec_heat_fric()
  # cg.testing()
   #cg.convert_to_erich()
#    cg.execute()
  #  cg.molecfric()
  # cg.ageostrophic()
#    cg.smag()
 #   cg.heatflux()
