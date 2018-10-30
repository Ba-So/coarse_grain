#!/usr/bin/env python
# coding=utf-8
from modules.paralleldecorators import Mp, ParallelNpArray, shared_np_array
from modules.debugdecorators import TimeThis, PrintArgs, PrintReturn
from modules.cio import glob_files
import xarray as xr

mp = Mp(False, 2)

class coarse_grain(object):

    def __init__(self, path, experiment_name):
        self._ready = True
        self._path = path + experiment_name + '/'
        self._experiment_name = experiment_name
        self.find_files(self._path, self._experiment_name)
        self.load_necessary(self._datafile, self._gridfile)

    def set_mp(self, switch, num_procs):
        Mp.toggle_switch(switch)
        Mp.change_num_procs(num_procs)

    def set_ready(self, state):
        self._ready = state

    def find_files(self, path, ex_name):
        #look for data output
        look_for = ex_name + "*_slice.nc"
        files = glob_files(path, look_for)
        setattr(self, '_datafile', files[0])
        look_for = "icon*-grid_refined_*.nc"
        files = glob_files(path, look_for)
        setattr(self, '_gridfile', files[0])

    def load_necessary(self, datafile, gridfile):
        '''to guarante that all the basic necesities are met'''
        #extract necessary variables from datafile
        nec_data = [
            'RHO',
            'vlat',
            'vlon',
            ]
        path = datafile
        self.load_data(path, nec_data)
        #extract necessary variables from gridfile
        nec_grid = [
            'coords',
            'member_idx',
            'member_rad',
            'area_member_idx',
            'cell_area',
            'coarse_area',
        ]
        path = gridfile
        self.load_data(path, nec_grid)

    def load_data(self, path, list):
        with xr.open_dataset(path) as ds:
            for variable in list:
                try:
                    setattr(self, variable, ds[variable].values)
                except KeyError:
                    print('{} not in file!'.format(variable))
                    self.set_ready(False)


if __name__ == '__main__':

    path = '/home1/kd031/projects/icon/experiments/'
    experiment_name = 'BCWcold'
    cg = coarse_grain(path, experiment_name)
    print cg._ready


