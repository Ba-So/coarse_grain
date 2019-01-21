import glob
import re
import os
import sys
from netCDF4 import Dataset
import numpy as np
#from debugdecorators import PrintArgs, PrintReturn

# Need this to read a file timestep wise, to minimize the amount of data
# carried around.
class IOcontroller(object):

    def __init__(self, experiment_path, grid=None, data=None):
        self.experiment_path = os.path.expanduser(experiment_path)
        if grid:
            self.gridfile = self.find(grid)
        if data:
            self.datafiles = self.find(data)

    def find(self, pattern):
        GridReg = re.compile(pattern, re.VERBOSE)
        grid_files = glob.glob(os.path.join(self.experiment_path, '*'))
        found = []
        for xfile in grid_files:
            xres = GridReg.search(xfile)
            if xres:
                found.append(xres.group(0))
        if len(found) == 0:
            print('error, no file found')
        else:
            print('multiple files found')
            print(found)
        return found

    def get_dimsize_from(self, where, dim, filenum = 0):
        if where == 'data':
            xfile = self.datafiles[filenum]
        elif where == 'grid':
            xfile = self.gridfile[0]
        else:
            sys.exit('invalid file: {}'.format(xfile))

        with Dataset(os.path.join(self.experiment_path, xfile), 'r') as xdata:
            dim_keys = [str(xkey) for xkey in xdata.dimensions.keys()]
            if dim in dim_keys:
                dim_val = xdata.dimensions[dim].size
            else:
                sys.exit('dimension {} not a dimension of file: {}'.format(dim, xfile))
        return dim_val

    def new_dimension(self, where, dimname, dimsize, filenum=0):
        if where == 'data':
            xfile = self.datafiles[filenum]
        elif where == 'grid':
            xfile = self.gridfile[0]
        else:
            sys.exit('invalid file: {}'.format(xfile))

        with Dataset(os.path.join(self.experiment_path, xfile), 'r+') as xdata:
            if not(dimname in xdata.dimensions.keys()):
                dim = xdata.createDimension(dimname, dimsize)
            else:
                pass

    def load_from(self, where, var, filenum = 0):
        if where == 'data':
            xfile = self.datafiles[filenum]
        elif where == 'grid':
            xfile = self.gridfile[0]
        else:
            sys.exit('invalid file: {}'.format(xfile))

        with Dataset(os.path.join(self.experiment_path, xfile), 'r') as xdata:
            var_keys = [str(xkey) for xkey in xdata.variables.keys()]
            if var in var_keys:
                data = xdata.variables[var][:]
            else:
                sys.exit('variable doesn\'t exist as variable in file: {}'.format(var))

        # switch rows, so ncell is in front
        data = np.moveaxis(data, -1, 0)

        return data

    def isin(self, where, what, filenum = 0):
        if where == 'data':
            xfile = self.datafiles[filenum]
        elif where == 'grid':
            xfile = self.gridfile[0]
        else:
            sys.exit('invalid file: {}'.format(xfile))

        with Dataset(os.path.join(self.experiment_path, xfile), 'r') as xdata:
            var_keys = [str(xkey) for xkey in xdata.variables.keys()]
            if var in var_keys:
                out=True
            else:
                out=False
        return out

    def write_to(self, where, data, filenum=0,
        name='data', dtype = 'f8', dims = ('time', 'lev', 'cell2',),
        attrs = {'long_name': 'no name'}):

        if where == 'data':
            xfile = self.datafiles[filenum]
            # switch rows, so ncell is back in the back
            data = np.moveaxis(data, 0, -1)
        elif where == 'grid':
            xfile = self.gridfile[0]
            data = np.moveaxis(data, 0, -1)
        # write to disk
        with Dataset(os.path.join(self.experiment_path, xfile), 'r+', format='NETCDF4_CLASSIC') as xdata:
            var_keys = [str(xkey) for xkey in xdata.variables.keys()]
            if name not in var_keys:
                print('writing new variable to file...')
                newvar = xdata.createVariable(name, dtype, dims)
                newvar.setncatts(attrs)
                newvar[:,] = data[:,]
            else:
                print('editing variable values in file...')
                print(name)
                xdata[name][:,] = data[:,]

if __name__ == '__main__':
    IO = IOcontroller('~/projects/icon/experiments/BCWcold', r'iconR\dB\d{2}-grid.nc', r'BCWcold_R2B07_slice_refined_4.nc')
    data = IO.load_from('grid', 'clat')
    print(data.shape)
    data = IO.load_from('data', 'V')
    print(data.shape)
    dim = IO.get_dimension_from('data', 'ncells')
    print(dim)

