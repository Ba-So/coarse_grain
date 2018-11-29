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
    # TODO write script

    def __init__(self, experiment_path, grid, data):
        self.experiment_path = os.path.expanduser(experiment_path)
        self.gridfile = self.find(grid)
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

    def load_from(self, where, var, filenum = 0):
        if where == 'data':
            xfile = self.datafiles[filenum]
        elif where == 'grid':
            xfile = self.gridfile[0]
        else:
            sys.exit('invalid file')

        with Dataset(os.path.join(self.experiment_path, xfile), 'r') as xdata:
            var_keys = [str(xkey) for xkey in xdata.variables.keys()]
            if var in var_keys:
                data = xdata.variables[var][:]
            else:
                sys.exit('variable doesn\'t exist as variable in gridfile')

        if where == 'data':
            # switch rows, so ncell is in front
            data = np.moveaxis(data, -1, 0)

        return data

    def write_to(
        self, where, data, filenum=0,
        name='data', dtype = 'f8', dims = ('time', 'lev', 'cell2'),
        attrs = {'long name': 'no name'}
    ):
        if where == 'data':
            xfile = self.datafiles[filenum]
            xdata = np.moveaxis(data, 0, -1)
        elif where == 'grid':
            xfile = self.gridfile[0]
        # switch rows, so ncell is back in the back
        # write to disk
        with Dataset(os.path.join(self.experiment_path, xfile), 'r') as xdata:
            newvar = xdata.createVariable(name, dtype, dims)
            newvar.setncattrs(attrs)
            xdata[name][:] = data
            xdata.close()

if __name__ == '__main__':
    IO = IOcontroller('~/projects/icon/experiments/BCW_coarse', r'iconR\dB\d{2}-grid.nc', r'BCWcg_R\dB\d{2}_U.nc')
    data = IO.load_from('grid', 'clat')
    print(data.shape)
    data = IO.load_from('data', 'U')
    print(data.shape)

