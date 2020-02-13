import glob
import re
import os
import sys
from netCDF4 import Dataset
import numpy as np
# from debugdecorators import PrintArgs, PrintReturn

# Need this to read a file timestep wise, to minimize the amount of data
# carried around.

class IOcontroller(object):

    def __init__(self, experiment_path, grid=None, data=None, out=True):
        self.experiment_path = os.path.expanduser(experiment_path)
        if grid:
            self.gridfile = self.find(grid)
        if data:
            self.datafiles = self.find(data)
            print('datafiles: {}'.format(self.datafiles))
            if out:
                if not os.path.exists(os.path.join(self.experiment_path,'filtered')):
                    os.makedirs(os.path.join(self.experiment_path, 'filtered'))
                self.outfiles = [os.path.join('filtered',data[:-3] + '_helper.nc') for data in self.datafiles]
                self.create_outfiles()
                self.resultfiles = [os.path.join('filtered',data[:-3] + '_results.nc') for data in self.datafiles]
                self.create_resfiles()

    def count_existing(self, pattern, where=None):
        if not where:
            where = self.experiment_path
        else:
            where = os.path.join(self.experiment_path, where)
        GridReg = re.compile(".*"+pattern+".*", re.VERBOSE)
        found_files = glob.glob(os.path.join(where, '*'))
        found = 0
        for xfile in found_files:
            xres = GridReg.search(xfile)
            if xres:
                found += 1
        return found

    def find(self, pattern, where=None):
        if not where:
            where = self.experiment_path
        else:
            where = os.path.join(self.experiment_path, where)
        GridReg = re.compile(".*"+pattern+".*", re.VERBOSE)
        grid_files = glob.glob(os.path.join(where, '*'))
        found = []
        for xfile in grid_files:
            xres = GridReg.search(xfile)
            if xres:
                found.append(xres.group(0))
        if len(found) == 0:
            print('error, no file found')
            sys.exit('no file found: {}'.format(pattern))
        else:
            print('multiple files found')
            print(found)
        found = [os.path.basename(finding) for finding in found]
        return found

    def get_filepath(self, where, filenum = 0):
        if where == 'data':
            xfile = self.datafiles[filenum]
        elif where == 'newdata':
            xfile = self.outfiles[filenum]
        elif where == 'grid':
            xfile = self.gridfile[0]
        elif where == 'results':
            xfile = self.resultfiles[filenum]
        else:
            sys.exit('invalid where: {}'.format(where))
        return xfile

    def get_dimsize_from(self, where, dim, filenum = 0):
        xfile = self.get_filepath(where, filenum)

        with Dataset(os.path.join(self.experiment_path, xfile), 'r') as xdata:
            dim_keys = [str(xkey) for xkey in xdata.dimensions.keys()]
            if dim in dim_keys:
                dim_val = xdata.dimensions[dim].size
            else:
                sys.exit('dimension {} not a dimension of file: {}'.format(dim, xfile))
        return dim_val

    def new_dimension(self, where, dimname, dimsize, filenum=0):
        xfile = self.get_filepath(where, filenum)

        with Dataset(os.path.join(self.experiment_path, xfile), 'r+') as xdata:
            if not(dimname in xdata.dimensions.keys()):
                dim = xdata.createDimension(dimname, dimsize)
            else:
                pass

    def load_from(self, where, var, filenum = 0):
        xfile = self.get_filepath(where, filenum)

        with Dataset(os.path.join(self.experiment_path, xfile), 'r') as xdata:
            var_keys = [str(xkey) for xkey in xdata.variables.keys()]
            if var in var_keys:
                data = xdata.variables[var][:]
            else:
                sys.exit('variable doesn\'t exist as variable in file: {}'.format(var))

        # switch rows, so ncell is in front
        data = np.moveaxis(data, -1, 0)

        return data

    def load(self, xfile, var, time=None, lev=None):
        if not xfile in self.datafiles:
            sys.exit('xfile not in my filelist: {}'.format(xfile))
        with Dataset(os.path.join(self.experiment_path, xfile), 'r') as xdata:
            var_keys = [str(xkey) for xkey in xdata.variables.keys()]
            if var in var_keys:
                data = xdata.variables[var][:]
            else:
                sys.exit('variable doesn\'t exist as variable in file: {}'.format(var))
        # switch rows, so ncell is in front
        data = np.moveaxis(data, -1, 0)
        # in case only specific times or level are to be investigated
        if lev and time:
            data = data[:, time, lev]
        elif lev:
            data = data[:, :, lev]
        elif time:
            data = data[:, time, :]
        return data

    def load_all_data(self, var):
        data = self.load_from('data', var, 0)
        if len(self.datafiles) > 1 :
            for i, file in enumerate(self.datafiles):
                data = np.concatenate((data, self.load_from('data', var, i)), axis = 1)
        return data

    def check_for(self, what, filenum = 0):
        wheres = ['results', 'newdata', 'data', 'grid']
        foundit = {i:[False, ''] for i in what}
        allfound = True
        for who in what:
            found = False
            for where in wheres:
                found = self.isin(where, who, filenum)
                if found:
                    foundit[who] = [found, where]
        for key, item in foundit.items():
            if not item[0]:
                allfound = False

        print(foundit, allfound)

        return allfound, foundit

    def isin(self, where, what, filenum = 0):
        xfile = self.get_filepath(where, filenum)

        with Dataset(os.path.join(self.experiment_path, xfile), 'r') as xdata:
            var_keys = [str(xkey) for xkey in xdata.variables.keys()]
            if what in var_keys:
                out=True
            else:
                out=False
        return out

    def write_to(self, where, data, filenum=0,
        name='data', dtype = 'f8', dims = ('time', 'lev', 'cell2',),
        attrs = {'long_name': 'no name'}):

        xfile = self.get_filepath(where, filenum)
        #if not(where == 'grid'):
            # switch rows, so ncell is back in the back
        data = np.moveaxis(data, 0, -1)
        if where == 'data':
            # never write to original file
            xfile = self.get_filepath('newdata', filenum)
        # write to disk
        path = os.path.join(self.experiment_path, xfile)
        with Dataset(path, 'r+') as xdata:
            var_keys = [str(xkey) for xkey in xdata.variables.keys()]
            if name not in var_keys:
                print('writing new variable <{}> to file <{}>'.format(name, path))
                newvar = xdata.createVariable(name, dtype, dims, zlib=True, complevel=9)
                newvar.setncatts(attrs)
                newvar[:,] = data[:,]
            else:
                print('editing variable <{}> values in file <{}>'.format(name, path))
                xdata[name][:,] = data[:,]
        print('done.')

    def create_outfiles(self):
        for i, datafile in enumerate(self.datafiles):
            if not os.path.isfile(os.path.join(self.experiment_path, self.outfiles[i])):
                print('calling create outfile')
                with Dataset(os.path.join(self.experiment_path, datafile)) as (src
                    ), Dataset(os.path.join(self.experiment_path, self.outfiles[i]), "w") as (dst
                    ):
                    print('{}'.format(self.outfiles[i]))
                    print('files open')
                    # copy global attributes all at once via dictionary
                    dst.setncatts(src.__dict__)
                    # copy dimensions
                    for name, dimension in src.dimensions.items():
                        dst.createDimension(
                            name, (len(dimension) if not dimension.isunlimited() else None)
                        )
                    # copy basic neccessary variables over:
                    include = [
                        'THETA_V', 'ZF3', 'ZSTAR', 'EXNER', # neccessary variables for plotting
                        'hyai', 'hyam', 'hybi', 'hybm',
                        'lev', 'lev_2', 'time',
                        'vlat', 'vlat_vertices',
                        'vlon', 'vlon_vertices'
                    ]
                    for name, variable in src.variables.items():
                        if name in include:
                            dst.createVariable(
                                name, variable.datatype,
                                variable.dimensions, zlib=True,
                                complevel=9
                            )
                            # copy variable attributes all at once via dict:
                            dst[name].setncatts(src[name].__dict__)
                            dst[name][:] = src[name][:]

    def create_resfiles(self):
        for i, resultfile in enumerate(self.resultfiles):
            if not os.path.isfile(os.path.join(self.experiment_path, resultfile)):
                print('calling create resultfile')
                with Dataset(os.path.join(self.experiment_path, self.datafiles[i])) as (src
                    ), Dataset(os.path.join(self.experiment_path, resultfile), "w") as (dst
                    ):
                    print('files open')
                    # copy global attributes all at once via dictionary
                    dst.setncatts(src.__dict__)
                    # copy dimensions
                    for name, dimension in src.dimensions.items():
                        dst.createDimension(
                            name, (len(dimension) if not dimension.isunlimited() else None)
                        )
                    # copy basic neccessary variables over:
                    include = [
                        'THETA_V', 'ZF3', 'ZSTAR', 'EXNER', # neccessary variables for plotting
                        'hyai', 'hyam', 'hybi', 'hybm',
                        'lev', 'lev_2', 'time',
                        'vlat', 'vlat_vertices',
                        'vlon', 'vlon_vertices'
                    ]
                    for name, variable in src.variables.items():
                        if name in include:
                            dst.createVariable(
                                name, variable.datatype,
                                variable.dimensions, zlib=True,
                                complevel=9
                            )
                            # copy variable attributes all at once via dict:
                            dst[name].setncatts(src[name].__dict__)
                            dst[name][:] = src[name][:]

if __name__ == '__main__':
    IO = IOcontroller('~/projects/icon/experiments/BCW05', r'iconR\dB\d{2}-grid.nc', r'BCW_R2B05L70.nc')
    data = IO.load_from('grid', 'vlat')
    print(data.shape)
    data = IO.load_from('data', 'V')
    print(data.shape)

