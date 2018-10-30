import xarray as xr
import data_op as dop
import math_op as mop
import itertools
import glob
import os
from debugdecorators import PrintArgs, PrintReturn

# Need this to read a file timestep wise, to minimize the amount of data
# carried around.
def glob_files(path, lookfor):
    return [
        n for n in
        glob.glob(path + lookfor) if os.path.isfile(n)
        ]

def read_netcdfs(path):
    """
        reads data from .nc files:
        path: Path to file
        qty : qty to be read
        """
    ds = xr.open_dataset(path)

    return ds

def rename_dims_vars(ds, dims = None):
    '''
        rename dimensions or variables of xarray Dataset
        input:
            ds: xr.dataset,
            dims: dict of {'old_name' : 'newname'}
        output:
            ds_o : xr.dataset with renamed dimesions
        '''
    if not dims:
        dims={
            'cell2' : 'ncells',
            'vlon' : 'clon',
            'vlat' : 'clat'
              }
    for key, item in dims.iteritems():
        if key in ds or key in ds.dims:
            ds = ds.rename({key : item})
        else:
            print('{} not found in dataset').format(key)

    return ds


def rename_attr(ds, which, attr):
    '''
        rename attributes of xarray Dataset
        input:
            ds: xr.dataset,
            dims: dict of {'old_name' : 'newname'}
        output:
            ds_o : xr.dataset with renamed dimesions
        '''
    for key, item in attr.iteritems():
        if key in ds:
            ds[key].attrs[which] = item

    return ds

def extract_variables(ds, variables):
    '''
        create new dataset containing only specific variables
        input:
            ds : xr.dataset,
            variables : array of strings
        output:
            ds_o : xr.dataset
        '''
    #variables = ['U', 'V']
    ds_o = ds[variables]

    return ds_o

def write_netcdf(path, ds):
    """writes Information to .nc file
    """
    print ('writing to {}').format(path)
    ds.to_netcdf(path, mode='w')


