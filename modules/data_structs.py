#!/usr/bin/env python
# coding=utf-8
import xarray as xr
from mo_operations import operations


class coarse_graining():

    def __init__(self, p_data, p_grid):
        self.p_data = p_data
        self.p_grid = p_grid
        self.prepare()

    def query_vars(self):
        """prints variables in p_data on screen"""
        return None

    def prepare(self):
        ds = xr.open_dataset(self.p_data)
        self.ncells = ds.dims['ncells']
        ds.close()
        self.data = []
        for cell in range(self.ncells):
            self.cell_data.append(cell_data(cell, self.p_data, self.p))

    def communicate(self):
        """enables communication between cells"""






class cell_data(operations):
    """idea: have object cell, which contains all info and functions"""

    def __init__(self, cell, p_data, p_grid):

        self.cell_num = cell
        self.p_data = p_data
        self.p_grid = p_grid

    def prepare(self):
        ds = xr.open_dataset(self.p_grid)
        self.coarse_members = ds['area_member_idx'].values[self.cell,:]
        self.coarse_area = ds['coarse_area'].values[self.cell,:]
        ds.close()

    def get_neighbours(self):
        """asks for info from neighbors"""


