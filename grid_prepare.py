#!/usr/bin/env python
# coding=utf-8
import argparse
from shutil import copyfile
from os import path
import sys
import itertools
import numpy as np
import xarray as xr
from decorators.paralleldecorators import gmp, ParallelNpArray, shared_np_array
from decorators.debugdecorators import TimeThis, PrintArgs, PrintReturn
from decorators.functiondecorators import requires
import modules.cio as cio
import modules.math_mod as math
import modules.new_grad_mod as grad

class Grid_Preparator(object):
    def __init__(self, path, gridn, num_rings):
        self.num_rings = num_rings
        self.xfile = cio.IOcontroller(path, grid=gridn)

    def create_array(self, shape, dtype='float'):
        data_list = np.zeros(shape)
        shrd_list = shared_np_array(np.shape(data_list), dtype)
        shrd_list[:] = data_list[:]
        return shrd_list

    def find_pentagons(self):
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        num_edges = np.array([6 for i in range(0, ncells)], dtype=int)
        cni = self.xfile.load_from('grid', 'vertices_of_vertex')

        zeroes = np.argwhere(cni == 0)
        #0-ncells 1-ne

        for i in zeroes:
            num_edges[i[0]] = 5
            if i[1] != 5:
                cni[i[0], i[1]] = cni[i[0], 5]
            else:
                cni[i[0], 5] = cni[i[0], 4]

        cni[:] -= 1

        self.xfile.write_to(
            'grid', cni, dtype = 'i4', name='vertices_of_vertex',
            dims=['ne', 'vertex']
        )
        self.xfile.write_to(
            'grid', num_edges, dtype = 'i4', name='num_edges',
            dims=['vertex']
        )

    def compute_hex_area_members(self):
        # number of hexagons in coarse area
        num_hex = math.num_hex_from_rings(self.num_rings)

        ncells = self.xfile.get_dimsize_from('grid', 'vertex')

        a_nei_idx = self.create_array([ncells, num_hex], dtype='int')
        helper = np.empty([ncells, num_hex])
        helper.fill(-1) #masking
        a_nei_idx[:] = helper[:]

        num_edges = self.xfile.load_from('grid', 'num_edges')
        cell_neighbour_idx = self.xfile.load_from('grid', 'vertices_of_vertex')
        cell_index = np.arange(0, ncells)

        define_hex_area(cell_neighbour_idx, num_hex, num_edges, cell_index, a_nei_idx)
        self.xfile.new_dimension('grid', 'num_hex', num_hex)
        print('writing')
        self.xfile.write_to(
            'grid', a_nei_idx, dtype='i4', name='area_member_idx',
            dims=['num_hex','vertex']
        )

    def compute_coarse_area(self):
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        co_ar = self.create_array([ncells])
        co_ar[:] = np.array([0.0 for i in range(0, ncells)])
        cell_area = self.xfile.load_from('grid', 'dual_area_p')
        a_nei_idx = self.xfile.load_from('grid', 'area_member_idx')

        coarse_area(cell_area, a_nei_idx, co_ar)

        self.xfile.write_to('grid', co_ar, name='coarse_area', dims=['vertex'])

    def compute_coarse_gradient_nfo(self):
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        cell_area = self.xfile.load_from('grid', 'dual_area_p')
        coarse_area = self.xfile.load_from('grid', 'coarse_area')

        lon = self.xfile.load_from('grid', 'vlon')
        lat = self.xfile.load_from('grid', 'vlat')
        coords = [[loni, lati] for loni, lati in itertools.izip(lon, lat)]
        times_rad = 2
        max_members = 2 + 6 * times_rad/2 * (times_rad/2 + 1) / 2
        grad_coords = self.create_array([ncells, 4, 2])
        grad_dist = self.create_array([ncells])
        grad_dist[:] = 0.0
        int_index = self.create_array([ncells, 4, max_members], dtype='int')
        int_index[:] = -1
        int_dist = self.create_array([ncells, 4, max_members, 2])
        int_dist[:] = 10000.0
        cell_idx = np.arange(ncells)

        grad.compute_coarse_gradient_nfo(
            coords, coarse_area, cell_area, cell_idx, #input
            grad_coords, grad_dist, int_index, int_dist #output
        )
        print(np.min(int_dist))
        print(np.max(int_dist))

        self.xfile.new_dimension('grid', 'gradnum', 4)
        self.xfile.new_dimension('grid', 'lonlat', 2)
        self.xfile.write_to(
            'grid', grad_coords, name='grad_coords',
            dims=['gradnum', 'lonlat', 'vertex'],
            attrs={'long_name': 'gradient coordinates'}
        )
        self.xfile.write_to(
            'grid', grad_dist, name='grad_dist',
            dims=['vertex'],
            attrs={'long_name': 'gradient distances'}
        )
        self.xfile.new_dimension('grid', 'maxmem', max_members)
        self.xfile.write_to(
            'grid', int_index, dtype='i4', name='int_idx',
            dims=['gradnum', 'maxmem', 'vertex'],
            attrs={'long_name': 'interpolation indices'}
        )
        self.xfile.write_to(
            'grid', int_dist, name='int_dist',
            dims=['gradnum', 'maxmem', 'lonlat', 'vertex'],
            attrs={'long_name': 'interpolation distances'}
        )

    def compute_fine_gradient_nfo(self):
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        cell_area = self.xfile.load_from('grid', 'dual_area_p')

        lon = self.xfile.load_from('grid', 'vlon')
        lat = self.xfile.load_from('grid', 'vlat')
        coords = [[loni, lati] for loni, lati in itertools.izip(lon, lat)]
        times_rad = 2
        max_members = 2 + 6 * times_rad/2 * (times_rad/2 + 1) / 2
        grad_coords = self.create_array([ncells, 4, 2])
        grad_dist = self.create_array([ncells])
        grad_dist[:] = 0.0
        int_index = self.create_array([ncells, 4, max_members], dtype='int')
        int_index[:] = -1
        int_dist = self.create_array([ncells, 4, max_members, 2])
        int_dist[:] = 10000.0
        cell_idx = np.arange(ncells)

        grad.compute_fine_gradient_nfo(
            coords, cell_area, cell_idx, #input
            grad_coords, grad_dist, int_index, int_dist #output
        )
        print(np.min(int_dist))
        print(np.max(int_dist))

        self.xfile.new_dimension('grid', 'gradnum', 4)
        self.xfile.new_dimension('grid', 'lonlat', 2)
        self.xfile.write_to(
            'grid', grad_coords, name='f_grad_coords',
            dims=['gradnum', 'lonlat', 'vertex'],
            attrs={'long_name': 'gradient coordinates'}
        )
        self.xfile.write_to(
            'grid', grad_dist, name='f_grad_dist',
            dims=['vertex'],
            attrs={'long_name': 'gradient distances'}
        )
        self.xfile.new_dimension('grid', 'maxmem', max_members)
        self.xfile.write_to(
            'grid', int_index, dtype='i4', name='f_int_idx',
            dims=['gradnum', 'maxmem', 'vertex'],
            attrs={'long_name': 'interpolation indices'}
        )
        self.xfile.write_to(
            'grid', int_dist, name='f_int_dist',
            dims=['gradnum', 'maxmem', 'lonlat', 'vertex'],
            attrs={'long_name': 'interpolation distances'}
        )

    def execute(self):
        print('locating pentagons...')
        self.find_pentagons()
        print('computing coarse area member indices...')
        self.compute_hex_area_members()
        print('computing coarse area...')
        self.compute_coarse_area()
        # print('preparing local gradient computation...')
        # self.compute_l_gradient_nfo()
        print('preparing coarse gradient computation...')
        self.compute_coarse_gradient_nfo()
        print('preparing fine gradient computation...')
        self.compute_fine_gradient_nfo()

        #write stuff


@TimeThis
@requires({
    'full' : ['cell_neighbour_idx', 'num_hex', 'num_edges'],
    'slice' : ['cell_index', 'a_nei_idx']
})
@ParallelNpArray(gmp)
def define_hex_area(cell_neighbour_idx, num_hex, num_edges, cell_index, a_nei_idx):

    for idx, ani in itertools.izip(cell_index, a_nei_idx):
        jh = 0
        jh_c = 1
        ani[0] = idx
        check_num_hex = num_hex
        while jh_c < check_num_hex:
            idx_n = int(ani[jh])

            if (num_edges[idx_n] == 5):
                check_num_hex -= 1
                if jh_c >= check_num_hex:
                    break

            for jn in range(0, num_edges[idx_n]):
                idx_c = cell_neighbour_idx[idx_n, jn]

                if idx_c in ani:
                    pass
                elif jh_c < check_num_hex:
                    ani[jh_c] = idx_c
                    jh_c += 1
                else:
                    break
                    print('define_hex_area: error jh_c too large')
            jh += 1


@TimeThis
@requires({
    'full' : ['cell_area'],
    'slice' : ['area_member_idx', 'coarse_area']
})
@ParallelNpArray(gmp)
def coarse_area(cell_area, area_member_idx, coarse_area):
    ca = []

    for ami in area_member_idx:
        areas = cell_area[np.extract(np.greater(ami, -1), ami)]
        ca.append(np.sum(areas))

    coarse_area[:] = ca


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare ICON grid-files for Coarse-Graining.')
    parser.add_argument(
        'path_to_file',
        metavar = 'path',
        type = str,
        nargs = '+',
        help='a string specifying the path to the gridfile'
    )
    parser.add_argument(
        'grid_name',
        metavar = 'gridn',
        type = str,
        nargs = '+',
        help='a string specifying the name of the gridfile'
    )
    parser.add_argument(
        'num_rings',
        metavar = 'num_rings',
        type = int,
        nargs = '+',
        help = 'an integer specifying the number of rings to be coarse-grained over.'
    )
    args = parser.parse_args()
    print(
        'preparing the gridfile {} for coarse grainig over {} rings.'
    ).format(path.join(args.path_to_file[0], args.grid_name[0]), args.num_rings[0])
    new_name = args.grid_name[0][:-3] +  '_refined_{}.nc'.format(args.num_rings[0])
    copyfile(
        path.join(args.path_to_file[0], args.grid_name[0]),
        path.join(args.path_to_file[0], new_name)
    )
    gmp.set_parallel_proc(True)
    gmp.set_num_procs(12)
    print(new_name)
    GP = Grid_Preparator(args.path_to_file[0], new_name, args.num_rings[0])
  #  GP.compute_l_gradient_nfo()
    GP.execute()
