#!/usr/bin/env python
# coding=utf-8
import argparse
import itertools
import numpy as np
import xarray as xr
from decorators.paralleldecorators import gmp, ParallelNpArray, shared_np_array
from decorators.depugdecorators import TimeThis, PrintArgs, PrintReturn
from decorators.functiondecorators import requires
import modules.cio as cio
import modules.math_mod as math

class Grid_Preparator(object):
    def __init__(self, path, gridn, num_rings):
        self.num_rings = num_rings
        self.xfile = cio.IOcontroller(path, grid)

    def create_array(self, shape):
        data_list = np.zeros(shape)
        shrd_list = shared_np_array(np.shape(data_list))
        shrd_list[:] = data_list[:]
        return shrd_list

    def find_pentagons():
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        num_edges = np.array([6 for i in range(0, ncells)])
        cni = self.xfile.load_from('grid', 'vertices_of_vertex')

        zeroes = np.argwhere(cell_neighbour_idx == 0)

        for i in zeroes:
            num_edges[i[1]] = 5
            if i[0] != 5:
                cni[i[0], i[1]] = cni[5,i[1]]
            else:
                cni[5, i[1]] = cni[4,i[1]]

        cni -= 1

        self.xfile.write_to('grid', cni, name='vertices_of_vertex', dims=['vertex'])
        self.xfile.new_dimension('grid', 'edg', 6)
        num_edges = np.moveaxis(num_edges, 0, -1)
        self.xfile.write_to('grid', num_edges, name='num_edges', dims=['edg','vertex'])

    def compute_hex_area_members():
        # number of hexagons in coarse area
        num_hex = math.num_hex_from_rings(self.num_rings)

        ncells = self.xfile.get_dimsize_from('grid', 'vertex')

        a_nei_idx = self.create_array([ncells, num_hex])
        helper = np.empty([ncells, num_hex])
        helper.fill(-1) #masking
        a_nei_idx[:] = helper[:]

        num_edges = self.xfile.load_from('grid', 'num_edges')
        cell_neighbour_idx = self.xfile.load_from('grid', 'vertices_of_vertex')
        cell_index = np.arange(0, necells)

        define_hex_area(cell_neighbour_idx, num_edges, cell_index, a_nei_idx)
        a_nei_idx = np.moveaxis(a_nei_idx, 0, -1)
        self.xfile.new_dimension('grid', 'num_hex', num_hex)
        self.xfile.write_to('grid', a_nei_idx, name='area_member_idx', dims=['num_hex','vertex'])

    def compute_coarse_area():
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        co_ar = self.create_array([ncells])
        co_ar[:] = np.array([0.0 for i in range(0, ncells)])
        cell_area = np.moveaxis(self.xfile.load_from('grid', 'cell_area'), -1, 0)
        a_nei_idx = np.moveaxis(self.xfile.load_from('grid', 'area_member_idx'), -1, 0)

        coarse_area(cell_area, a_nei_idx, co_ar)

        self.xfile.write_to('grid', co_ar, name='coarse_area', dims=['vertex'])

    def compute_gradient_nfo():
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        cell_area = np.moveaxis(self.xfile.load_from('grid', 'cell_area'), -1, 0)
        coarse_area = np.moveaxis(self.xfile.load_from('grid', 'coarse_area'), -1, 0)

        lon = self.xfile.load_from('grid', 'lon')
        lat = self.xfile.load_from('grid', 'lat')
        lonlat = [[loni, lati] for loni, lati in itertools.izip(lon, lat)]

        coords = self.create_array([ncells, 4, 2])
        cell_index = np.arange(0, ncells)





    def execute(self):
        vertex_index = self.file.load_from('grid', 'vertex_index')
        dual_area_p = self.file.load_from('grid', 'dual_area_p')

        self.account_for_pentagons()
        self.define_hex_area()
        self.coarse_area()
        self.get_gradient_nfo()

        #write stuff


@TimeThis()
@requires({
    'full' : ['cell_neighbour_idx', 'num_edges'],
    'slice' : ['cell_index', 'a_nei_idx']
})
@ParallelNpArray(gmp)
def define_hex_area(cell_neighbour_idx, num_edges, cell_index, a_nei_idx):

    for ani, idx in itertools.izip(a_nei_idx, cell_index):
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
                idx_c = cell_neighbour_idx[jn, idx_n]

                if idx_c in ani:
                    pass
                elif jh_c < check_num_hex:
                    ani[jh_c] = idx_c
                    jh_c += 1
                else:
                    break
                    print('define_hex_area: error jh_c too large')

            jh += 1


@TimeThis()
@requires({
    'full' : ['cell_area'],
    'slice' : ['area_member_idx', 'coarse_area']
})
@ParallelNpArray(gmp)
def coarse_area(cell_area, area_member_idx, coarse_area):
    for ami, ca in itertools.izip(area_member_idx, cell_area):
        areas = cell_area[np.where(ami > -1)[0]]
        ca[:] = np.sum(areas)


@TimeThis()
@requires({
    'full' : ['lon', 'lat'],
    'slice' : ['coarse_area', 'cell_area', 'cell_index', 'coordinates', 'member_idx', 'member_rad']
})
@ParallelNpArray(gmp)
def compute_gradient_nfo(lon, lat, coarse_area, cell_area, cell_index, coordinates, member_idx, member_rad):

    times_rad = 2
    max_members = 1 + 6 * times_rad/2 * (times_rad/2 + 1) / 2
    ncells = len(lon)

    for coar, cear, idx, coordi, memidx, memrad in itertools.izip(coarse_area, cell_area, cell_index, coordinates, member_idx, member_rad):
        d = 2 * math.radius(coar)
        coordi[:] = gradient_coordinates(lon[idx], lat[idx], d)

        check_rad = times_rad * mo.radius(cear)

        bounds = np.zeros((2, 2, 2))
        candidates = np.empty([400])
        candidates.fill(-1)


        for j in range(4):
            bounds[:, :, :] = max_min_bounds(coordi[j, 0], coordi[j, 1], check_rad)
            test_lat_1 = np.all([
                np.greater_equal(lat, bounds[0, 0, 0]),
                np.less_equal(lat, bounds[0, 1, 0])
            ], 0)

            test_lat_2 = np.all([
                np.greater_equal(lat, bounds[1, 0, 0]),
                np.less_equal(lat, bounds[1, 1, 0])
            ], 0)

            test_lat = np.any([test_lat_2, test_lat_1], 0)

            test_lon_1 = np.all([
                np.greater_equal(lon, bounds[0, 0, 1]),
                np.less_equal(lon, bounds[0, 1, 1])
            ], 0)

            test_lon_2 = np.all([
                np.greater_equal(lon, bounds[1, 0, 1]),
                np.less_equal(lon, bounds[1, 1, 1])
            ], 0)

            test_lon = np.any([test_lon_2, test_lon_1], 0)

            test = np.all([test_lat, test_lon], 0)

            helper = list(itertools.compress(range(ncells), test))

            candidates[:len(helper)] = helper

            check = candidates[np.where(candidates > -1)[0]]
            cntr = 0
            for k, kidx in enumerate(check):
                check_r = mo.arc_len(
                    coordi[j, :],
                    [lon[kidx], lat[kidx]]
                )
                if check_r <= check_rad:
                    memidx[j, cntr] = idx
                    memrad[j, cntr] = check_r
                    cntr += 1

def gradient_coordinates(lon, lat, d):

    lat_min = lat - d
    lat_max = lat + d

    if lat_max > np.pi/2:
        # NP in query circle:
        lat_max = lat_max - np.pi
    elif lat_min < -np.pi/2:
        # SP in query circle:
        lat_min = lat_min + np.pi
    d = d / np.cos(lonlat[1])
    lon_min = lon - r
    lon_max = lon + r

    if lon_min < -np.pi:
        lon_min = lon_min + np.pi
    elif lon_max > np.pi:
        lon_max = lon_max - np.pi

    coords = np.array([
        np.array([lon_min, lat_min]),
        np.array([lon_min, lat_max]),
        np.array([lon_max, lat_min]),
        np.array([lon_max, lat_max])
    ])

    return coords

def max_min_bounds(lon, lat, r):
    lock = False
    lat_min = lat - r
    lat_max = lat + r
    bounds = np.empty([2, 2, 2])
    bounds.fill(-4)
    if lat_max > np.pi/2:
        #NP
        bounds[0, :, :] = [[lat_min, -np.pi], [np.pi/2, np.pi]]
        lock = True
    elif lat_min < -np.pi/2:
        #NP
        bounds[0, :, :] = [[-np.pi/2, -np.pi], [lat_max, np.pi]]
        lock = True
    else:
        bounds[0, :, :] = [[lat_min, -np.pi], [lat_max, np.pi]]

    if not lock:
        lat_T = np.arcsin(np.sin(lat) / np.cos(r))
        # and so on ....


