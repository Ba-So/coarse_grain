#!/usr/bin/env python
# coding=utf-8
import argparse
from shutil import copyfile
from os import path
import itertools
import numpy as np
import xarray as xr
from decorators.paralleldecorators import gmp, ParallelNpArray, shared_np_array
from decorators.debugdecorators import TimeThis, PrintArgs, PrintReturn
from decorators.functiondecorators import requires
import modules.cio as cio
import modules.math_mod as math

class Grid_Preparator(object):
    def __init__(self, path, gridn, num_rings):
        self.num_rings = num_rings
        self.xfile = cio.IOcontroller(path, gridn)

    def create_array(self, shape):
        data_list = np.zeros(shape)
        shrd_list = shared_np_array(np.shape(data_list))
        shrd_list[:] = data_list[:]
        return shrd_list

    def find_pentagons(self):
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        num_edges = np.array([6 for i in range(0, ncells)])
        cni = self.xfile.load_from('grid', 'vertices_of_vertex')

        zeroes = np.argwhere(cni == 0)
        #0-ncells 1-ne

        for i in zeroes:
            num_edges[i[0]] = 5
            if i[1] != 5:
                cni[i[0], i[1]] = cni[i[0], 5]
            else:
                cni[i[0], 5] = cni[i[0], 4]

        cni -= 1

        self.xfile.write_to(
            'grid', cni, dtype = 'i2', name='vertices_of_vertex',
            dims=['ne', 'vertex']
        )
        self.xfile.write_to(
            'grid', num_edges, dtype = 'i2', name='num_edges',
            dims=['vertex']
        )

    def compute_hex_area_members(self):
        # number of hexagons in coarse area
        num_hex = math.num_hex_from_rings(self.num_rings)

        ncells = self.xfile.get_dimsize_from('grid', 'vertex')

        a_nei_idx = self.create_array([ncells, num_hex])
        helper = np.empty([ncells, num_hex])
        helper.fill(-1) #masking
        a_nei_idx[:] = helper[:]

        num_edges = self.xfile.load_from('grid', 'num_edges')
        cell_neighbour_idx = self.xfile.load_from('grid', 'vertices_of_vertex')
        cell_index = np.arange(0, ncells)

        define_hex_area(cell_neighbour_idx, num_hex, num_edges, cell_index, a_nei_idx)
        a_nei_idx = np.moveaxis(a_nei_idx, 0, -1)
        self.xfile.new_dimension('grid', 'num_hex', num_hex)
        self.xfile.write_to(
            'grid', a_nei_idx, dtype = 'i2', name='area_member_idx',
            dims=['num_hex','vertex']
        )

    def compute_coarse_area(self):
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        co_ar = self.create_array([ncells])
        co_ar[:] = np.array([0.0 for i in range(0, ncells)])
        cell_area = self.xfile.load_from('grid', 'cell_area')
        a_nei_idx = self.xfile.load_from('grid', 'area_member_idx')

        coarse_area(cell_area, a_nei_idx, co_ar)

        self.xfile.write_to('grid', co_ar, name='coarse_area', dims=['vertex'])

    def compute_gradient_nfo(self):
        ncells = self.xfile.get_dimsize_from('grid', 'vertex')
        cell_area = self.xfile.load_from('grid', 'cell_area')
        coarse_area = self.xfile.load_from('grid', 'coarse_area')

        lon = self.xfile.load_from('grid', 'vlon')
        lat = self.xfile.load_from('grid', 'vlat')

        times_rad = 2
        max_members = 1 + 6 * times_rad/2 * (times_rad/2 + 1) / 2
        grad_coords = self.create_array([ncells, 4, 2])
        grad_index = self.create_array([ncells, 4, max_members])
        grad_index[:] = -1
        grad_dist = self.create_array([ncells, 4, max_members])
        grad_dist[:] = -1
        cell_idx = np.arange(ncells)

        compute_gradient_nfo(
            lon, lat, cell_area, coarse_area, cell_idx, #input
            grad_coords, grad_index, grad_dist #output
        )

        self.xfile.new_dimension('grid', 'gradnum', 4)
        self.xfile.new_dimension('grid', 'lonlat', 2)
        self.xfile.write_to(
            'grid', grad_coords, name='gradient coordinates',
            dims=['gradnum', 'lonlat', 'vertex']
        )
        self.xfile.new_dimension('grid', 'maxmem', max_members)
        self.xfile.write_to(
            'grid', grad_index, dtype = 'i2', name='gradient index',
            dims=['gradnum', 'maxmem', 'vertex']
        )
        self.xfile.write_to(
            'grid', grad_dist, name='gradient distances',
            dims=['gradnum', 'maxmem', 'vertex']
        )


    def execute(self):
        print('locating pentagons...')
        self.find_pentagons()
        print('computing coarse area member indices...')
        self.compute_hex_area_members()
        print('computing coarse area...')
        self.compute_coarse_area()
        print('preparing gradient computation...')
        self.compute_gradient_nfo()

        #write stuff


@TimeThis
@requires({
    'full' : ['cell_neighbour_idx', 'num_hex', 'num_edges'],
    'slice' : ['cell_index', 'a_nei_idx']
})
@ParallelNpArray(gmp)
def define_hex_area(cell_neighbour_idx, num_hex, num_edges, cell_index, a_nei_idx):

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
    for ami, ca in itertools.izip(area_member_idx, cell_area):
        areas = cell_area[np.where(ami > -1)[0]]
        ca = np.sum(areas)

@TimeThis
@requires({
    'full' : ['lon', 'lat'],
    'slice' : ['coarse_area', 'cell_area', 'cell_index',
               'grad_coords', 'grad_idx', 'grad_dist']
})
@ParallelNpArray(gmp)
def compute_gradient_nfo(lon, lat, coarse_area, cell_area, cell_index, grad_coords, grad_idx, grad_dist):
    '''return:
        the coordinates of E,W,N,S points for the computation of the gradients
        [ncells, i, j] i{0..3} E,W,N,S ; j{0,1} lon, lat
        the indices of points around those points to be distance averaged over
        [ncells, i, j] i{0..3} E,W,N,S ; j{0..max_members}
        the distance of points around those points for distance average
        [ncells, i, j] i{0..3} E,W,N,S ; j{0..max_members}
        '''

    times_rad = 2
    max_members = 1 + 6 * times_rad/2 * (times_rad/2 + 1) / 2
    ncells = len(lon)

    for coar, cear, idx, coordi, memidx, memrad in itertools.izip(
        coarse_area, cell_area, cell_index, grad_coords, grad_idx, grad_dist
    ):
        d = 2 * math.radius(coar)
        coordi[:] = gradient_coordinates(lon[idx], lat[idx], d)

        check_rad = times_rad * math.radius(cear)

        bounds = np.zeros((2, 2, 2))
        candidates = np.empty([400])
        candidates.fill(-1)


        for j in range(4): # in range(E,W,N,S)
            bounds[:, :, :] = max_min_bounds(
                coordi[j, 0], coordi[j, 1], check_rad
            )

            test_lat_1 = np.all([
                np.greater_equal(lat, bounds[0, 0, 1]),
                np.less_equal(lat, bounds[0, 1, 1])
            ], 0)

            test_lat_2 = np.all([
                np.greater_equal(lat, bounds[1, 0, 1]),
                np.less_equal(lat, bounds[1, 1, 1])
            ], 0)

            test_lat = np.any([test_lat_2, test_lat_1], 0)

            test_lon_1 = np.all([
                np.greater_equal(lon, bounds[0, 0, 0]),
                np.less_equal(lon, bounds[0, 1, 0])
            ], 0)

            test_lon_2 = np.all([
                np.greater_equal(lon, bounds[1, 0, 0]),
                np.less_equal(lon, bounds[1, 1, 0])
            ], 0)

            test_lon = np.any([test_lon_2, test_lon_1], 0)

            test = np.all([test_lon, test_lat], 0)
            # check: i switched lat and lon here

            helper = list(itertools.compress(range(ncells), test))

            candidates[:len(helper)] = helper

            check = candidates[np.where(candidates > -1)[0]]
            cntr = 0
            for k, kidx in enumerate(check):
                check_r = math.arc_len(
                    coordi[j, :],
                    [lon[kidx], lat[kidx]]
                )
                if check_r <= check_rad:
                    memidx[j, cntr] = idx
                    memrad[j, cntr] = check_r
                    cntr += 1

def gradient_coordinates(lon, lat, d):
    '''
        computes the locations to the West, East and South, North
        to be used for the computation of the gradients.
        Values around these poits will later be Distance-Averaged onto
        these points.
        input:
            lon, lat: coordinates of center point
            d:        Distance of W,E,S,N points from center´
        returns:
            coords [i, j]: i {0..3}: E,W,N,S, j {0,1}: lon, lat
    '''

    lat_min = lat - d
    lat_max = lat + d

    if lat_max > np.pi/2:
        # NP in query circle:
        lat_max = lat_max - np.pi
    elif lat_min < -np.pi/2:
        # SP in query circle:
        lat_min = lat_min + np.pi
    d = d / np.cos(lat)
    lon_min = lon - d
    lon_max = lon + d

    if lon_min < -np.pi:
        lon_min = lon_min + np.pi
    elif lon_max > np.pi:
        lon_max = lon_max - np.pi

    coords = np.array([
        np.array([lon_max, lat]), #East Point
        np.array([lon_min, lat]), #West Point
        np.array([lon, lat_max]), #North Point
        np.array([lon, lat_min])  #South Point
    ])
    return coords

def max_min_bounds(lon, lat, r):
    '''
        computes the maximum and minimum lat and lon
        values on a circle on a spere.
        input:
            lon, lat - coordinates of center point
            r        - radius of area to be bounded
        output:
            bounds: array [i, j, k]
                i = {0, 1} for special cases around tho poles
                j = {0, 1} containig (W, S)-min and (E, N)-max
                k = {0, 1} [lon, lat]
    '''

    ispole = False
    lat_min = lat - r
    lat_max = lat + r
    bounds = np.empty([2, 2, 2])
    bounds.fill(-4)
    if lat_max > np.pi/2:
        #NP
        bounds[0, :, :] = [[-np.pi, lat_min], [np.pi, np.pi/2]]
        ispole = True
    elif lat_min < -np.pi/2:
        #SP
        bounds[0, :, :] = [[-np.pi, -np.pi/2], [np.pi, lat_max]]
        ispole = True
    else:
        bounds[0, :, :] = [[-np.pi, lat_min], [np.pi, lat_max]]

    #lon bounds only neccesarry when no pole in query circle
    if not ispole:
        #helper
        lat_t = np.arcsin(np.sin(lat) / np.cos(r))
        #computing delta longditude
        d_lon = np.arccos(
            (np.cos(r) - np.sin(lat_t) * np.sin(lat))
            /(np.cos(lat_t) * np.cos(lat))
        )
        lon_min = lon - d_lon
        lon_max = lon + d_lon

        # Funky stuff in case of meridan in query
        if lon_min < -np.pi:
            bounds[1, :, :] = bounds[0, :, :]
            bounds[0, 0, 0] = lon_min + 2 * np.pi
            bounds[0, 1, 0] = np.pi
            #and
            bounds[1, 0, 0] = - np.pi
            bounds[1, 1, 0] = lon_max
        elif lon_max > np.pi:
            bounds[1, :, :] = bounds[0, :, :]
            bounds[0, 0, 0] = lon_min
            bounds[0, 1, 0] = np.pi
            #and
            bounds[1, 0, 0] = - np.pi
            bounds[1, 1, 0] = lon_max - 2 * np.pi
        else:
            bounds[0, 0, 0] = lon_max
            bounds[0, 1, 1] = lon_max

    return bounds

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
        'preparing the gridfile {}{} for coarse grainig over {} rings.'
    ).format(args.path_to_file[0], args.grid_name[0], args.num_rings[0])
    new_name = args.grid_name[0][:-3] +  '_refined_{}.nc'.format(args.num_rings[0])
    copyfile(
        path.join(args.path_to_file[0], args.grid_name[0]),
        path.join(args.path_to_file[0], new_name)
    )
    gmp.set_parallel_proc(True)
    gmp.set_num_procs(12)
    GP = Grid_Preparator(args.path_to_file[0], new_name, args.num_rings[0])
    GP.execute()


