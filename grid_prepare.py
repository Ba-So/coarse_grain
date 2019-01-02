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
        grad_index = self.create_array([ncells, 4, max_members], dtype='int')
        grad_index[:] = -1
        grad_dist = self.create_array([ncells, 4, max_members])
        grad_dist[:] = -1
        cell_idx = np.arange(ncells)

        compute_gradient_nfo(
            lon, lat, coarse_area, cell_area, cell_idx, #input
            grad_coords, grad_index, grad_dist #output
        )

        self.xfile.new_dimension('grid', 'gradnum', 4)
        self.xfile.new_dimension('grid', 'lonlat', 2)
        self.xfile.write_to(
            'grid', grad_coords, name='grad_coords',
            dims=['gradnum', 'lonlat', 'vertex'],
            attrs={'long_name': 'gradient coordinates'}
        )
        self.xfile.new_dimension('grid', 'maxmem', max_members)
        self.xfile.write_to(
            'grid', grad_index, dtype='i4', name='grad_idx',
            dims=['gradnum', 'maxmem', 'vertex'],
            attrs={'long_name': 'gradient indices'}
        )
        self.xfile.write_to(
            'grid', grad_dist, name='grad_dist',
            dims=['gradnum', 'maxmem', 'vertex'],
            attrs={'long_name': 'gradient distances'}
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

    for ami in itertools.izip(area_member_idx):
        areas = cell_area[np.where(ami > -1)[0]]
        ca.append(np.sum(areas))
    coarse_area[:] = ca

@TimeThis
@requires({
    'full' : ['lon', 'lat'],
    'slice' : ['coarse_area', 'cell_area', 'cell_idx',
               'grad_coords', 'grad_idx', 'grad_dist']
})
@ParallelNpArray(gmp)
def compute_gradient_nfo(lon, lat, coarse_area, cell_area, cell_idx, grad_coords, grad_idx, grad_dist):
    '''return:
        the coordinates of E,W,N,S points for the computation of the gradients
        [ncells, i, j] i{0..3} E,W,N,S ; j{0,1} lon, lat
        the indices of points around those points to be distance averaged over
        [ncells, i, j] i{0..3} E,W,N,S ; j{0..max_members}
        the distance of points around those points for distance average
        [ncells, i, j] i{0..3} E,W,N,S ; j{0..max_members}
        '''

    times_rad = 2.1
    ncells = len(lon)
    start = 0

    for i, idx in enumerate(cell_idx):

        d = 2 * math.radius(coarse_area[i])
        check_rad = times_rad * math.radius(cell_area[i])

        grad_coordi = gradient_coordinates(lon[idx], lat[idx], d)
        grad_coords[i, :] = grad_coordi

        for j in range(4): # in range(E,W,N,S)
            bounds = max_min_bounds(
                grad_coordi[j, 0], grad_coordi[j, 1], check_rad
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

            check = list(itertools.compress(range(ncells), test))

            cntr = 0
            for k, kidx in enumerate(check):
                check_r = math.arc_len(
                    grad_coordi[j, :],
                    [lon[kidx], lat[kidx]]
                )
                if check_r <= check_rad:
                    grad_idx[i, j, cntr] = kidx
                    grad_dist[i, j, cntr] = check_r
                    cntr += 1
            if cntr == 0:
                sys.exit('no one found \n {} \n {}'.format(grad_coordi, bounds))


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
        lat_max = np.pi/2
    elif lat_min < -np.pi/2:
        # SP in query circle:
        lat_min = -np.pi/2

    d_lon = d / np.cos(lat)
    lon_min = lon - d_lon
    lon_max = lon + d_lon

    # project points beyond 180E/180W onto W/E
    # make 180 unique-> project all -180 onto 180!
    if d >= np.pi:
        # happens in case of pole
        lon_min = -np.pi/2
        lon_max =  np.pi/2
        # to have a at least somewhat
        # meaningful definition for the poles
        # gradients break down anyway
    elif lon_min <= -np.pi:
        # lon_min <= -180 (180W) => project onto E
        lon_min = lon_min + 2 * np.pi
    elif lon_max > np.pi:
        # lon_min > 180 (180E) => project onto W
        lon_max = lon_max - 2* np.pi

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
        lat_max = np.pi/2

    elif lat_min < -np.pi/2:
        #SP
        lat_min < -np.pi/2

    # Give standard values
    bounds[0, :, :] = [[-np.pi, lat_min], [np.pi, lat_max]]
    # ----------- lons ------------
    # computing tangent longditudes to circle on sphere
    # compare Bronstein
    lat_t = np.arcsin(np.sin(lat) / np.cos(r))
    # computing delta longditude
    np.seterr(all='raise')
    try:
        d_lon = np.arccos(
            (np.cos(r) - np.sin(lat_t) * np.sin(lat))
            /(np.cos(lat_t) * np.cos(lat))
        )
    except:
        sys.exit('weird value: r{}, latt{}, lat{}'.format(r, lat_t, lat))
    lon_min = lon - d_lon
    lon_max = lon + d_lon

    # Funky stuff in case of meridan in query
    if lon_min < -np.pi:
        bounds[1, :, :] = bounds[0, :, :]
        bounds[0, 0, 0] = lon_min + 2 * np.pi # to be min value
        bounds[0, 1, 0] = np.pi # to be max value
        #and
        bounds[1, 0, 0] = - np.pi # to be min value
        bounds[1, 1, 0] = lon_max # to be max value
    elif lon_max > np.pi:
        bounds[1, :, :] = bounds[0, :, :]
        bounds[0, 0, 0] = lon_min # to be min value
        bounds[0, 1, 0] = np.pi # to be max value
        #and
        bounds[1, 0, 0] = - np.pi # to be min value
        bounds[1, 1, 0] = lon_max - 2 * np.pi # to be max value

    else:
        bounds[0, 0, 0] = lon_min # to be min value
        bounds[0, 1, 0] = lon_max # to be max value

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
        'preparing the gridfile {} for coarse grainig over {} rings.'
    ).format(path.join(args.path_to_file[0], args.grid_name[0]), args.num_rings[0])
    new_name = args.grid_name[0][:-3] +  '_refined_{}.nc'.format(args.num_rings[0])
    copyfile(
        path.join(args.path_to_file[0], args.grid_name[0]),
        path.join(args.path_to_file[0], new_name)
    )
    gmp.set_parallel_proc(True)
    gmp.set_num_procs(12)
    GP = Grid_Preparator(args.path_to_file[0], new_name, args.num_rings[0])
    GP.execute()


