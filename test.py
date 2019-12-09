#!/usr/bin/env python
# coding=utf-8
import unittest
import numpy as np
from math import radians
import modules.new_grad_mod as grad
import modules.math_mod as math
from decorators.paralleldecorators import gmp, shared_np_array
from plot import plot_coords, plot_bounds

test_len = 5
test_array = [[np.ones((3,3)), 10, [3,3]] for i in range(test_len)]
total_area = sum([k[1] for i,k in enumerate(test_array)])

rho = [np.ones((3,3)) for i in range(test_len)]
rho_bar = np.ones((3,3))

c_mem_idx = [[i for i in range(5)]]
ret = np.zeros((1,3,3))
return_array = shared_np_array(ret.shape)
return_array[:] = ret[:]

class TestMath(unittest.TestCase):
    def test_avg_bar(self):
        self.assertTrue((np.ones((3,3)) ==  math.avg_bar(test_array, total_area)).all())

    def test_avg_hat(self):
        self.assertTrue((np.ones((3,3)) ==  math.avg_hat(test_array, rho, rho_bar, total_area)).all())

    def test_bar_scalar_lin(self):
        gmp.set_parallel_proc(False)
        return_array[:] = np.zeros((1,3,3))
        math.bar_scalar(test_array, [total_area], c_mem_idx, return_array)
        self.assertTrue(([np.ones((3,3))] == return_array).all())

    def test_hat_scalar_lin(self):
        gmp.set_parallel_proc(False)
        return_array[:] = np.zeros((1,3,3))
        math.hat_scalar(test_array, rho, [rho_bar], [total_area], c_mem_idx, return_array)
        self.assertTrue(([np.ones((3,3))] == return_array).all())

    def test_hat_scalar_par(self):
        gmp.set_parallel_proc(True)
        return_array[:] = np.zeros((1,3,3))
        math.hat_scalar(test_array, rho, [rho_bar], [total_area], c_mem_idx, return_array)
        self.assertTrue(([np.ones((3,3))] == return_array).all())

    def test_bar_scalar_par(self):
        gmp.set_parallel_proc(True)
        return_array[:] = np.zeros((1,3,3))
        math.bar_scalar(test_array, [total_area], c_mem_idx, return_array)
        self.assertTrue(([np.ones((3,3))] == return_array).all())

    def test_get_polar(self):
        self.assertTrue((0.0, np.pi/2) == math.get_polar(0.0, 0.0))
        self.assertTrue((np.pi, 0.0) == math.get_polar(0.0, np.pi/2))
        self.assertTrue((0.0, 0.0) == math.get_polar(0.0, -np.pi/2))
        self.assertTrue((0.0, 0.0) == math.get_polar(np.pi, np.pi/2))
        self.assertTrue((0.0, 0.0) == math.get_polar(-np.pi, np.pi/2))
        self.assertTrue((np.pi, np.pi/2) == math.get_polar(np.pi, 0.0))
        self.assertTrue((-np.pi, np.pi/2) == math.get_polar(-np.pi, 0.0))
        self.assertTrue((-np.pi/2, np.pi/2) == math.get_polar(-np.pi/2, 0.0))
        self.assertTrue((np.pi/2, np.pi/2) == math.get_polar(np.pi/2, 0.0))
        self.assertTrue((-np.pi/2, np.pi/4) == math.get_polar(np.pi/2, np.pi/4))

    def test_rotate_vec_local_global(self):
        # test for invariance of vectors at equator
        for lon in [0.0, np.pi/2, np.pi, -np.pi, -np.pi/2]:
            for lat in [-np.pi/2, 0.0, np.pi/2]:
                plon, plat = math.get_polar(lon, lat)
                x = [3.0, 0.0, [lon, lat]]
                y = [3.0, 0.0, [lon, lat]]
                resx, resy = math.rotate_vec_to_local(plon, plat, x, y)
                resx, resy = math.rotate_vec_to_global(plon, plat, resx, resy)
                self.assertAlmostEqual(x[0], resx[0], places=7)
                self.assertAlmostEqual(y[0], resy[0], places=7)

    def test_rotate_vec_local(self):
        # test for invariance of vectors at equator
        for lon in [0.0, np.pi/2, np.pi, -np.pi, -np.pi/2]:
            for lat in [-np.pi/2, 0.0, np.pi/2]:
                plon, plat = math.get_polar(lon, lat)
                x = [3.0, 0.0, [lon, lat]]
                y = [3.0, 0.0, [lon, lat]]
                resx, resy = math.rotate_vec_to_local(plon, plat, x, y)
                self.assertAlmostEqual(x[0], resx[0], places=7)
                self.assertAlmostEqual(y[0], resy[0], places=7)

        x = [3.0, 0.0, [np.pi/2, np.pi/4]]
        y = [3.0, 0.0, [np.pi/2, np.pi/4]]
        plon, plat = math.get_polar(-np.pi, np.pi/3)
        print(plon, plat, -np.pi, np.pi/3)
        resx, resy = math.rotate_vec_to_local(plon, plat, x, y)
        print(resx[0], resy[0])

    def test_central_diff(self):
        self.assertAlmostEqual(1, math.central_diff(4,2,1))
        self.assertAlmostEqual(2, math.central_diff(8,4,1))

    def test_scalar_flucts(self):
        test_array = [[np.ones((3,3)), 10, [3,3]] for i in range(test_len)]
        avg_array = np.ones((3,3))
        avg_array.fill(2)

        for i in range(test_len):
            self.assertTrue(
                np.allclose(
                    test_array[i][0],
                    math.scalar_flucts(test_array, avg_array)[i][0]
                )
            )
            self.assertTrue(
                np.allclose(
                    test_array[i][1],
                    math.scalar_flucts(test_array, avg_array)[i][1]
                )
            )

    def test_vec_flucts(self):
        test_array = [[np.ones((3,3)), 10, [np.pi/2,np.pi/4]] for i in range(test_len)]
        avg_array = np.ones((3,3))
        avg_array.fill(2)

        self.assertTrue(
            np.allclose(
                test_array[0][0],
                math.vec_flucts(test_array, test_array, avg_array, avg_array)[0][0][0]
            )
        )

    def test_arc_len(self):
        p_x = [radians(-1.7297222222222221), radians(53.32055555555556)]
        p_y = [radians(-1.6997222222222223), radians(53.31861111111111)]
        self.assertAlmostEqual(0.0003146080424221651, math.arc_len(p_x, p_y))

    def test_rotation_jacobian(self):
        '''testing the rotation jacobian, for it's edge cases'''
        # a test to see jacobian just beyond pole.
        c_lon = 0.0
        c_lat = np.pi/3
        t_lon = np.pi/8
        t_lat = np.pi/4
        vec_x = [1, 2, [t_lon, t_lat]]
        vec_y = [1, 2, [t_lon, t_lat]]
        plon, plat = math.get_polar(c_lon, c_lat)
        print('jacobian: {}'.format(math.rotation_jacobian(t_lon, t_lat, plon, plat)))
        print('vector: {}'.format(math.rotate_vec_to_local(plon, plat, vec_x, vec_y)))
        c_lon = np.pi - 0.0001
        plon, plat = math.get_polar(c_lon, c_lat)
        print('jacobian: {}'.format(math.rotation_jacobian(t_lon, t_lat, plon, plat)))
        print('vector: {}'.format(math.rotate_vec_to_local(plon, plat, vec_x, vec_y)))

    def test_get_dx_dy(self):
        p_x = [0.0, 0.0]
        p_y = [np.pi/8, np.pi/8]
        dx, dy = math.get_dx_dy(p_x, p_y)
        print(math.arc_len(p_x, p_y), np.arccos(np.cos(dx) * np.cos(dy)))
        self.assertAlmostEqual(math.arc_len(p_x, p_y), np.arccos(np.cos(dx) * np.cos(dy)))

    def test_lst_sq_intp(self):
        x_values = [[np.array([[5]]), 0.0] for i in range(5)]
        c_coord = [0.0, 0.0]
        distance = [[0.2, 0.1] for i in range(3)]
        for i in range(2): distance.append([-0.4, -0.2])
        self.assertAlmostEqual(5, math.lst_sq_intp(x_values, c_coord, distance)[0][0][0])


class TestGrad(unittest.TestCase):
    def test_contains_pole(self):
        lat = np.pi/2
        r = 0.5
        self.assertEqual('NP', grad.contains_pole(lat,r))
        self.assertEqual('SP', grad.contains_pole(-lat, r))
        self.assertEqual(None, grad.contains_pole(0, r))
        self.assertEqual('NP', grad.contains_pole(lat-r,r))
        self.assertEqual('SP', grad.contains_pole(-lat+r,r))

    def test_get_tangent_lons(self):
        r = np.pi / 4
        lon = np.pi
        polecase=[-np.pi, np.pi]
        lat = np.pi / 2
        self.assertListEqual(polecase, grad.get_tangent_lons([lon, lat], r))
        lat = np.pi / 2 - r
        self.assertListEqual(polecase, grad.get_tangent_lons([lon, lat], r))
        lat = -np.pi / 2 + r
        self.assertListEqual(polecase, grad.get_tangent_lons([lon, lat], r))
        lat = -np.pi / 2
        self.assertListEqual(polecase, grad.get_tangent_lons([lon, lat], r))
        lat = -np.pi / 5
        self.assertNotEqual(polecase, grad.get_tangent_lons([lon, lat], r))


    def test_handle_meridians(self):
        # not so weird cases
        lon_max = np.pi / 2
        lon_min = -np.pi - np.pi / 8
        expectedbounds = [[-np.pi, lon_max],[np.pi-np.pi/8 ,np.pi]]
        self.assertListEqual(expectedbounds, grad.handle_meridians(lon_min, lon_max))
        lon_max = np.pi + np.pi / 8
        lon_min = np.pi / 2
        expectedbounds = [[lon_min, np.pi],[-np.pi , -np.pi + np.pi / 8]]
        self.assertListEqual(expectedbounds, grad.handle_meridians(lon_min, lon_max))
        lon_max = np.pi/4
        lon_min = -np.pi/4
        expectedbounds = [[lon_min, lon_max],[-4, -4]]
        self.assertListEqual(expectedbounds, grad.handle_meridians(lon_min, lon_max))

    def test_check_bounds(self):
        pair = [0, 0]
        bounds = [
            [[-np.pi, np.pi], [-4, -4]],
            [-np.pi/2, np.pi/2]
        ]
        self.assertTrue(grad.check_bounds(pair, bounds))
        bounds = [
            [[-np.pi/2, np.pi], [-np.pi, -np.pi/2]],
            [-np.pi/2, np.pi/2]
        ]
        self.assertTrue(grad.check_bounds(pair, bounds))

    def test_mer_points(self):
        c_coord = [0.0, 0.0]
        distance = np.pi/4
        #
        self.assertListEqual([[0.0, distance],[0.0, -distance]], grad.mer_points(c_coord, distance))
        # noth pole edge case:
        c_coord = [0.0, np.pi/4]
        distance = np.pi/2
        self.assertListEqual([[np.pi, np.pi/4],[0.0, -np.pi/4]], grad.mer_points(c_coord, distance))
        # south pole edge case:
        c_coord = [0.0, -np.pi/4]
        distance = np.pi/2
        self.assertListEqual([[0.0, np.pi/4],[np.pi, -np.pi/4]], grad.mer_points(c_coord, distance))
        # north and south pole edge case
        c_coord = [0.0, 0.0]
        distance = np.pi
        self.assertListEqual([[np.pi, np.pi - distance],[np.pi, -np.pi + distance]], grad.mer_points(c_coord, distance))

    def test_zon_points(self):
        c_coord = [0.0, 0.0]
        distance = np.pi / 8
        # we expect the points to be simply np.pi/8 east/west:
        self.assertListEqual([[distance, 0.0], [-distance, 0.0]], grad.zon_points(c_coord, distance))

    def test_get_distance(self):
        lat = 0.0
        distance = np.pi/8
        self.assertAlmostEqual(distance, grad.get_distance(lat, distance))
        # edge case, where just acceptable
        distance = np.pi/4
        lat = np.pi/3 # cos(lat) = 0.5 > 2 * distance == np.pi/2
        self.assertAlmostEqual(np.pi/2, grad.get_distance(lat, distance))
        # just beyond edge case, where we expect -4
        distance = np.pi/4 + 0.0001
        lat = np.pi/3 # cos(lat) = 0.5 > 2 * distance == np.pi/2
        self.assertEqual(None, grad.get_distance(lat, distance))

    def test_convert_rad_m(self):
        distance = np.pi/8
        r = 6.37111*10**6
        self.assertAlmostEqual(r/16, grad.convert_rad_m(distance), places=7)
    def test_meridian_care(self):
        self.assertAlmostEqual(np.pi/2, grad.meridian_care(-3 * np.pi / 2))
        self.assertAlmostEqual(-np.pi/2, grad.meridian_care(3 * np.pi / 2))
        self.assertEqual(np.pi/4, grad.meridian_care(np.pi/4))
        self.assertEqual(-np.pi/4, grad.meridian_care(-np.pi/4))

    def test_clean_indices(self):
        test = [-1, 2, -1, -1, -2, 0, 3, 1000]
        expect = [2, 0, 3, 1000]
        self.assertListEqual(expect, grad.clean_indices(test))
        test = [-1, -1]
        expect = []
        self.assertListEqual(expect, grad.clean_indices(test))
        test = []
        expect = []
        self.assertListEqual(expect, grad.clean_indices(test))
        test = [-1, 2, -1, 3, 3]
        grind = grad.clean_indices(test)
        for k in grind: self.assertIsInstance(k, int)
        test = [3.0, 4.5, -1, 3, 3]
        grind = grad.clean_indices(test)
        for k in grind: self.assertIsInstance(k, int)


if __name__ == '__main__':
    unittest.main()
