#!/usr/bin/env python
# coding=utf-8
import unittest
import numpy as np
import modules.math_mod as math
from decorators.paralleldecorators import gmp, shared_np_array

test_len = 5
test_array = [[np.ones((3,3)), 10, [3,3]] for i in range(test_len)]
total_area = sum([k[1] for i,k in enumerate(test_array)])

rho = [np.ones((3,3)) for i in range(test_len)]
rho_bar = np.ones((3,3))

c_mem_idx = [[i for i in range(5)]]
ret = np.zeros((1,3,3))
return_array = shared_np_array(ret.shape)
return_array[:] = ret[:]
print(type(return_array))

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
        self.assertTrue((np.pi, np.pi/2) == math.get_polar(0.0, 0.0))
        self.assertTrue((np.pi, 0.0) == math.get_polar(0.0, np.pi/2))
        self.assertTrue((0.0, 0.0) == math.get_polar(0.0, -np.pi/2))
        self.assertTrue((0.0, 0.0) == math.get_polar(np.pi, np.pi/2))
        self.assertTrue((0.0, 0.0) == math.get_polar(-np.pi, np.pi/2))
        self.assertTrue((0.0, np.pi/2) == math.get_polar(np.pi, 0.0))
        self.assertTrue((0.0, np.pi/2) == math.get_polar(-np.pi, 0.0))
        self.assertTrue((np.pi/2, np.pi/2) == math.get_polar(-np.pi/2, 0.0))
        self.assertTrue((-np.pi/2, np.pi/2) == math.get_polar(np.pi/2, 0.0))

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

    def test_dist_avg_scalar(self):
        test_array = [[np.ones((3,3)), 10, [3,3]] for i in range(test_len)]
        coords_rads= [[[0.0, 0.0], 1.0] for i in range(test_len)]
        self.assertTrue((np.ones((3,3)) == math.dist_avg_scalar(test_array, coords_rads)).all())

    def test_dist_avg_vec(self):
        test_array = [[np.ones((3,3)), 10, [0,0]] for i in range(test_len)]
        coords_rads= [[[0.0, 0.0], 1.0] for i in range(test_len)]
        # outcome is ok truth check isn't
        self.assertTrue(
            np.allclose(
                np.ones((3,3)),
                math.dist_avg_vec(test_array, test_array, coords_rads)[0]
            )
        )
        self.assertTrue(
            np.allclose(
                np.ones((3,3)),
                math.dist_avg_vec(test_array, test_array, coords_rads)[1]
            )
        )
    def test_central_diff(self):
        self.assertAlmostEqual(1, math.central_diff(4,2,1))

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


if __name__ == '__main__':
    unittest.main()
