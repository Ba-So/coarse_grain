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
        print(type(return_array))
        math.bar_scalar(test_array, [total_area], c_mem_idx, return_array)
        self.assertTrue(([np.ones((3,3))] == return_array).all())

    def test_hat_scalar_lin(self):
        gmp.set_parallel_proc(False)
        return_array[:] = np.zeros((1,3,3))
        math.hat_scalar(test_array, rho, [rho_bar], [total_area], c_mem_idx, return_array)
        self.assertTrue(([np.ones((3,3))] == return_array).all())

    def test_hat_scalar_par(self):
        gmp.set_parallel_proc(True)
        print(gmp.switch)
        return_array[:] = np.zeros((1,3,3))
        math.hat_scalar(test_array, rho, [rho_bar], [total_area], c_mem_idx, return_array)
        self.assertTrue(([np.ones((3,3))] == return_array).all())

    def test_bar_scalar_par(self):
        gmp.set_parallel_proc(True)
        return_array[:] = np.zeros((1,3,3))
        math.bar_scalar(test_array, [total_area], c_mem_idx, return_array)
        self.assertTrue(([np.ones((3,3))] == return_array).all())

if __name__ == '__main__':
    unittest.main()


