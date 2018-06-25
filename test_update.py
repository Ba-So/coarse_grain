#!/usr/bin/env python
# coding=utf-8

import unittest
import numpy as np
import update as up
import global_vars as gv

class TestUpdater(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestUpdater, self).__init__(*args, **kwargs)
        self.update = up.Updater()
        self.testdict = {
            'a' : np.array([1,2,3]),
            'b' : np.array([3,4,5])
        }

    def test_complete(self):
        self.testdict = {
            'a' : np.array([1,2,3]),
            'b' : np.array([3,4,5])
        }
        self.update.complete('grid_nfo', self.testdict)
        for key, item in gv.globals_dict['grid_nfo'].iteritems():
            self.assertTrue(np.all([self.testdict[key], item]))

    def test_up_entry(self):
        self.testdict = {
            'a' : np.array([1,2,3]),
            'b' : np.array([3,4,5])
        }
        test_slice = {
            'c' : np.array([5,6,7]),
            'd' : np.array([8,9,10])
        }
        self.testdict.update(test_slice)
        self.update.up_entry('grid_nfo', test_slice)
        for key, item in gv.globals_dict['grid_nfo'].iteritems():
            self.assertTrue(np.all([self.testdict[key], item]))

    def test_rm_entry(self):
        self.testdict = {
            'a' : np.array([1,2,3]),
            'b' : np.array([3,4,5]),
            'c' : np.array([5,6,7]),
            'd' : np.array([8,9,10])
        }
        test_slice = ['c' , 'd']

        self.testdict.pop(*test_slice)
        self.update.rm_entry('grid_nfo', test_slice)
        for key, item in gv.globals_dict['grid_nfo'].iteritems():
            self.assertTrue(np.all([self.testdict[key], item]))

    def test_part(self):
        self.testdict = {
            'a' : np.array([1,2,3]),
            'b' : np.array([3,4,5])
        }
        test_slice = {'a' : np.array([4,5])}
        test_index = [(slice(1,3))]
        self.testdict['a'][test_index] = test_slice['a']
        self.update.part('grid_nfo', test_slice, test_index)
        for key, item in self.testdict.iteritems():
            self.assertTrue(np.all([item, gv.globals_dict['grid_nfo'][key] ]))


if __name__ == '__main__':
    unittest.main()

