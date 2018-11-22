#!/usr/bin/env python
# coding=utf-8
import ctypes
import numpy as np
from multiprocessing import Process, Queue, Array
from debugdecorators import TimeThis, PrintArgs, PrintReturn

class Mp(object):
    def __init__(self, num_procs, switch):
        self.num_procs = num_procs
        self.switch = switch
    def toggle_switch(self):
        self.switch = not(self.switch)
    def change_num_procs(self, num_procs):
        self.num_procs = num_procs

class ParallelNpArray(object):
    """do parallelisation using shared memory"""
    def __init__(self, mp=None):
        self._mp = mp
    def __call__(self, func):
        def _parall(*args, **kwargs):
            if self._mp.switch:
                if hasattr(_parall, "needs"):
                    print('Yay')
                slices = self.prepare_slices(args[0], self._mp)
                result_queue = []
                processes = []
                if hasattr(_parall, 'needs'):
                    args_sliced = [
                        {
                            need : kwargs[need] for need in needs[full]
                        } for i in range(self._mp.num_procs)
                    ]
                    for i in range(self._mp.num_procs):
                        for need in needs[part] if arg in needs[part]:
                            args_sliced.update({})

                for i in range(self._mp.num_procs):
                    p = Process(target=func, args=args)
                    processes.append(p)
                for p in processes:
                    p.run()
            else:
               func(*args)
        return _parall

    @staticmethod
    #@PrintArgs
    def array_worker(func, fargs, x_slice):
        """worker to do the slicing and all"""
        print('here worker')
        args = [arg[x_slice] for arg in fargs]
        func(*args)

    @staticmethod
    #@PrintReturn
    def prepare_slices(x_list, mp):
        """returns appropriate slices for worker"""
        l_shape = np.shape(x_list)
        if mp.num_procs > 1:
            len_slice = l_shape[0] // (mp.num_procs - 1)
        else:
            len_slice = l_shape[0]
        len_slice = max(len_slice, 1)
        slices = []
        for proc in range(mp.num_procs-1):
            slices.append(slice(proc * len_slice, (proc + 1) * len_slice))
        slices.append(slice((mp.num_procs-1)*len_slice, l_shape[0]))
        return slices

# @DebugDecorator.PrintReturn
# @DebugDecorator.PrintArgs
def shared_np_array(shape):
    """Form shared memory 1D numpy array"""
    from multiprocessing import Array
    arr_len = np.product(shape)
    shared_array_base = Array(ctypes.c_double, arr_len)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)
    return shared_array
