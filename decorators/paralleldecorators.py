#!/usr/bin/env python
# coding=utf-8
import ctypes
import numpy as np
from multiprocessing import Process, Queue, Array
from debugdecorators import TimeThis, PrintArgs, PrintReturn

class Mp(object):
    def __init__(self, switch, num_procs):
        self.num_procs = num_procs
        self.switch = switch

    def set_parallel_proc(self, state):
        self.switch = state

    def set_num_procs(self, num_procs):
        self.num_procs = num_procs

#global standard value
gmp = Mp(False, 2)

class ParallelNpArray(object):
    """do parallelisation using shared memory"""
    def __init__(self, mp=None):
        self._mp = mp
    def __call__(self, func):
        def _parall(*args, **kwargs):
            if self._mp.switch:
                slices = self.prepare_slices(args[0], self._mp)
                result_queue = []
                processes = []
                if hasattr(_parall, 'needs'):
                    args_sliced = []
                    # relies on full stacks being always in first positions.
                    # convention!
                    len_full = len(_parall.needs['full'])
                    for i in range(self._mp.num_procs):
                        args_i = list(args[:len_full])
                        for arg in args[len_full:]:
                            args_i.append(arg[slices[i]])
                        args_sliced.append(args_i)
                else:
                    args_sliced = []
                    for i in range(self._mp.num_procs):
                        for arg in args:
                            args_i.append(arg[slices[i]])
                        args_sliced.append(args_i)
                #initiate processes with the sliced args
                for xarg in args_sliced:
                    p = Process(target=func, args=xarg)
                    processes.append(p)
                    p.start()
                for p in processes:
                    p.join()
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
