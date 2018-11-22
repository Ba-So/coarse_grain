#!/usr/bin/env python
# coding=utf-8
import sys

# Basic construction of my new class structured vision

def ensure_inputs(needed):
    '''checks for presence of necessary variables'''
    def decorator(fn):
        def decorated(*args, **kwargs):
            present = [need in args[0] for need in needed]
            if all(present):
                return fn(*args, **kwargs)
            raise Exception(
                "Necessary variables missing {}".format(
                    [need for need in needed if need not in args[0]]
                )
            )
        return decorated
    return decorator

def time_this(fn):
    def decorated(*args, **kwargs):
        import datetime
        before = datetime.datetime.now()
        x = fn(*args, **kwargs)
        after = datetime.datetime.now()
        print("Elapsed Time: {}".format(after - before))
        return x
    return decorated

def multiprocess(fn):
    def decorated(*args, **kwargs):
        if kwargs.get('mp'):
            item_list = []
            key_list = []
            for item, key in kwargs['rundict']:
                item_list.append(item)
                key_list.append(item)

            from multiprocessing import Pool
            pool = Pool(processes = num_procs)
            result = np.array(pool.map(partial(recombine_dict(fn), keys=key_list), item_list)
            out = {}
            for i, item in enumerate(kwargs['name']):
                out.update({item : result[:, i,]})
            return out
        else:
            return fn(*args, **kwargs)
    return decorated

def recombine_dict(fn):
    def decorated(*args, **kwargs):
        rec_dict = {}
        for item,i in enumerate(item_list):
            rec_dict[key_list[i]] = item
        return fn(rec_dict)



