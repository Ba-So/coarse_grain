#!/usr/bin/env python
# coding=utf-8
import sys

# Basic construction of my new class structured vision

def requires_variables(needed):
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
        x = original_function(*args, **kwargs)
        after = datetime.datetime.now()
        print("Elapsed Time: {}".format(after - before))
        return x
    return decorated

