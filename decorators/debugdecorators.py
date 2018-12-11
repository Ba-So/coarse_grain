#!/usr/bin/env python
# coding=utf-8
import datetime

class TimeThis(object):
    def __init__(self, decorated):
        self._decorated = decorated
    def __call__(self, *args, **kwargs):
        before = datetime.datetime.now()
        print('Timing.')
        x = self._decorated(*args, **kwargs)
        after = datetime.datetime.now()
        diff = after - before
        print("Elapsed Time: {}min {}s".format(diff.seconds // 60, diff.seconds % 60))
        return x

class PrintArgs(object):
    def __init__(self, decorated):
        self._decorated = decorated

    def __call__(self, *args, **kwargs):
        print("Name of func '{}':".format(self._decorated.__name__))
        print("-"*10)
        print("args: {}".format(args))
        print("-"*10)
        print("kwargs: {}".format(kwargs))
        print("\n")
        return self._decorated(*args, **kwargs)

class PrintReturn(object):
    def __init__(self, decorated):
        self._decorated = decorated

    def __call__(self, *args, **kwargs):
        out = self._decorated(*args, **kwargs)
        print("return value: {}".format(out))
        return out
