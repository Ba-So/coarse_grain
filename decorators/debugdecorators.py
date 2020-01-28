#!/usr/bin/env python
# coding=utf-8
import datetime

class TimeThis(object):
    def __init__(self, decorated):
        self._decorated = decorated
    def __call__(self, *args, **kwargs):
        self.before = datetime.datetime.now()
        print('Timing.')
        x = self._decorated(*args, **kwargs)
        self.after = datetime.datetime.now()
        self.diff = self.after - self.before
        print("before {} , after {}").format(self.before, self.after)
        print("Elapsed Time: {} min {} s".format(int(self.diff.total_seconds() // 60), self.diff.total_seconds() % 60))
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
