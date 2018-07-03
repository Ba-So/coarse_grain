#!/usr/bin/env python
# coding=utf-8

#update functions for global
# update for dicts??
import global_vars as gv

class Updater():

    def complete(self, varn, values):
        gv.globals_dict[varn] = values
        return None

    def up_entry(self, varn, values):
        # expects values to be dict
        gv.globals_dict[varn].update(values)
        return None

    def rm_entry(self, varn, values):
        # expects values to be dict
        for value in values:
            gv.globals_dict[varn].pop(value, None)
        return None

    def rm_all(self, varn):
        keys = []
        for key in gv.globals_dict[varn].iterkeys():
            keys.append(key)

        for key in keys:
            gv.globals_dict[varn].pop(key, None)
        return None

    def part(self, varn, values, indices):
        # expects values to be what?
        for key, array in values.iteritems():
            gv.globals_dict[varn][key][indices] = array
        return None

    def up_mp(self, values):
        gv.mp.update(values)
        return None

    def append(values):
        gv.mp['out'].append(values)

    def scrub():
        gv.mp['out'] = []


