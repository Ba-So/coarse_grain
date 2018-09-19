#!/usr/bin/env python
# coding=utf-8
import update as up
import global_vars as gv

if __name__ == '__main__':
    update = up.Updater()
    update.up_entry('data_run', {'dyad' : []})
    update.append_entry('data_run', {'dyad': [1,2,3]})
    update.append_entry('data_run', {'dyad': [1,2,3]})
    print(gv.globals_dict['data_run']['dyad'])

