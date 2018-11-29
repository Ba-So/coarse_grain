#!/usr/bin/env python
# coding=utf-8
'''
data_structs:
    Due to the parallelization it became neccessary that individual
    data points know their associated cellnumber, and, while we're at it,
    their lat lon coordinates. The structure is a list of dicts.
    >Waste full of memory, better pass array /w associated cell numbers?
    >Maybe have this for better readability?
    Form: data = [{},{},...,{},{}]
    The Problem is conversion between them.
    Array of array doesn't add any value since i may as well pass sevaral,
    just decreases readability.

    Research:
        So looking turned out, that dicts consume way more memory in python.
        I'd like to reduce that. Therefore I'm going to screw them out of my
        code entirely, except for tiny things. Turns out having dictionaries
        of stuff is expensivaly unneccesary.

        Probably i'm going to throw readability over board. and just go array
        of arrays? This is just a rundown of what I put where.
    '''

grad_nfo = [len(ncells)] [len(4)] [[len(2)], [len(members)], [len(members)]]
#           ncells      neighbors centercord member_idx      member_rad
# neighbors: neighboring coordinates for gradient comp
# member_idx: ncell index of members of neighboring patch
# member_rad: distance of member from center
coordinates = [len(ncells)] [len(2)]
cell_area = [len(ncells)]
