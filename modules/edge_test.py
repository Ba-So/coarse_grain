#!/usr/bin/env python
# coding=utf-8
import numpy as np

def arc_len(p_x, p_y):
    '''computes the length of a geodesic arc on a sphere (in radians)
    using the haversine formula. Is precise for distances smaller than
    half the circumference of earth.
    p_x : lon, lat
    p_y : lon, lat
    out d in radians'''
    dlon = p_y[0] - p_x[0]
    dlat = p_y[1] - p_x[1]

    num = (np.cos(p_y[1]) * np.sin(dlon))**2 + (np.cos(p_x[1]) * np.cos(p_y[1]) - np.sin(p_x[1]) * np.cos(p_y[1]) * np.cos(dlon))**2
    denum = np.sin(p_x[1]) * np.sin(p_y[1]) + np.cos(p_x[1]) * np.cos(p_y[1]) * np.cos(dlon)
    d_2 = np.arctan(np.sqrt(num)/denum)
    hav = np.sin(dlat / 2)**2 + np.cos(p_x[1]) * np.cos(p_y[1]) * np.sin(dlon / 2)**2
    d = 2 * np.arcsin(np.sqrt(hav))
    return d, d_2

if __name__ == '__main__':
    p_x = [np.pi/4 ,np.pi/4 - 0.1]
    p_y = [np.pi/4 ,np.pi/4]
    print(arc_len(p_x, p_y))
    print(arc_len(p_y, p_x))
    p_x = [-np.pi+0.003 ,np.pi/4]
    p_y = [np.pi-0.003 ,np.pi/4]
    print(arc_len(p_x, p_y))
    print(arc_len(p_y, p_x))
    p_x = [np.pi/2 - np.pi ,np.pi/2-0.05]
    p_y = [np.pi/2 ,np.pi/2-0.05]
    print(arc_len(p_x, p_y))
    print(arc_len(p_y, p_x))


