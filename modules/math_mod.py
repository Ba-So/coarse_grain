#!/usr/bin/env python
# coding=utf-8

def avg_bar(data, c_area, ret):
    """computes the bar average of data"""
    vals = np.sum(self.data[data], 0)
    ret[:] = np.divide(vals, self.coarse_area)

def avg_hat(data, c_area, rho, rho_bar, ret):
    """computes the hat average of data"""
    vals = np.sum(np.multiply(data, rho, 0))
    ret[:] = np.divide(vals, (c_area, rho_bar))

def fluctsof(data, data_avg, ret):
    """computes deviations from local mean"""
    ret[:] = np.subtract(data_avg, data)

def gradient():
    """computes the gradient of a quantity using the information in gradient_nfo"""

