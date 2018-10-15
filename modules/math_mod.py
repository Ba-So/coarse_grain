#!/usr/bin/env python
# coding=utf-8

class math_mod():

    def avg_bar(self, data):
        """computes the bar average of data"""
        vals = np.sum(self.data[data], 0)
        self.data_bar[data] = np.divide(vals, self.coarse_area)
        return

    def avg_hat(self, data):
        """computes the hat average of data"""
        vals = np.sum(np.multiply(self.data[data], self.rho, 0))
        self.data_hat[data] = np.divide(vals, (self.coarse_area, self.data_bar['rho']))
        return

    def fluctsof(self, data):
        """computes deviations from local mean"""
        self.data_flucts[data] = np.subtract(self.data_bar[data], self.data[data])
        return


