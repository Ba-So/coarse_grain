#!/usr/bin/env python
# coding=utf-8

import os
import sys
import glob
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import argparse

import modules.cio as cio
import modules.math_mod as mo
import modules.phys_mod as po
import main as ma


def plot_histogram_2(data, name, lev=None, time=None):
    print("level {}".format(lev))
    print("time {}".format(time))
    matplotlib.rcParams.update({'font.size':20})

    mu = np.mean(data)
    average= np.average(data)
    sigma = np.std(data)
    d = np.abs(data - average)
    mdev = np.median(d)
    print('shape: {}'.format(np.shape(data)))
    if lev and time:
        data = data[:, time, lev]
    elif lev:
        data = data[:, :, lev]
    elif time:
        data = data[:, time, :]
    data = data.ravel()


    legend_text = '\n'.join((
        r'$\mu={:+.2E}$'.format(mu, ),
        r'$\sigma = {:+.2E}$'.format(sigma, )))
    props = {
        'boxstyle' : 'round',
        'facecolor' : 'wheat',
        'alpha' : 0.5,
    }
    fig, ax = plt.subplots()

    weights = np.ones_like(data)/float(len(data))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()

    n, bins, patches = ax.hist(data, weights=weights, bins=50, density=True, log=True)
    #y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    #ax.plot(bins, y, '--')

    ax.text(
        0.5, 0.95,
        legend_text,
        transform= ax.transAxes,
        fontsize = 20,
        verticalalignment = 'top',
        horizontalalignment = 'right',
        bbox= props
    )

    oname = name
    if lev:
        oname = oname + '_l{}'.format(lev)
    else:
        pass

    if time:
        oname = oname + '_t{}'.format(time)
    else:
        pass

    plt.savefig('bilder/histogram_{}'.format(oname),dpi=73)
    plt.show()

def plot_timeline(data, name):
    data = np.average(data, (1,2))
    plt.figure(figsize = (8,6), dpi=80)
    plt.subplot(111)
    X = np.linspace(0, 14, len(data), endpoint = True)
    plt.plot(X, data, linewidth=2.5, linestyle='-')
    plt.xlim(0, 14)
    plt.xticks([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_title(r'time series of global average of {}'.format(name))

    plt.savefig('bilder/timeline_{}'.format(name), dpi=80)
    plt.show()

def plot_slice(data, name):
    None

def collect_files(path):
    files = [
        n for n in
        sorted(glob.glob(path + '/*_refined*.nc')) if
        os.path.isfile(n)
        and 'grid' not in n
        and 'disc' not in n
    ]
    grid = [
        n for n in
        glob.glob(path + '/*grid*refined*.nc') if
        os.path.isfile(n)
    ]

    return files, grid


class Run(object):
    def __init__(self, args):
        self.path = args.path
        self.file = args.file
        self.out_path = args.out_path
        self.IO = cio.IOcontroller(self.path, data=self.file, out=False)
        self.vars = args.vars
        self.lev=args.lev
        if isinstance(args.lev_start, int) and isinstance(args.lev_end, int):
            self.lev_range=[args.lev_start, args.lev_end]
            if not args.lev_start < args.lev_end:
                sys.exit('invalid level range: {}').format(args.lev_range)
        else:
            self.lev_range=None
        self.time=args.time
        if isinstance(args.time_start, int) and isinstance(args.time_end, int):
            self.time_range=[args.time_start, args.time_end]
            if not args.time_start < args.time_end:
                sys.exit('invalid time range: {}').format(args.time_range)
        else:
            self.time_range=None
    def run(self):
        for var in self.vars:
            if not self.IO.isin('data', var):
                print('{} not found in dataset'.format(var))
                break
            print('defining bins ...')
            self.define_bins(var)
            print('computing histograms ...')
            self.histogram(var)
            print('computing statistics ...')
            self.statistics(var)
            print('plotting histograms ...')
            self.plot_histogram(var)
            print('done')

    def getlabels(self, var):
        if var == 'KIN_TRANS':
            ylabel=r'$log(P(\tilde \epsilon _M$))'
            xlabel=r'$\tilde \epsilon _M\ [J\ s^{-1}\ m^{-3}]$'
            what=r'\tilde \epsilon _M'
        elif var == 'INT_TRANS':
            ylabel=r'$log(P(\tilde \epsilon _T$))'
            xlabel=r'$\tilde \epsilon _T\ [J\ s^{-1}\ m^{-3}]$'
            what=r'\tilde \epsilon _T'
        elif var == 'T_FRIC':
            ylabel=r'$log(P(\tilde \sigma _M$))'
            xlabel=r'$\tilde \sigma _M\ [J\ s^{-1}\ m^{-3}]$'
            what=r'\tilde \sigma _M'
        elif var == 'INT_TRANS':
            ylabel=r'$log(P(\tilde \sigma _T$))'
            xlabel=r'$\tilde \sigma _T\ [J\ s^{-1}\ m^{-3}]$'
            what=r'\tilde \sigma _T'
        else:
            ylabel=r'$log(P({}))$'.format(var)
            xlabel=r'${}$'.format(var)
            what=r'\tilde random'
        return xlabel, ylabel, what

    def bin_min_max_num(self, data):
        dat_min = np.min(data)
        dat_max = np.max(data)
        dat_num = np.size(data)
        return dat_min, dat_max, dat_num

    def define_bins(self, var):
        # comparisons with None always come out False.
        bin_min = 10000
        bin_max = -10000
        self.total_num = 0
        # get max/min & total number of values, consecutively for data files
        for xfile in self.IO.datafiles:
            # load
            data = self.IO.load(xfile, var, self.time, self.lev, self.time_range, self.lev_range)
            int_min, int_max, dat_num = self.bin_min_max_num(data)
            del data
            # compare with prev values
            if int_min < bin_min:
                bin_min = int_min
            if int_max > bin_max:
                bin_max = int_max
            self.total_num += dat_num
        # get number of bin by square-root formula
        bin_num = int(np.ceil(np.sqrt(self.total_num)))
        if bin_num >= 100:
            bin_num = 100
        # get binsize / step between bins.
        step = (bin_max - bin_min) / bin_num
        # define binning, numpy histograms takes individual bin bounds
        # [1, 2, 3, 4] as bins => [1,2) [2,3) .. as bins.
        self.bins = [bin_min + i * step for i in range(bin_num + 1)]

    def histogram(self, var):
        # create empty histogram for compounding
        total_histogram = np.zeros(len(self.bins)-1)
        # sequentially evaluate datasets
        for xfile in self.IO.datafiles:
            data = self.IO.load(xfile, var, self.time, self.lev, self.time_range, self.lev_range)
            # compute histogram from data using precomputed bins
            int_hist = np.histogram(data, bins=self.bins, density=False)[0]
            del data
            total_histogram = np.add(total_histogram, int_hist)
        # norm the histogram, to reflect actual probabilites.
        self.pdf = np.divide(total_histogram, self.total_num)
        print('sanity check sum(pdf): {}'.format(np.sum(self.pdf)))

    def statistics(self, var):
        self.mean = self.arithmetic_mean(var)
        self.stdev = self.std_dev(var, self.mean)

    def arithmetic_mean(self, var):
        num = 0
        mean = 0
        for xfile in self.IO.datafiles:
            data = self.IO.load(xfile, var, self.time, self.lev, self.time_range, self.lev_range)
            num += data.size
            mean += np.sum(data)
            del data
        return mean / num

    def std_dev(self, var, mean):
        num = 0
        std = 0
        for xfile in self.IO.datafiles:
            data = self.IO.load(xfile, var, self.time, self.lev, self.time_range, self.lev_range)
            num += np.size(data)
            std += np.sum(np.square(np.subtract(data, mean)))
            del data
        return np.sqrt(std / (num - 1))

    def plot_histogram(self, var):
        ''' plots histograms.
            optionally for spcific levels or time steps or slices of them
            '''

        width = (self.bins[1] - self.bins[0])
        center = np.divide(np.add(self.bins[:-1], self.bins[1:]), 2)
        matplotlib.rcParams.update({'font.size':20})
        xlabel,ylabel,what = self.getlabels(var)

        legend_text = '\n'.join((
            r'$\bar{}={:.2E}$'.format(what, self.mean, ),
            r'$Var({})= {:.2E}$'.format(what, self.stdev, )))
        props = {
            'boxstyle' : 'round',
            'facecolor' : 'wheat',
            'alpha' : 0.5,
        }
        fig, ax = plt.subplots()
        ax.text(
            1, 0.95,
            legend_text,
            transform= ax.transAxes,
            fontsize = 16,
            verticalalignment = 'top',
            horizontalalignment = 'right',
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        plt.axvline(x=0,color='red')

        plt.bar(center, self.pdf, align='center', width=width)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.yscale('log')
        plt.tight_layout()

        oname = var
        if self.lev:
            oname = oname + '_l{}'.format(self.lev)
        elif not self.lev_range == None:
            oname = oname + '_l{}-{}'.format(self.lev_range[0], self.lev_range[1])
        else:
            pass

        if not self.time==None:
            oname = oname + '_t{}'.format(self.time)
        elif not self.time_range == None:
            oname = oname + '_t{}-{}'.format(self.time_range[0], self.time_range[1])
        else:
            pass

        plt.savefig(os.path.join(self.out_path,'histogram_{}.eps'.format(oname)),dpi=73,format='eps')
        plt.show()
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Data from Coarse Grained.')
    parser.add_argument(
        '-p',
        dest = 'path',
        type = str,
        help='a string specifying the path to experiment'
    )
    parser.add_argument(
        '-op',
        dest = 'out_path',
        default = ''
        type = str,
        help='a string specifying the output path of figures'
    )
    parser.add_argument(
        '-f',
        dest = 'file',
        type = str,
        help='a string specifying the name of experiment outfiles'
    )
    parser.add_argument(
        '-l',
        dest = 'lev',
        default = None,
        type = int,
        help = 'level to be investigated',
    )
    parser.add_argument(
        '-ls',
        dest = 'lev_start',
        default = None,
        type = int,
        help = 'start of level range to be investigated',
    )
    parser.add_argument(
        '-le',
        dest = 'lev_end',
        default = None,
        type = int,
        help = 'end of level range to be investigated',
    )
    parser.add_argument(
        '-t',
        dest = 'time',
        default = None,
        type = int,
        help = 'time to be investigated',
    )
    parser.add_argument(
        '-ts',
        dest = 'time_start',
        default = None,
        type = int,
        help = 'start of time range to be investigated',
    )
    parser.add_argument(
        '-te',
        dest = 'time_end',
        default = None,
        type = int,
        help = 'end of time range to be investigated',
    )
    parser.add_argument(
        '-v',
        dest = 'vars',
        type = str,
        nargs = '+',
        help = 'names of variables to be analysed and plotted'
    )
    args = parser.parse_args()
    print(
        'analysing and plotting for variables: {}'.format(args.vars)
    )
    print(args)
    runner = Run(args)
    runner.run()
#    runner.run(lev=43)
#    runner.run()
#    for time in range(9):
#        runner.run(lev=43, time=time)
