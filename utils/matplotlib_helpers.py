# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import numpy as np
import os
import sys

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



#-------------------------------------------------------------------------
# data = tuple of data matrices
# legend = tuple of labels
# each tuple entry is plotted into a subplot
# legend entries for each subplot are comma-separated entries in the name tuple

def draw_subplots(data, legend, filename=None):

    if filename is None:
        plt.ion()                       #interactive, non-blocking plots
        fig = plt.gcf()
        if fig is None:
            fig = plt.figure()
        else:
            fig.clf()
    else:
        plt.switch_backend('agg')
        plt.ioff()
        fig = plt.figure()

    #define custom color wheel
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    #create empty subplots
    fig, axes = plt.subplots(nrows=len(data), ncols=1)
    fig.set_size_inches(10, min(2*len(data),10))

    #use vertical padding of 2.5x the font size between the subplots to avoid overlapping labels
    fig.tight_layout(h_pad=2.5, rect=(0,0,1,0.97))
   
    #each sublabels,data tuple goes into a subplot
    for i, x in enumerate(data):
        ax = plt.subplot(len(data), 1, i+1)

        if x.ndim==1:
            x = x[:,np.newaxis]
        elif x.ndim==2 and x.shape[0]<x.shape[1]:
            x = x.T
        elif x.ndim>2:
            print('x.ndim must be less than 3')
            quit(0)

        #extract labels for each subplot
        labels = legend[i].split(',')
        c_idx = 0
        for j in range(x.shape[1]):
            if j<len(labels):
                label = labels[j]
            else:
                label = None

            #draw each column of x with a different color and label
            ax.plot(x[:,j], color=colors[c_idx], label=label)
            c_idx = (c_idx+1)%len(colors)

        #set axis limits and grid
        ax.set_xlim([0, x.shape[0]])
        ax.set_ylim([np.floor(np.amin(x)), np.ceil(np.amax(x))])
        ax.grid(color='k', linestyle='--', linewidth=0.25)
        #insert labels above each subplot
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(labels), mode="expand", borderaxespad=0.)


    if filename is None:
        plt.pause(0.001)
    else:
        fig.savefig(filename, dpi=100)




#-------------------------------------------------------------------------
# data = tuple of data matrices
# legend = tuple of labels
# clim = color limits (min, max)
def draw_subpcolor(data, legend, clim, filename):

    colors = [(1,0,0), (1,1,1), (0,0,1)]              # R -> W -> B
    cmap = LinearSegmentedColormap.from_list('meow', colors, N=64)

    plt.switch_backend('agg')
    plt.ioff()
    fig = plt.figure()

    #create empty subplots
    fig, axes = plt.subplots(nrows=len(data), ncols=1)
    fig.set_size_inches(10, min(5*len(data),20))

    #use vertical padding of 2x the font size between the subplots to avoid overlapping labels
    fig.tight_layout(h_pad=2, rect=(0,0,0.99,0.97))

    #each legend,data tuple goes into a subplot
    for i, x in enumerate(data):
        ax = plt.subplot(len(data), 1, i+1)
        plt.pcolor(x, cmap=cmap)
        plt.title(legend[i])
        ax.set_xlim([0, x.shape[1]])
        ax.set_ylim([0, x.shape[0]])
        plt.colorbar(aspect=20, fraction=0.05)
        plt.clim(*clim)

    fig.savefig(filename, dpi=200)




# ---------------------------------------------------------------------
def pcolor(x, filename='pcolor', x_min=None, x_max=None):

    plt.switch_backend('agg')

    path, name = os.path.split(filename)
    if not os.path.exists(path) and path != '':
        os.makedirs(path)

    # set font
    font = {'weight': 'bold', 'size': 12}
    matplotlib.rc('font', **font)

    plt.ioff()
    fig = plt.figure()

    plt.pcolor(x.T, cmap='jet')
    if x_min is not None and x_max is not None: plt.clim(x_min, x_max)
    plt.colorbar()
    plt.xlabel('nfram')
    plt.ylabel('nbin')
    plt.title(name)

    # save figure, scaled*1.5
    plt.savefig(filename, dpi=fig.dpi*1.5)
    plt.close(fig)

