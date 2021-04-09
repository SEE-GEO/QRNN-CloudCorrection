#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:32:49 2020

@author: inderpreet

Plot the average uncertainty in different ckoud impact bins for all MWI channels
"""
import matplotlib.pyplot as plt
import numpy as np
import stats as S
from ici_mwi_alone import iciData
plt.rcParams.update({'font.size': 26})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")
import stats
from tabulate import tabulate

#%%
def read_qrnn(file, inChannels, target):

    data = iciData(test_file, 
                   inChannels, target, 
                   batch_size = batchSize)  

    qrnn = QRNN.load(file)
    y_pre, y_prior, y0, y, y_pos_mean = S.predict(data, qrnn, add_noise = True)
    
    return y_pre, y_prior, y0, y, y_pos_mean

def predict(test_data, qrnn):
    """
    predict the posterior mean and median
    """
    x = (test_data.x - test_data.mean)/test_data.std

    y_pre = qrnn.predict(x.data)
    y_prior = test_data.x
    y0 = test_data.y0
    y = test_data.y
    y_pos_mean = qrnn.posterior_mean(x.data)
    
    return y_pre, y_prior, y0, y, y_pos_mean

def bin_uncertainty(y_pre, y_prior, y0):
    """
    calculate average uncertainty in cloud impact bins

    Parameters
    ----------
    y_pre : array containing predicted quantiles
    y_prior : array containing measurement data
    y0 : array containing NFCS simulations
    Returns
    -------
    mean_ci, bins : the average uncertainty (mean_ci) in bins

    """

 #   dtb = y_pre[:, 3]- y_prior
    dtb =  y_prior - y0
    print (dtb.min(), dtb.max())
    ci = y_pre[:, 5] - y_pre[:, 1]
    bins = np.arange(-100, 0, 2)
    A = np.digitize(dtb, bins)
    mean_ci = []
    for i in range(49):
        ix = np.where(A == i)[0]
        if np.sum(ix) >0:
            mean_ci.append(np.mean(ci[ix]))
        else:
            mean_ci.append(np.nan)
#    plt.plot(bins[:-1], mean_ci, '--')
    return mean_ci, bins

def bin_errors(y_pre, y_prior, y0):
 #   dtb = y_pre[:, 3]- y_prior
    dtb =  y_prior - y0
    er = y_pre[:, 3] - y0
    bins = np.arange(-100, 0, 2)
    A = np.digitize(dtb, bins)
    mean_ci = []
    for i in range(49):
        ix = np.where(A == i)[0]
        if np.sum(ix) >0:
            mean_ci.append(np.mean(er[ix]))
        else:
            mean_ci.append(np.nan)
#    plt.plot(bins[:-1], mean_ci, '--')
    return mean_ci, bins


def histogram(ci, bins):
    hist = np.histogram(ci, bins, density = True)
    return hist[0]

if __name__ == "__main__":

    #%% input parameters
    depth     = 4
    width     = 128
    quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
    batchSize = 128

    targets = [ 'I1V', 'I2V', 'I3V',]
    sat = ['ici', 'ici_mwi', 'ici_mwi', 'ici', 'ici']
    targets_mwi = ['MWI-14', 'MWI-15', 'MWI-16', 'MWI-17', 'MWI-18'] 
    test_file = "TB_ICI_mwi_test.nc"

    iq = np.argwhere(quantiles == 0.5)[0,0]
    #%%
    colors = ['r', 'b', 'g', 'k', 'm']    
    channels = ['I1V','MWI-15', 'MWI-16', 'I2V', 'I3V']
    fig, ax = plt.subplots(1, 1, figsize = [8,8])
    fig1, ax1 = plt.subplots(1, 1, figsize = [8,8])
    ci_bins = np.arange(0, 40, 1)
    #ci_bins = np.arange(-30, 30, 1)
    center_ci_bins = (ci_bins[:-1] + ci_bins[1:])/2
    bins = np.arange(-100, 0, 2)
    center = (bins[:-1] + bins[1:])/2
    #%%
    for i, target in enumerate(channels):

    #    inChannels_single = np.array([target, 'I5V' , 'I6V', 'I7V', 'I8V', 'I9V', 'I10V', 'I11V'])
    #    file_single = 'qrnn_%s_%s_%s_%s_single.nc'%(sat[i], depth, width, target)

        inChannels_single = np.array(['I1V', 'I2V', 'I3V', 'MWI-15', 'MWI-16'])
        file_single = 'qrnn_ici_%s_%s_%s_mwi-alone.nc'%(depth, width, target)

        print (file_single)
        i183, = np.argwhere(inChannels_single == target)[0]
        print (i183)
        y_pre, y_prior, y0, y, y_pos_mean = read_qrnn(file_single, \
                                                      inChannels_single, target )
    #    im = np.abs(y_pre[:, 3] - y_prior[:, i183]) <= 5
        im = np.abs(y_prior[:, i183] - y0) <= 15
        print ((1 - np.sum(im)/im.size)* 100)

        mean_ci, bins = bin_uncertainty(y_pre[:], y_prior[:, 0], y0[:])
        ax.plot(center, mean_ci, linewidth = '2.5' , color = colors[i])

        ax.xaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.grid(which = 'both', alpha = 0.2)
        ci = y_pre[:, 5] - y_pre[:, 1]
        hist = histogram(ci, ci_bins)




        ax1.plot(center_ci_bins, hist, color = colors[i], linestyle = '--', \
                 linewidth = 2.5)

    #    ax1.set(ylim = [0, 1])

    ax.set_ylabel(r'Mean confidence interval ($2\sigma$) [K]')
    ax.set_xlabel('Cloud impact [K]')
    ax1.xaxis.set_minor_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(10))
    ax1.grid(which = 'both', alpha = 0.2)
    ax1.set_yscale('log')                           
    (ax.legend(targets_mwi,
                prop={'size': 20}, frameon = False))      
    (ax1.legend(targets_mwi,
                prop={'size': 20}, frameon = False))                             
    ax1.set_ylabel(r'Occurence frequency [#/K]')
    ax1.set_xlabel('Uncertainty [K]')                                

    fig.savefig('Figures/cloud_impact_uncertainty_MWI.pdf', bbox_inches = 'tight')         
