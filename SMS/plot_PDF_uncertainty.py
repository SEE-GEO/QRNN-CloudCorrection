#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:54:58 2020

@author: inderpreet

plot error distributions in uncertainty bins
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import netCDF4
plt.rcParams.update({'font.size': 32})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy.stats import skew

from typhon.retrieval.qrnn import set_backend, QRNN

from aws_test_data import awsTestData

set_backend("pytorch")
plt.rcParams["text.usetex"]

#%%
def PDF_uncertainty_bins(y_pre, y0, ulim):
    dtb =(y_pre[:, 3] - y0)
    uncertain = y_pre[:, 5] - y_pre[:, 1]
 #I1V
    #ulim = [3, 4] #I2V
    #ulim = [1, 1.5 ]#I3V
    
    im = uncertain < ulim[0]
    print (np.sum(im))
    bins = np.arange(-12.5, 15., 0.8)
    hist0 = np.histogram(dtb[im], bins, density = True)
    
    
    im = np.logical_and((uncertain < ulim[1]), ( uncertain >= ulim[0]) )
    hist1 = np.histogram(dtb[im], bins, density = True)
 
#    im = np.logical_and((uncertain < ulim[2]), ( uncertain >= ulim[1]) )
#    hist2 = np.histogram(dtb[im], bins, density = True)

    
    im = uncertain >=ulim[1]
    hist2 = np.histogram(dtb[im], bins, density = True)
    
    
    return hist0[0], hist1[0],  hist2[0], bins
#%%
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

#%% input parameters
depth     = 4
width     = 128
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
batchSize = 128

channels =  ['C32', 'C33', 'C34']
C=  ['AWS-32', 'AWS-33', 'AWS-34']

binstep = 0.5
bins = np.arange(-20, 15, binstep)
iq = np.argwhere(quantiles == 0.5)[0,0]
ulim = [5, 8]
inpath = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/AWS/")
#%% 

fig, ax = plt.subplots(1, 3, figsize = [30, 10])
plt.subplots_adjust(wspace = 0.001)

for i, c183 in enumerate(channels):

    inChannels = np.array([c183, 'C41', 'C42', 'C43', 'C44'])

    test_data = awsTestData(os.path.join(inpath, "data/TB_AWS_m60_p60_noise_four_test.nc"),
                            inChannels, option = 4)     
    qrnn = QRNN.load(os.path.join(inpath, "qrnn_data/qrnn_4_128_%s.nc"%c183))
    y_pre, y_prior, y0, y, y_pos_mean = predict(test_data, qrnn)
    
    hist0, hist1, hist2, bins = PDF_uncertainty_bins(y_pre, y0, ulim)
    center = (bins[:-1] + bins[1:])/2
    ax[i].plot(center, hist0, 'k', linewidth = 2.5)
    ax[i].plot(center, hist1, 'r', linewidth = 2.5)
    ax[i].plot(center, hist2, 'b', linewidth = 2.5)
#    ax[i].plot(center, hist3, 'g', linewidth = 2.5)
    ax[i].xaxis.set_minor_locator(MultipleLocator(5))
    ax[i].grid(which = 'both', alpha = 0.4)
    
    ax[i].set_ylim(0, 1)
    ax[i].set_title("Channel:%s"%C[i], fontsize = 30)
ax[0].set_ylabel('Occurence frequency [#/K]', fontsize = 32)
ax[1].set_xlabel('Deviation to noise free clear-sky [K]', fontsize = 32)
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[1].legend([  '0 - ' + str(ulim[0]) + ' K',
            str(ulim[0]) +' - ' + str(ulim[1]) + ' K',
             '> ' + str(ulim[1]) + ' K' ], title = "uncertainty bins (2$\sigma$)", prop={'size': 24}, \
             frameon = False, bbox_to_anchor=(1, 1.0), ncol=3)



fig.savefig('Figures/PDF_uncertainty_bins_QRNN-single.pdf')                               
                                