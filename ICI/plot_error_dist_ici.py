#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:35:37 2020

@author: inderpreet

this code plots the PDF of the predictions and errors of best estimate (median)
ICI channels

This script is used to plot Figure 9 of the article.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import stats as S
from ici import iciData
plt.rcParams.update({'font.size': 32})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")

#%% input parameters
depth     = 4
width     = 128
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
batchSize = 128

targets = ['I1V', 'I2V','I3V']
inpath  = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/ICI/")
test_file = os.path.join(inpath, "data/TB_ICI_test.nc")
output_file = 'Figures/error_distribution_QRNN-single.pdf'

binstep = 0.5
bins = np.arange(-20, 15, binstep)
iq = np.argwhere(quantiles == 0.5)[0,0]

#%% Plot error of best estimate for all ICI channels

fig, ax = plt.subplots(1, 3, figsize = [30, 10])
plt.subplots_adjust(wspace = 0.001)
for i,target in enumerate(targets):
    inChannels = np.array([target, 'I5V' , 'I6V', 'I7V', 'I8V', 'I9V', 'I10V', 'I11V'])
    data = iciData(test_file, 
                   inChannels, target, 
                   batch_size = batchSize)  

    i183, = np.argwhere(inChannels == target)[0]

# read QRNN    
    file = os.path.join(inpath, 'qrnn_output/qrnn_ici_%s_%s_%s_single.nc'%(depth, width, target))
    print (file)
    qrnn = QRNN.load(file)
    y_pre, y_prior, y0, y, y_pos_mean = S.predict(data, qrnn, add_noise = True)
    im = np.abs(y_pre[:, iq] - y_prior[:, i183]) < 5.0
#   im = (np.abs(y_pre[:, 0] - y_pre[:, 6] )<= 10.2) 
    print ('rejected obs', (1 - np.sum(im)/im.size)* 100)
    hist_noise, hist_pre, hist_prior, hist_pos_mean, hist_pos_mean_5, hist_filter  = \
        S.calculate_all_histogram(y, y0, y_pre, y_prior, iq, bins, im, i183)
                                
                                
    center = (bins[:-1] + bins[1:]) / 2

    ax[i].plot(center, hist_noise[0], 'k', linewidth = 2.5, label = "Noise")
    ax[i].plot(center, hist_prior[0], 'g', linewidth = 2.5, label = "All-sky")
    ax[i].plot(center, hist_pre[0],'b', linewidth = 2.5, label = "Predicted (All)")

    ax[i].plot(center, hist_pos_mean_5[0], 'r', linewidth = 2.5, label = "Predicted (5K)")
 #   ax[i].plot(center, hist_filter[0], 'r', linewidth = 2.5)
    ax[i].set_yscale('log')
#    ax[i].set_yticklabels([])
#    ax[i].set_xticklabels([]) 
    ax[i].xaxis.set_minor_locator(MultipleLocator(1))

    ax[i].grid(which = 'both', alpha = 0.2)
    ax[i].set_title('Channel:%s'%target, fontsize = 28)

#    ax[i].set(ylim = [0, 1])
    
ax[0].set_ylabel(r'Occurence frequency [K$^{-1}$]')
ax[1].set_xlabel('Deviation from NFCS simulations [K]')
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
                            
(ax[2].legend( prop={'size': 32}, frameon = False, bbox_to_anchor=(0.5, -0.12),ncol=4))                                
 
ax[0].annotate('(a)',
            xy=(0.075, .97), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=26)      
ax[1].annotate('(b)',
            xy=(.38, .97), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=26)                            
ax[2].annotate('(c)',
            xy=(.69, .97), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=26) 
                                
fig.savefig(output_file, bbox_inches = 'tight')                               
fig.savefig("Figures/ici_error_distribution.png", bbox_inches = 'tight')                                   