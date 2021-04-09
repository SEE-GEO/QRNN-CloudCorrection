#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:35:37 2020

@author: inderpreet

"""


import matplotlib.pyplot as plt
import numpy as np
import stats as S
from ici_mwi import iciData
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
test_file = "TB_ICI_test.nc"
output_file = 'Figures/error_distribution_QRNN-single.pdf'

binstep = 0.5
bins = np.arange(-20, 15, binstep)
iq = np.argwhere(quantiles == 0.5)[0,0]

#%% Plot error distributions for all ICI channels

fig, ax = plt.subplots(1, 3, figsize = [30, 10])
plt.subplots_adjust(wspace = 0.001)
for i,target in enumerate(targets):
    inChannels = np.array([target, 'I5V' , 'I6V', 'I7V', 'I8V', 'I9V', 'I10V', 'I11V'])
    data = iciData(test_file, 
                   inChannels, target, 
                   batch_size = batchSize)  

    i183, = np.argwhere(inChannels == target)[0]

# read QRNN    
    file = 'qrnn_ici_%s_%s_%s_single.nc'%(depth, width, target)
    print (file)
    qrnn = QRNN.load(file)
    y_pre, y_prior, y0, y, y_pos_mean = S.predict(data, qrnn, add_noise = True)
    im = np.abs(y_pre[:, iq] - y_prior[:, i183]) < 5.0
    hist_noise, hist_pre, hist_prior, hist_pos_mean, hist_pos_mean_5 = \
        S.calculate_all_histogram(y, y0, y_pre, y_prior, iq, bins, im, i183)
                                
                                
    center = (bins[:-1] + bins[1:]) / 2

    ax[i].plot(center, hist_noise[0], 'k', linewidth = 2.5)
    ax[i].plot(center, hist_prior[0], 'g', linewidth = 2.5)
    ax[i].plot(center, hist_pre[0],'b', linewidth = 2.5)
    ax[i].plot(center, hist_pos_mean_5[0], 'r', linewidth = 2.5)
    ax[i].set_yscale('log')
#    ax[i].set_yticklabels([])
#    ax[i].set_xticklabels([]) 
    ax[i].xaxis.set_minor_locator(MultipleLocator(1))
    ax[i].yaxis.set_minor_locator(MultipleLocator(5))
    ax[i].grid(which = 'both', alpha = 0.2)
    ax[i].set_title('Channel:%s'%target, fontsize = 28)

#    ax[i].set(ylim = [0, 1])
    
ax[0].set_ylabel('Occurence frequency [#/K]')
ax[1].set_xlabel('Deviation to noise free clear-sky [K]')
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
                            
(ax[2].legend(["Noise", "Uncorrected", "Predicted (all)", "Predicted (5K)"],
            prop={'size': 32}, frameon = False, bbox_to_anchor=(0.6, -0.12),ncol=4))                                
                                
                                
fig.savefig(output_file, bbox_inches = 'tight')                               
                                
#%% plot PDF of the predictions for ICI channels

fig, ax = plt.subplots(1, 3, figsize = [30, 10])
plt.subplots_adjust(wspace = 0.001)
bins = np.arange(230, 300, 1)

for i,target in enumerate(targets):
    inChannels = np.array([target, 'I5V' , 'I6V', 'I7V', 'I8V', 'I9V', 'I10V', 'I11V'])
    data = iciData(test_file, 
                   inChannels, target, 
                   batch_size = batchSize)  

    i183, = np.argwhere(inChannels == target)[0]

# read QRNN    
    file = 'qrnn_ici_%s_%s_%s_single.nc'%(depth, width, target)
    print (file)
    qrnn = QRNN.load(file)
    y_pre, y_prior, y0, y, y_pos_mean = S.predict(data, qrnn, add_noise = True)
                               
    h0 = np.histogram(y0, bins, density = True)
    h_p = np.histogram(y_pre[:, iq], bins, density = True)
    center = (bins[:-1] + bins[1:])/2

    ax[i].plot(center, h0[0], linewidth = 2.5, color = 'r')
    ax[i].plot(center, h_p[0], linewidth = 2.5, color =  'b')
    ax[i].set_yscale('log')

#    ax[i].set_xlabel('TB [K]')
    ax[i].xaxis.set_minor_locator(MultipleLocator(5))
    ax[i].grid(which = 'both', alpha = 0.2)
    ax[i].set_title('Channel:%s'%target, fontsize = 28)
#    ax[i].set(ylim = [0, 1])
    
ax[0].set_ylabel('Occurence frequency [#/K]')
ax[1].set_xlabel('TB[K]')
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
0.6
(ax[2].legend(["Simulated", "Predicted"],
            prop={'size': 32}, frameon = False, bbox_to_anchor=(0.1, -0.12),ncol=2))                                
                                
                                
fig.savefig('Figures/PDF_predictions_ICI.pdf', bbox_inches = 'tight')                                              
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
    
qrnn = QRNN.load(file)  