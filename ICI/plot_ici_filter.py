#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:57:41 2020

@author: inderpreet

plot the average deviations from different filterin thresholds
"""


import matplotlib.pyplot as plt
import numpy as np
import stats as S
from ici import iciData
plt.rcParams.update({'font.size': 26})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")
import stats
#%%
def read_qrnn(file, inChannels, target):

    data = iciData(test_file, 
                   inChannels, target, 
                   batch_size = batchSize)  



# read QRNN    
#    file = 'qrnn_ici_%s_%s_%s_single.nc'%(depth, width, target)
#    print (file)
    qrnn = QRNN.load(file)
    y_pre, y_prior, y0, y, y_pos_mean = S.predict(data, qrnn, add_noise = True)
    
    return y_pre, y_prior, y0, y, y_pos_mean


#%% input parameters
depth     = 4
width     = 128
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
batchSize = 128

targets = ['I1V', 'I2V', 'I3V']
test_file = "TB_ICI_test.nc"

iq = np.argwhere(quantiles == 0.5)[0,0]

filters = np.arange(5, 0, -0.5)
c = ['r', 'b', 'k']
linetype = [':', '--', '-']
#%%
fig, ax = plt.subplots(1, 1, figsize = [8,8]) 
ax2 = ax.twinx() 
for i,target in enumerate(targets):
    inChannels_single = np.array([target, 'I5V' , 'I6V', 'I7V', 'I8V', 'I9V', 'I10V', 'I11V'])
    file_single = 'qrnn_ici_%s_%s_%s_single.nc'%(depth, width, target)
    print (file_single)
    i183, = np.argwhere(inChannels_single == target)[0]
    
    y_pre, y_prior, y0, y, y_pos_mean = read_qrnn(file_single, inChannels_single, target )
    
    bias = ()
    rej = []
    for j in filters:
            im = np.abs(y_pre[:, 3] - y_prior[:, i183]) <= j
            bias      += stats.calculate_bias(y_prior, y0, y, y_pre[:, 3], im, i183)
            rej.append((1 - np.sum(im)/im.size)* 100)

    bias  = np.array(list(bias))
#    rej = list(rej)    
    color = 'tab:red'
    ax.plot( bias[[4, 9, 14, 19, 24, 29, 34, 39, 44, 49]], color = color,\
            linewidth = 2.5, linestyle = linetype[i])
    ax.set_ylim(-0.35, 0)


 # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.plot(rej, linestyle = linetype[i], color = color, \
             linewidth = 2.5)
    
    ax2.set_ylim(37, 0)
    
color = 'tab:blue'    
ax2.set_ylabel('% Rejected', color=color)  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor=color)

color = 'tab:red'    
ax.set_ylabel('Average bias [K]', color=color)  # we already handled the x-label with ax1
ax.tick_params(axis='y', labelcolor=color)

ax.set_xlabel('Filtering threshold [K]')  
ax.set_xticks(np.arange(0, 10, 2))
ax.set_xticklabels(filters[[0, 2, 4, 6, 8]])  

ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.grid(which = 'both', alpha = 0.2)

(ax.legend(targets,
            prop={'size': 22}, frameon = False, bbox_to_anchor=(1., -0.12), ncol = 3))  
(ax2.legend(['     ', '     ', '     '],
            prop={'size': 22}, frameon = False, bbox_to_anchor=(1, -0.14), ncol = 3)) 

fig.savefig("Figures/different_filtering_thresholds.pdf", bbox_inches = 'tight')