#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:35:37 2020

@author: inderpreet

this code plots the output of QRNN for few selected cases
"""


import matplotlib.pyplot as plt
import numpy as np
import stats as S
from ici_mwi import iciData
plt.rcParams.update({'font.size': 26})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")

#%% input parameters
depth     = 4
width     = 128
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
batchSize = 128

targets = ['I1V']
test_file = "TB_ICI_test.nc"

iq = np.argwhere(quantiles == 0.5)[0,0]

#%% Plot the output of QRNN, posteriori distribution

fig, ax = plt.subplots(1, 1, figsize = [12, 8])
x = np.arange(-3, 4, 1)
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
    im = np.where(np.logical_and(((y_pre[:, 3]).astype(int) == 270), \
                                 (np.abs(y_pre[:, 5] - y_pre[:, 1])> 10), 
                  (np.abs(y_pre[:, 6] - y_pre[:, 0])< 20)))[0]  
    for i in im[0:1]:
        ax.plot(quantiles,  y_pre[i, :], '-ro',  linewidth = 2.5)
        
    im = np.where(np.logical_and(((y_pre[:, 3]).astype(int) == 270),\
                                 (np.abs(y_pre[:, 6] - y_pre[:, 0])< 10)))[0]  
    for i in im[0:1]:
        ax.plot( quantiles, y_pre[i, :],  '-bo',  linewidth = 2.5)
        
    im = np.where(np.logical_and(((y_pre[:, 3]).astype(int) == 270),\
                                 (np.abs(y_pre[:, 6] - y_pre[:, 0])> 20)))[0]
   
    for i in im[0:1]:
        ax.plot(quantiles, y_pre[i, :],  '-ko',  linewidth = 2.5)   
        
ax.set_xlabel('Quantiles')
ax.set_ylabel('TB[K]')

ax.set_xticks(quantiles)
ax.set_xticklabels((r'$-3\sigma$', r'$-2\sigma$', r'$-1\sigma$', 0, \
                    r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'), rotation = 85, fontsize = 26)
  
ax.yaxis.set_minor_locator(MultipleLocator(5))
    
ax.grid(which = 'both', alpha = 0.2)

#(ax.legend(["Noise", "Uncorrected", "Predicted (all)", "Predicted (5K)"],
#           prop={'size': 22}, frameon = False, bbox_to_anchor=(0.6, -0.12),ncol=4))                                
                                
                                
fig.savefig('Figures/posterior_distribution_%s.pdf'%(targets[0]), bbox_inches = 'tight')                               
fig.savefig('Figures/posterior_distribution_%s.png'%(targets[0]), bbox_inches = 'tight')                                 