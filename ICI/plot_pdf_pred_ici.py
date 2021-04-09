#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:35:37 2020

@author: inderpreet

this code plots the PDF of the predictions and errors of best estimate (median)
ICI channels
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
#targets = ['I2V']
test_file = "TB_ICI_test.nc"
output_file = 'Figures/error_distribution_QRNN-single.pdf'

binstep = 0.5
bins = np.arange(-20, 15, binstep)
iq = np.argwhere(quantiles == 0.5)[0,0]
          
                                
#%% plot PDF of the predictions for ICI channels
N = len(targets)
fig, ax = plt.subplots(1, N, figsize = [N*10, 10])
plt.subplots_adjust(wspace = 0.001)
bins = np.arange(200, 295, 1)

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
       
    h_prior = np.histogram(y_prior[:, i183], bins, density = True)                        
    h0 = np.histogram(y0, bins, density = True)
    h_p = np.histogram(y_pre[:, iq], bins, density = True)
    center = (bins[:-1] + bins[1:])/2

    ax[i].plot(center, h_prior[0], linewidth = 2.5, color = 'k', label = "All-sky")
    ax[i].plot(center, h0[0], linewidth = 2.5, color = 'r', label = "Clear-sky")
    ax[i].plot(center, h_p[0], linewidth = 2.5, color =  'b', label = "Predicted")
    ax[i].set_yscale('log')

#    ax[i].set_xlabel('TB [K]')
    ax[i].xaxis.set_minor_locator(MultipleLocator(5))
    ax[i].grid(which = 'both', alpha = 0.2)
    ax[i].set_title('Channel:%s'%target, fontsize = 32)
#    ax[i].set(ylim = [0, 1])
    
  
ax[0].set_ylabel('Occurence frequency [#/K]')
ax[1].set_xlabel('TB[K]')
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
(ax[2].legend(prop={'size': 32}, frameon = False, bbox_to_anchor=(0.1, -0.12),ncol=3))     
                           
                                  
                                
fig.savefig('Figures/PDF_predictions_ICI.pdf', bbox_inches = 'tight')                                              
fig.savefig('Figures/PDF_predictions_ICI.png', bbox_inches = 'tight')                                 
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
    
qrnn = QRNN.load(file)  