#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:57:41 2020

@author: inderpreet

calculate statistics for point estimates from QRNN applied to AWS

This script is used for Table 7  of the article.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import ICI.stats as stats
from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")

from tabulate import tabulate
from aws_test_data import awsTestData

#%% input parameters
depth     = 4
width     = 256
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
batchSize = 128

targets = [ 'C32','C33','C34', 'C35', 'C36']

iq = np.argwhere(quantiles == 0.5)[0,0]
inpath = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/AWS/")
#%%
for i,target in enumerate(targets):
    
    inChannels = np.array([target,'C41', 'C42', 'C43', 'C44'])     
    file = os.path.join(inpath, 'qrnn_data', 'qrnn_%s_%s_%s.nc'%(depth, width, target))

    test_data = awsTestData(os.path.join(inpath, "data", "TB_AWS_m60_p60_noise_four_test.nc"), 
               inChannels, option = 4)

    i183, = np.argwhere(inChannels == target)[0]
        
    qrnn = QRNN.load(file)
    
    y_pre, y_prior, y0, y, y_pos_mean  = stats.predict(test_data, qrnn, \
                                                   add_noise = False, aws = True)
    im = np.abs(y_pre[:, 3] - y_prior[:, i183]) <= 5
    print ((1 - np.sum(im)/im.size)* 100)
    
    
    bia      = stats.calculate_bias(y_prior, y0, y, y_pre[:, 3], im, i183)
    std      = stats.calculate_std(y_prior, y0, y, y_pre[:, 3], im, i183)
    ske      = stats.calculate_skew(y_prior, y0, y, y_pre[:, 3], im, i183)
    mae      = stats.calculate_mae(y_prior, y0, y, y_pre[:, 3], im, i183)
    
    
#%%
    bia = list(bia )
    mae = list(mae )
    ske = list(ske )
    std = list(std )
#%%    
    sets = []
    for i in [0, 1,  2, 3, 4]:
        
        l = [bia[i], mae[i], std[i], ske[i]]  
        sets.append(l)
    sets_names = ['bias', 'mae', 'std', "skewness"]#, 'corrected(1sigma)', 'sreerekha et al', 'filtered(1sigma)']



    table  = [[sets_names[i], sets[0][i], \
                               sets[1][i],
                               sets[2][i],
                               sets[3][i],
                               sets[4][i],

           ] for i in range(4)]

    print(tabulate(table
             ,  tablefmt="latex", floatfmt=".2f"))


#%%
    bins = np.arange(-40, 10, 0.5)
    hist = np.histogram(y_pre[:, 3] - y0, bins, density = True)
    
    fig, ax = plt.subplots(1, 1, figsize= [8,8])
    ax.set_yscale('log')
    ax.plot(bins[:-1], hist[0])