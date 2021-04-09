#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:57:41 2020

@author: inderpreet

calculate statistics for ICI point estimates, the results are given in latex format

This script is used for Table 6 of the article
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import stats as S
from ici import iciData
plt.rcParams.update({'font.size': 26})
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

def correction(y_pre, y_prior, i183):    
    bins = np.arange(-100, 30, 2)
    im = np.abs(y_pre[:, 3] - y_prior[:, i183]) >= 0.0005
    
    cloud = y_prior[im, i183] - y0[im]
    corr = y_prior[im, i183] - y_pre[im, 3]   
    
    hist= np.histogram(cloud, bins, density = True)
    hist1 = np.histogram(corr, bins, density = True)
    
    return hist[0], hist1[0], bins

#%% input parameters
depth     = 4
width     = 128
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
batchSize = 128

targets = ['I1V', 'I2V', 'I3V']
inpath  = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/ICI/")
test_file = os.path.join(inpath, "data/TB_ICI_test.nc")

iq = np.argwhere(quantiles == 0.5)[0,0]

color = ['k', 'r', 'b']

Y_pre     = []
Y_pre_all = []

#%%
fig, ax = plt.subplots(1,1)
for i,target in enumerate(targets):
    inChannels_single = np.array([target, 'I5V' , 'I6V', 'I7V','I8V', 'I9V', 'I10V', 'I11V'])
    file_single = os.path.join(inpath, 'qrnn_output/qrnn_ici_%s_%s_%s_single.nc'%(depth, width, target))
    print (file_single)
    i183, = np.argwhere(inChannels_single == target)[0]
    
    y_pre, y_prior, y0, y, y_pos_mean = read_qrnn(file_single, inChannels_single, target )
    im = np.abs(y_pre[:, 3] - y_prior[:, i183]) <= 5
    print ((1 - np.sum(im)/im.size)* 100)
    
    
    bia      = stats.calculate_bias(y_prior, y0, y, y_pre[:, 3], im, i183)
    std      = stats.calculate_std(y_prior, y0, y, y_pre[:, 3], im, i183)
    ske      = stats.calculate_skew(y_prior, y0, y, y_pre[:, 3], im, i183)
    mae      = stats.calculate_mae(y_prior, y0, y, y_pre[:, 3], im, i183)
    
    Y_pre.append(y_pre[:, 3])
    
    inChannels_all = np.array(['I1V', 'I2V', 'I3V', 'I5V' , 'I6V', 'I7V', 'I8V', 'I9V', 'I10V', 'I11V'])
    file_all = os.path.join(inpath, 'qrnn_output/qrnn_ici_%s_%s_%s.nc'%(depth, width, target))
    print (file_all)
    i183, = np.argwhere(inChannels_all == target)[0]
    y_pre, y_prior1, y01, y, y_pos_mean1 = read_qrnn(file_all, inChannels_all, target )
    im = np.abs(y_pre[:, 3] - y_prior1[:, i183]) <=5
    print ((1 - np.sum(im)/im.size)* 100)
    bia_A      = stats.calculate_bias(y_prior1, y0, y, y_pre[:, 3], im, i183)
    std_A      = stats.calculate_std(y_prior1, y0, y, y_pre[:, 3], im, i183)
    ske_A      = stats.calculate_skew(y_prior1, y0, y, y_pre[:, 3], im, i183)
    mae_A      = stats.calculate_mae(y_prior1, y0, y, y_pre[:, 3], im, i183)

    Y_pre_all.append(y_pre[:, 3])
#%%
    bia = list(bia + bia_A)
    mae = list(mae + mae_A)
    ske = list(ske + ske_A)
    std = list(std + std_A)
#%%    
    sets = []
#    for j in [0, 1, 4, 2, 3, 9, 7, 8]:
    for j in [0, 1, 2, 3, 7, 8]:
        
        l = [bia[j], mae[j], std[j], ske[j]]  
        sets.append(l)
    sets_names = ['bias', 'mae', 'std', "skewness"]#, 'corrected(1sigma)', 'sreerekha et al', 'filtered(1sigma)']

#%%

    table  = [[sets_names[ii], sets[0][ii], \
                               sets[1][ii],
                               sets[2][ii],
                               sets[3][ii],
                               sets[4][ii],
                               sets[5][ii],
#                               sets[6][ii],
#                               sets[7][ii],
           ] for ii in range(4)]

    print(tabulate(table
             ,  tablefmt="latex", floatfmt=".2f"))
#%%

    hist, hist1, bins = correction(y_pre, y_prior, i183)
    ax.plot(bins[:-1],hist, '--', color = color[i], )
    ax.plot(bins[:-1],hist1, color = color[i])
    ax.legend(["cloud", "corr"])  
    ax.set_yscale('log')  
    
#%%

fig, ax = plt.subplots(1, 1)
ax.scatter( y_prior[:, i183]-y_pre[:, iq], y_prior[:, i183] - y0)
x = np.arange(-150, 0, 1)
y = x
ax.plot(x, y)











 