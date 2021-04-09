#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 22:09:27 2020

@author: inderpreet

PLot the uncertainties in randomly chosen cases
This script is used to plot Figure 6 of the article.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import netCDF4
from read_qrnn import read_qrnn
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")
import random
plt.rcParams.update({'font.size': 26})
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


#%% input parameters
depth     = 3
width     = 128
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
batchSize = 128

targets = [11, 12, 13, 14, 15]
test_file = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/MWHS/data/TB_MWHS_test.nc")

iq = np.argwhere(quantiles == 0.5)[0,0]

#qrnn_dir = "C89+150"
#qrnn_dir = "C150"
qrnn_dir = "C89+150"

if qrnn_dir == "C89+150":
    channels = [ 1, 10]
  
target = 14 

#%% read input data
        
print(qrnn_dir, channels)

qrnn_path = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/MWHS/qrnn_output/all_with_flag/%s/"%(qrnn_dir))

inChannels = np.concatenate([[target], channels])

print(qrnn_dir, channels, inChannels)
    
qrnn_file = os.path.join(qrnn_path, "qrnn_mwhs_%s.nc"%(target))

print (qrnn_file)
i183, = np.argwhere(inChannels == target)[0]

y_pre, y_prior, y0, y, y_pos_mean = read_qrnn(qrnn_file, test_file, inChannels, target)

fig, ax = plt.subplots(1, 1, figsize = [8, 8])
x = np.arange(-3, 4, 1)
ii = 0
y_all = []
randomList = random.sample(range(0, 65000), 1500)
for i in randomList:
    ii +=1
#for i in ind:
    y1 = y_pre[i,  :] - y_pre[i, 3]
    y_all.append(y1)    
    ax.plot(x, y1, color = colors["grey"], alpha = 0.4)


#%% add box

y_all = np.stack(y_all)
box1 = ax.boxplot(y_all, positions = x, showfliers=False,  widths = 0.9)
for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box1[item], color="darkred")

#%%
y_normal = np.random.normal(270, 1.0, 24000)
q_normal = np.quantile(y_normal, quantiles , axis = 0)
ax.plot( x, q_normal - q_normal[3], 'b', linewidth =2)
ax.tick_params(axis='x', which='major', pad=10)

ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.grid(which = 'both', alpha = 0.2)

ax.set_xlabel("Quantiles")
ax.set_ylabel("Prediction uncertainty [K]")
ax.set_xticks(x)
ax.set_xticklabels((r'$-3\sigma$', r'$-2\sigma$', r'$-1\sigma$', 0,r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'), va='center')


q_prior = np.quantile(y_prior, quantiles, axis = 0)
#ax.plot(x, q_prior - q_prior[3])

q0 = np.quantile(y0, quantiles, axis = 0)
#ax.plot(x, q0 - q0[3])





ax.tick_params(axis='x', which='major', pad=20)
ax.set_title("MWHS-2 Channel %s"%str(target), fontsize = 24)
fig.savefig('Figures/prediction_uncertainty_MWHS_%s.pdf'%(target), bbox_inches = 'tight')
