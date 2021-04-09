#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:46:41 2020

@author: inderpreet
This code plots the calibration curves for channel 14 MWHS

This script is used to plot Figure 7 of the article.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from ICI.calibration import calibration
from read_qrnn import read_qrnn
plt.rcParams.update({'font.size': 26})
import os
from tables_SI import get_SI_land, get_SI_ocean
from mwhs import mwhsData
from scipy.stats import norm

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
path = os.path.expanduser('~/Dendrite/Projects/AWS-325GHz/MWHS/data/')
allChannels = np.arange(1, 16, 1)
#%% plot calibration for QRNN output
        
print(qrnn_dir, channels)

qrnn_path = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/MWHS/qrnn_output/all_with_flag/%s/"%(qrnn_dir))

inChannels = np.concatenate([[target], channels])

print(qrnn_dir, channels, inChannels)
    
qrnn_file = os.path.join(qrnn_path, "qrnn_mwhs_%s.nc"%(target))

print (qrnn_file)
i183, = np.argwhere(inChannels == target)[0]

y_pre, y_prior, y0, y, y_pos_mean = read_qrnn(qrnn_file, test_file, inChannels, target)

# calibration plot data with correction greater than 15K

fig, ax = plt.subplots(1, 1, figsize = [8,8])   

im = np.arange(0, y0.size, 1)
a1, a2, a3, a4, a5, a6, intervals  = calibration(y_pre, y0, im, quantiles)
    

(ax.plot(intervals[:], [ a1/len(y0[:]), a2/len(y0[:]), a3/len(y0[:]), 
                           a4/len(y0[:]), a5/len(y0[:]),
                          ], 'r.-', ms = 15, linewidth = 2.5, label="All data"))

im = np.where(np.abs(y_pre[:, iq] - y_prior[:, i183]) > 5)[0]
a1, a2, a3, a4, a5, a6, intervals  = calibration(y_pre, y0, im, quantiles)     

(ax.plot(intervals[:], [ a1/len(y0[im]), a2/len(y0[im]), a3/len(y0[im]), 
                           a4/len(y0[im]), a5/len(y0[im]),
                          ], 'b.-', ms = 15, linewidth = 2.5, label = "correction > 5 K"))


ax.set_title("MWHS-2 Channel %s"%str(target), fontsize = 24)

#%% calculate calibration of MWHS error model ECMWF

TB_ob = np.load(os.path.join(path, 'TB_obs.npy'))
TB_fg = np.load(os.path.join(path, 'TB_fg.npy'))
TB_cl = np.load(os.path.join(path, 'TB_cl.npy'))
err   = np.load(os.path.join(path, 'obs_err.npy'))
i89, = np.argwhere(allChannels == 1)[0]
i150, = np.argwhere(allChannels == 10)[0]
ii183, = np.argwhere(allChannels == target)[0]
data = mwhsData(test_file, 
                   inChannels, target, ocean = False, test_data = True) 
# SI
SI_land = get_SI_land(TB_ob, TB_fg, i89, i150)
SI_ocean = get_SI_ocean(TB_ob, TB_fg, TB_cl, i89, i150)

SI_land = SI_land[data.im]
SI_ocean = SI_ocean[data.im]        
iocean = np.squeeze(data.lsm[:] == 0)
iland = ~iocean

SI_land[iocean] = SI_ocean[iocean]
SI = SI_land.copy()

im = np.abs(SI) <= 5

y_fil = y_prior[im, i183]
y0_fil = y0[im]


TB_ob = TB_ob[:, data.im]
obs_err = np.abs(TB_ob[ii183, im] - y0[im])
err = err[:, data.im]
intervals = norm.interval(quantiles, loc = 0, scale = 1.0)[1]

errors = np.zeros([err[14, im].shape[0], 7])  
for i in range(7):
    errors[:, i] = intervals[i] * err[ii183, im]
im = np.arange(0, y0[im].size, 1)
a1, a2, a3, a4, a5, a6, intervals  = calibration(errors, obs_err, im, quantiles)
    

(ax.plot(intervals[:], [ a1/len(y0[im]), a2/len(y0[im]), a3/len(y0[im]), 
                           a4/len(y0[im]), a5/len(y0[im]),
                          ], 'k.-', ms = 15, linewidth = 2.5, label = "SI"))

#%% set the plot parameters

x = np.arange(0,1.2,0.2)
y = x
ax.plot(x, y, 'k:', linewidth = 1.5)
ax.set(xlim = [0, 1], ylim = [0,1])
ax.set_aspect(1.0)
ax.set_xlabel("Predicted frequency")
ax.set_ylabel("Observed frequency")
ax.xaxis.set_minor_locator(MultipleLocator(0.2))
ax.grid(which = 'both', alpha = 0.2)
fig.savefig('Figures/calibration_plot_%s'%target)

(ax.legend(prop={'size': 22}, frameon = False))  

fig.savefig("Figures/calibration_QRNN_MWHS_%s.pdf"%target, bbox_inches = 'tight')

#%% plot QRNN output
fig, ax = plt.subplots(1, 1, figsize = [8, 8])
bins = np.arange(220, 300, 1)
bin_center = (bins[:-1] + bins[1:]) / 2
hist_prior = np.histogram(y_prior[:, i183], bins, density = True)              
ax.plot(bin_center, hist_prior[0], 'g', linewidth = 2.5)

hist0 = np.histogram(y0, bins, density = True)              
ax.plot(bin_center, hist0[0], 'k' ,linewidth = 2.5)

hist_pre = np.histogram(y_pre[:, 3], bins, density = True)
ax.plot(bin_center, hist_pre[0], 'b', linewidth = 2.5)
                                    

ax.set_yscale('log')
ax.set_xlabel('Brightness temperature [K]')
ax.set_ylabel(r'Occurence frequency [K$^{-1}$]')
ax.legend(["All-sky ",  "Clear-sky ", "Predicted"],\
          prop={'size': 24}, frameon = False, loc = 2) 
ax.set_title("MWHS-2 Channel %s"%str(target), fontsize = 24)
ax.xaxis.set_minor_locator(MultipleLocator(2))

ax.grid(which = 'both', alpha = 0.2)
fig.savefig('Figures/QRNN_output_mwhs.pdf', bbox_inches = 'tight')
