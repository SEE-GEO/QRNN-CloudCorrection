#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:32:49 2020

This script is used to plot Figure 13 of the article.

@author: inderpreet
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import netCDF4
plt.rcParams.update({'font.size': 26})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy.stats import skew

from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")
plt.rcParams["text.usetex"]

inChannels = ['C34', 'C41', 'C42', 'C43', 'C44']
dtb_max_meas = 5
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])

from aws_test_data import awsTestData
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

def bin_uncertainty(y_pre, y_prior, y0):

    dtb =  y_prior - y0
#
    print (dtb.min(), dtb.max())
    ci = y_pre[:, 6] - y_pre[:, 0]
    bins = np.arange(-40, 1, 1)
    A = np.digitize(dtb, bins)
    mean_ci = []
    for i in range(bins.shape[0] - 1):
        ix = np.where(A == i)[0]
        if np.sum(ix) >0:
            mean_ci.append(np.mean(ci[ix]))
        else:
            mean_ci.append(np.nan)
#    plt.plot(bins[:-1], mean_ci, '--')
    return mean_ci, bins

def bin_errors(y_pre, y_prior, y0):
    dtb =  y_prior - y0
    er = y_pre[:, 3] - y0
    bins = np.arange(-40, 0, 1)
    A = np.digitize(dtb, bins)
    mean_ci = []
    for i in range(39):
        ix = np.where(A == i)[0]
        if np.sum(ix) >0:
            mean_ci.append(np.mean(er[ix]))
        else:
            mean_ci.append(np.nan)
#    plt.plot(bins[:-1], mean_ci, '--')
    return mean_ci, bins


def histogram(ci, bins):
    hist = np.histogram(ci, bins, density = True)
    return hist[0]

#%%
colors = ['r', 'b', 'g', 'k', 'm']   
#colors = ['#d7191c','#fdae61','y','#abd9e9','#2c7bb6'] 
channels = ['C32', 'C33', 'C34', 'C35', 'C36']
#aws_channels = ['AWS-32', 'AWS-33', 'AWS-34', 'AWS-35', 'AWS-36']
aws_channels = ['SMS-1', 'SMS-2', 'SMS-3', 'SMS-4', 'SMS-5']
fig, ax = plt.subplots(1, 1, figsize = [8,8])
fig1, ax1 = plt.subplots(1, 1, figsize = [8,8])
ci_bins = np.arange(0, 50, 1)
#ci_bins = np.arange(-30, 30, 1)
center_ci_bins = (ci_bins[:-1] + ci_bins[1:])/2
bins = np.arange(-40, 1, 1)
center = (bins[:-1] + bins[1:])/2
inpath = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/AWS/")

for i, c183 in enumerate(channels):

    inChannels = np.array([c183, 'C41', 'C42', 'C43', 'C44'])

    test_data = awsTestData(os.path.join(inpath, "data/TB_AWS_m60_p60_noise_four_test.nc"), 
                   inChannels, option = 4)     
    qrnn = QRNN.load(os.path.join(inpath, 'qrnn_data/qrnn_3_256_%s.nc'%c183))
    y_pre, y_prior, y0, y, y_pos_mean = predict(test_data, qrnn)
    
    mean_ci, bins = bin_uncertainty(y_pre, y_prior[:, 0], y0)
    #mean_ci, bins = bin_errors(y_pre, y_prior[:, 0], y0)
    ax.plot(center, mean_ci, linewidth = '3' , color = colors[i])
    
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(which = 'both', alpha = 0.2)
    ci = y_pre[:, 5] - y_pre[:, 1]
    hist = histogram(ci, ci_bins)
    
    
    ax1.plot(center_ci_bins, hist, color = colors[i], linewidth = 2.5)
#    ax[i].set(ylim = [0, 1])
    
ax.set_ylabel(r'Mean confidence interval ($\pm3\sigma$) [K]')
ax.set_xlabel('Cloud impact [K]')
ax1.xaxis.set_minor_locator(MultipleLocator(5))
ax1.yaxis.set_minor_locator(MultipleLocator(5))
ax1.grid(which = 'both', alpha = 0.2)
ax1.set_yscale('log')                           
(ax.legend(aws_channels,
            prop={'size': 22}, frameon = False))      
(ax1.legend(aws_channels,
            prop={'size': 22}, frameon = False))                             
ax1.set_ylabel(r'Occurence frequency [K$^{-1}$]')
ax1.set_xlabel('Uncertainty [K]')                                
                                
fig.savefig('Figures/cloud_impact_uncertainty_AWS.pdf', bbox_inches = 'tight')         
fig.savefig('Figures/histogram_uncertainty_AWS.pdf', bbox_inches = 'tight')     