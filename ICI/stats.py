#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:42:56 2020

@author: inderpreet
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
plt.rcParams.update({'font.size': 16})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy.stats import gaussian_kde

quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])

def predict(test_data, qrnn, add_noise = False, aws = False):
    """
    predict the posterior mean and median
    """
    
    if aws:
        x = (test_data.x - test_data.mean)/test_data.std
        y_prior = test_data.x
        y = test_data.y
        y0 = test_data.y0
                 
    else:
        
        if add_noise:
            x_noise = test_data.add_noise(test_data.x, test_data.index)
            x = (x_noise - test_data.mean)/test_data.std
            y_prior = x_noise
            y = test_data.y_noise
            
            y0 = test_data.y
        else:
            x = (test_data.x - test_data.mean)/test_data.std
            y_prior = test_data.x
            y = test_data.y_noise
            y0 = test_data.y
            
            if not test_data.ocean :
        
                x = np.concatenate((x, test_data.lsm ), axis = 1)

    y_pre = qrnn.predict(x.data)
    y_pos_mean = qrnn.posterior_mean(x.data)
    
    return y_pre, y_prior, y0, y, y_pos_mean

def calculate_all_histogram(y, y0, y_pre, y_prior, iq, bins, im, i183):
    hist_noise  = calculate_pdf(y, y0, bins)
    
    hist_pre    = calculate_pdf(y_pre[:, iq], y0, bins)
    
    hist_prior  = calculate_pdf(y_prior[:, i183], y0, bins)
    
    hist_pos_mean    = calculate_pdf(y_pre[:, iq], y0, bins)
    

    hist_pos_mean_5  = calculate_pdf(y_pre[:, iq], y0, bins, im)
    hist_filter  = calculate_pdf(y_prior[:, i183], y0, bins, im)
    
    return hist_noise, hist_pre, hist_prior, hist_pos_mean, hist_pos_mean_5, hist_filter    

def calculate_pdf(y, y0, bins, im = None):

    error = y - y0
    if im is not None:
        hist    = np.histogram(error[im], bins, density = True)
    else:
        hist    = np.histogram(error, bins, density = True)       
    
    return hist

def calculate_bias(y_prior, y0, y, y_pos_mean, im, itarget): 
    b0  = np.mean(y - y0)
    b1  = np.mean(y_prior[:, itarget] - y0)
    b2a = np.mean(y_pos_mean - y0) 
    b2b = np.mean(y_pos_mean[im] - y0[im])
    b1_filter = np.mean(y_prior[im, itarget] - y0[im])
    return b0, b1, b2a, b2b, b1_filter

def calculate_std(y_prior, y0, y, y_pos_mean, im, itarget):  
    std0  = np.std(y - y0)
    std1  = np.std(y_prior[:, itarget] - y0)
    std2a = np.std(y_pos_mean - y0)
    std2b = np.std(y_pos_mean[im] - y0[im])
    std1_filter  = np.std(y_prior[im, itarget] - y0[im])
    return std0, std1, std2a, std2b, std1_filter

def calculate_mae(y_prior, y0, y, y_pos_mean, im, itarget): 
    mae0  = np.mean(np.abs(y - y0))
    mae1  = np.mean(np.abs(y_prior[:, itarget] - y0))
    mae2a = np.mean(np.abs(y_pos_mean - y0)) 
    mae2b = np.mean(np.abs(y_pos_mean[im] - y0[im]))
    mae1_filter  = np.mean(np.abs(y_prior[im, itarget] - y0[im]))
    return mae0, mae1, mae2a, mae2b, mae1_filter

def calculate_skew(y_prior, y0, y, y_pos_mean, im, itarget):
    skew0  = skew(y - y0)
    skew1  = skew(y_prior[:, itarget] - y0)
    skew2a = skew(y_pos_mean - y0)
    skew2b = skew(y_pos_mean[im] - y0[im])
    skew1_filter  = skew(y_prior[im, itarget] - y0[im])
    return skew0, skew1, skew2a, skew2b, skew1_filter

# Calculate the point density
def density (x, y):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    return x, y, z

def calculate_statistics(y_prior, y0, y, y_pos_mean, im, itarget):
    b0, b1, b2a, b2b              = calculate_bias(y_prior, y0, y, y_pos_mean, im, itarget)
    std0, std1, std2a, std2b      = calculate_std(y_prior, y0, y, y_pos_mean, im, itarget)
    skew0 , skew1, skew2a, skew2b = calculate_skew(y_prior, y0, y, y_pos_mean, im, itarget)
    mae0, mae1, mae2a, mae2b      = calculate_mae(y_prior, y0, y, y_pos_mean, im, itarget)
    from tabulate import tabulate
    b        = [b0, b1,  b2a, b2b]
    std      = [std0, std1,  std2a, std2b]
    skewness = [skew0, skew1,  skew2a, skew2b]
    mae      = [mae0, mae1, mae2a, mae2b]
    rejected = 1 - np.sum(im)/im.size
    rejected = [0, 0, 0, rejected * 100]


    sets = ['Noise', 'uncorrected', 'corrected (all)', "corrected (filtered)"]#, 'corrected(1sigma)', 'sreerekha et al', 'filtered(1sigma)']



    table  = [[sets[i], b[i], mae[i], std[i], skewness[i], rejected[i]] for i in range(4)]

    print(tabulate(table
             , ["Dataset","bias", "mae", "std", "measure skewness", 'rejected'],  tablefmt="latex", floatfmt=".2f"))

def calculate_statistics_T(y_prior, y0, y, y_pos_mean, im, itarget):
    b0, b1, b2a, b2b              = calculate_bias(y_prior, y0, y, y_pos_mean, im, itarget)
    std0, std1, std2a, std2b      = calculate_std(y_prior, y0, y, y_pos_mean, im, itarget)
    skew0 , skew1, skew2a, skew2b = calculate_skew(y_prior, y0, y, y_pos_mean, im, itarget)
    mae0, mae1, mae2a, mae2b      = calculate_mae(y_prior, y0, y, y_pos_mean, im, itarget)
    from tabulate import tabulate
    rejected = 1 - np.sum(im)/im.size
    noise        = [b0, mae0, std0,  skew0, 0 ]
    uncorrected      = [b1, mae1, std1,  skew1, 0 ]
    corrected_all = [b2a, mae2a, std2a,  skew2a, 0 ]
    corrected_filtered      = [b2b, mae2b, std2b,  skew2b, rejected*100 ]




    sets = ['bias', 'mae', 'std', "skewness", "rejected"]#, 'corrected(1sigma)', 'sreerekha et al', 'filtered(1sigma)']



    table  = [[sets[i], noise[i], uncorrected[i], corrected_all[i], corrected_filtered[i]] for i in range(5)]

    print(tabulate(table
             , ["noise", "uncorrected", "corrected_all", "corrected_filtered"],  tablefmt="latex", floatfmt=".2f"))

def uncertainty(y_pre, target, filename):
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    x = np.arange(-3, 4, 1)
    ii = 0
    for i in range(0, 20000, 10):
        ii +=1
    #for i in ind:
        y1 = y_pre[i,  :] - y_pre[i, 3]

        ax.plot(x, y1)

    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which = 'both', alpha = 0.2)

    ax.set_xlabel("Quantiles")
    ax.set_ylabel("Prediction uncertainty [K]")
    ax.set_xticks(x)
    ax.set_xticklabels(quantiles)
    ax.set_title("Channel :%s"%target)
    y_pre[100,  :] - y_pre[100, 3]

    fig.savefig('Figures/%s_%s'%(filename, str(target)))

def scatter_error_uncertainty( y_pre, y0, target, filename):
    dtb =(y_pre[:, 3] - y0)
    x, y, z = density(y_pre[:, 5] - y_pre[:, 1], dtb[:] )
    fig, ax = plt.subplots(1,1, figsize = [8,8])
    ax.scatter(x, y, c=z, s=50, edgecolor='', alpha = 0.4)
    ax.set_ylabel('Error [K]')
    ax.set_xlabel('2-sigma uncertainty [K]')
    ax.set_title("Channel :%s"%target)
    ax.set_ylim(-20, 25)
    ax.set_xlim(0, 25)
    fig.savefig('Figures/%s_%s'%(filename, str(target)))