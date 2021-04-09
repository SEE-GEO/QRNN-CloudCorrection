#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:35:37 2020

@author: inderpreet

this code plots the PDF of the predicted error and observed error (error  of best estimate (median))

This script is used to plot Figure 8 of the article.
"""
import os
from mwhs import mwhsData
import matplotlib.pyplot as plt
import numpy as np
from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")
import ICI.stats as S
plt.rcParams.update({'font.size': 26})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")

#%%
def sample_posterior(x, nn, y0, y_pre):
    """
    estimates histogram of predicted error (random samples drawn from posterior)

    Parameters
    ----------
    x : input data for QRNN predictions
    nn : number of samples required for each case
    y0 : NFCS simulations
    y_pre : predicted median

    Returns
    -------
    hist_sample : frequency and bins of predicted errors
    """
    n = x.shape[0]
    y_pos = []
    for i in range(n):
        y_pos.append(qrnn.sample_posterior(x[i, :], nn))
    y_pos = np.array(y_pos)
    
    d = []
    for j in range(nn):
        
        d.append( y_pre - y_pos[:, j] )
    d = np.array(d).ravel()    
    hist_sample = np.histogram(d, bins, density = True)
    return hist_sample


def predict(test_data, qrnn, add_noise = False):
    """
    predict the posterior mean and median
    """
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
    
    return y_pre, y_prior, y0, y, y_pos_mean, x.data



if __name__ == "__main__":
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
        
        d = {"C89+150" : [1, 10],
             "C89+150+118" : [1, 10, 6, 7 ],
             "C150" : [10],
             "C89+150+183" : [1, 10, 11, 12, 13, 14, 15]           
            } 
          
        target = 14
        bins = np.arange(-20, 30 , 0.5)
        channels = d[qrnn_dir]
        
        #%% read input data
                
        fig, ax = plt.subplots(1, 1, figsize = [8, 8])
        print(qrnn_dir, channels)
        
        qrnn_path = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/MWHS/qrnn_output/all_with_flag/%s/"%(qrnn_dir))
        

        inChannels = np.concatenate([[target], channels])
 #       inChannels = np.array(channels)
        
        print(qrnn_dir, channels, inChannels)
            
        qrnn_file = os.path.join(qrnn_path, "qrnn_mwhs_%s.nc"%(target))
        
        print (qrnn_file)
        i183, = np.argwhere(inChannels == target)[0]
        
        data = mwhsData(test_file, 
                       inChannels, target, ocean = False, test_data = True)  
    
        qrnn = QRNN.load(qrnn_file)
        y_pre, y_prior, y0, y, y_pos_mean, x = predict(data, qrnn, \
                                                  add_noise = False)
    
        
        im = np.abs(y_pre[:, iq] - y_prior[:, i183]) < 5.0
        hist_noise, hist_pre, hist_prior, hist_pos_mean, hist_pos_mean_5, hist_filter  = \
            S.calculate_all_histogram(y, y0, y_pre, y_prior, iq, bins, im, i183)
                                    
        nn = 1   
        hist_sample = sample_posterior(x, nn, y0, y_pre[:,3])
        center = (bins[:-1] + bins[1:]) / 2

        ax.plot(center, hist_pre[0],'b', linewidth = 2.5)
    
        ax.plot(center, hist_sample[0], 'r--', linewidth = 2.5)

        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.grid(which = 'both', alpha = 0.2)

    
        ax.set_yscale('log')
    
    #    ax[i].set(ylim = [0, 1])
    #%%    
        ax.set_title('MWHS-2 Channel %s'%str(target), fontsize = 24)
    #     ax1.set_title('QRNN-all', fontsize = 28)
        ax.set_ylabel(r'Occurence frequency [K$^{-1}$]')
        ax.set_xlabel('Deviation from NFCS simulations [K]')

        ax.set_ylim(0.0001, 1)                            
    #     ax.set_ylim(0.0001, 1)
        ax.legend([ "Observed", "Predicted", ],
                     prop={'size': 22}, frameon = False, )                                
                                    
                                        
        fig.savefig('Figures/deviation_posterior_mwhs_samples_%s.pdf'%(target),\
                 bbox_inches = 'tight')                               
#%%
