#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:57:41 2020

@author: inderpreet
Compute and plot the correlations between different channels 

This script is used to plot Figure 5 of the article.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from read_qrnn import read_qrnn
plt.rcParams.update({'font.size': 26})
import seaborn as sns
#%%
def get_corr(im, Y_pre, Y):       

    A = (Y_pre[im, :] - Y[im]).T    
    cov = np.round(np.corrcoef(A), 2)
    
    return cov

if __name__ == "__main__":
    #%% input parameters
    depth     = 3
    width     = 128
    quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
    batchSize = 128
    
    targets = [11, 12, 13, 14, 15]
    
    test_file = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/MWHS/data/TB_MWHS_test.nc")
    
    iq = np.argwhere(quantiles == 0.5)[0,0]
    
    qrnn_dir = "C89+150"
    #qrnn_dir = "C150"
    #qrnn_dir = "C150+118"
         
            
    d = {"C89+150" : [1, 10],
         "C89+150+118" : [1, 10, 6, 7 ],
         "C150" : [10],
         "C89+150+183" : [1, 10, 11, 12, 13, 14, 15]           
        } 
    
    
    qrnn_dirs = ["C89+150", "C89+150+183"]
    #qrnn_dirs = ["C89+150", "C89+150+118", "C150" ]
    

    #%%
    # Set up the matplotlib figure
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
 #   plt.subplots_adjust(wspace = 0.02)
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap = "YlGnBu"
    #%%
   
    for ii, qrnn_dir in enumerate(qrnn_dirs): 
        Y_pre = []
        Y = []
        Y_prior = []
        
        for target in targets:
                   
            qrnn_path = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/MWHS/qrnn_output/all_with_flag/%s/"%(qrnn_dir))
            
            
            channels = np.array(d[qrnn_dir])
            
            if target not in channels:
                inChannels = np.concatenate([[target], channels])
            else:
                inChannels = channels
            
            print(qrnn_dir, channels, inChannels)
                
            qrnn_file = os.path.join(qrnn_path, "qrnn_mwhs_%s.nc"%(target))
            
            print (qrnn_file)
            i183, = np.argwhere(inChannels == target)[0]
            
            y_pre, y_prior, y0, y, y_pos_mean = read_qrnn(qrnn_file, test_file, inChannels, target)
            im = (np.abs(y_pre[:, 3] - y_prior[:, i183] )< 5) 
            
            Y_pre.append(y_pre[:, 3])
            Y.append(y0)
            Y_prior.append(y_prior[:, i183])
        Y         = np.stack(Y, axis = 1)
        Y_pre     = np.stack(Y_pre, axis = 1)
        Y_prior   = np.stack(Y_prior, axis = 1)
     
    
    #%%
        iclear = []
        for i in range(5):
            iclear.append(np.abs(Y_prior[:, i] - Y[:, i]) < 2.0)
        im = np.stack(iclear, axis = 1)
        im = np.logical_and.reduce(im, axis = 1)	    
        
        cov_noise = get_corr(im, Y_pre, Y)
        cov_cloud = get_corr(~im, Y_pre, Y)
        
        mask = np.triu(np.ones_like(cov_noise, dtype=bool))
    
        if ii == 0:
            IM = sns.heatmap(cov_noise, xticklabels = [11, 12, 13, 14, 15], yticklabels =[11, 12, 13, 14, 15], \
                        mask = mask , annot=True, fmt='g', cmap = cmap, vmin = 0, vmax = 1, 
                        ax = ax[0], cbar = False)  
                
            ax[0].set_title('QRNN-single (Clear)')    
      
            # sns.heatmap(cov_cloud, xticklabels = [11, 12, 13, 14, 15], yticklabels =[11, 12, 13, 14, 15], \
            #         mask = mask , annot=True, fmt='g', cmap = cmap, vmin = 0, vmax = 1, 
            #         ax = ax[1], cbar = False)      
            # ax[1].set_title(' QRNN-single (Cloudy)')     
        else:
    
            sns.heatmap(cov_noise, xticklabels = [11, 12, 13, 14, 15], yticklabels =[11, 12, 13, 14, 15], \
                        mask = mask , annot=True, fmt='g', cmap = cmap, vmin = 0, vmax = 1, 
                        ax = ax[1], cbar = False)  
            ax[1].set_title('QRNN-all (Clear) ')  
            ax[0].set_ylabel('MWHS-2 Channels', fontsize = 26)
            ax[0].set_xlabel('MWHS-2 Channels', fontsize = 26)
            
            ax[1].set_ylabel('MWHS-2 Channels', fontsize = 26)
            ax[1].set_xlabel('MWHS-2 Channels', fontsize = 26)
            # sns.heatmap(cov_cloud, xticklabels = [11, 12, 13, 14, 15], yticklabels =[11, 12, 13, 14, 15], \
            #         mask = mask , annot=True, fmt='g', cmap = cmap, vmin = 0, vmax = 1, 
            #         ax = ax[3], cbar = False)      
            # ax[3].set_title('QRNN-all (Cloudy)')
            mappable = IM.get_children()[0]
            cb = plt.colorbar(mappable, ax = [ax[0],ax[1]],\
                         orientation = 'horizontal', 
                         fraction=0.05)
            cb.ax.set_xlabel('correlation coefficient')    
                
        ax[0].annotate('(a)',
            xy=(0.055, .97), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=26)      
        ax[1].annotate('(b)',
            xy=(.545, .97), xycoords='figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=26)           
     
    #%%
    fig.savefig('Figures/correlation.pdf', bbox_inches = 'tight')    
        