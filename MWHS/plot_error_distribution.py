#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:57:41 2020

@author: inderpreet

plot the error distributions for MWHS-2

This script is used to plot Figure 4 of the article.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import ICI.stats as S
from read_qrnn import read_qrnn
plt.rcParams.update({'font.size': 26})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from tables_SI import get_SI_land, get_SI_ocean

from mwhs import mwhsData
#%%

#%% input parameters
depth     = 3
width     = 128
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
batchSize = 128

targets = [11, 12, 13, 14, 15]
targets = [14]
test_file = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/MWHS/data/TB_MWHS_test.nc")

iq = np.argwhere(quantiles == 0.5)[0,0]

qrnn_dir = "C89+150"
#qrnn_dir = "C150"
#qrnn_dir = "C150+118"
     
        
d = {"C89+150" : [1, 10],
     "C89+150+118" : [1, 10, 6, 7 ],
      "C89+150+118_v2" : [1, 10,  7, 8 ],
     "C150" : [10],
     "C89+150+183" : [1, 10, 11, 12, 13, 14, 15]           
    } 

Channels = [[1, 10], [1, 6, 7, 10]]
#qrnn_dirs = ["C89+150", "C89+150+118", "C89+150+118_v2" ]
qrnn_dirs = ["C89+150"]

bins = np.arange(-30, 10, 0.5)
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
iq = np.argwhere(quantiles == 0.5)[0,0]

allChannels = np.arange(1, 16, 1)


path = os.path.expanduser('~/Dendrite/Projects/AWS-325GHz/MWHS/data/')
TB_ob = np.load(os.path.join(path, 'TB_obs.npy'))
TB_fg = np.load(os.path.join(path, 'TB_fg.npy'))
TB_cl = np.load(os.path.join(path, 'TB_cl.npy'))
i89, = np.argwhere(allChannels == 1)[0]
i150, = np.argwhere(allChannels == 10)[0]

#%%

if __name__ == "__main__":
#%%
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    for target in targets:
    
        for qrnn_dir in qrnn_dirs:
                    
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
            im = (np.abs(y_pre[:, 3] - y_prior[:, i183] )<= 5.0) 
 

####-------------------------------------------------------------------
 
            print ('rejected obs', (1 - np.sum(im)/im.size)* 100)
    
            
            hist_noise, hist_pre, hist_prior, hist_pos_mean, hist_pos_mean_5, hist_filter  = \
                S.calculate_all_histogram(y, y0, y_pre, y_prior, iq, bins, im, i183)
        
             
            center = (bins[:-1] + bins[1:]) / 2
        
            ax.plot(center, hist_noise[0], 'k', linewidth = 2.5, label = "Noise")
            ax.plot(center, hist_prior[0], 'g', linewidth = 2.5, label = "All-sky")
            ax.plot(center, hist_pre[0],'b', linewidth = 2.5, label = "Predicted (All)")
        
#            ax.plot(center, hist_pos_mean_5[0], 'c', linewidth = 2.5, label = "Predicted (5K)")
#            ax.plot(center, hist_filter[0], 'y', linewidth = 2.5, label = "Filtered(5K)")
            ax.set_yscale('log')
    
            data = mwhsData(test_file, 
                   inChannels, target, ocean = False, test_data = True) 
    ###
    ###    SI approach
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
            print ("........")
    
            hist_SI = np.histogram(y_fil - y0_fil, bins, density = True)
            ax.plot(center, hist_SI[0], 'y', linewidth = 2.5, label = "Filtered (SI)")
            
            ax.xaxis.set_minor_locator(MultipleLocator(1))
    
            ax.grid(which = 'both', alpha = 0.2)
            ax.set_title('MWHS-2 Channel:%s'%target, fontsize = 24)
        
        #    ax.set(ylim = [0, 1])
            
            ax.set_ylabel(r'Occurence frequency [K$^{-1}$]')
            ax.set_xlabel('Deviation from NFCS simulations [K]')
                                        
            (ax.legend(prop={'size': 24}, frameon = False))                                
                                            
                                            
            fig.savefig("Figures/MWHS_error_dist_%s.pdf"%str(target), \
                        bbox_inches = 'tight')     
        
        
        
        
        
        
    
    
    
     
