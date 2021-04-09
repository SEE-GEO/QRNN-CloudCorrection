#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:57:41 2020

@author: inderpreet

calculate statistics for MWHS point estimates, the results are given in latex format

This script is used for table 5 of the article.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import ICI.stats as S
from read_qrnn import read_qrnn
plt.rcParams.update({'font.size': 26})
from tabulate import tabulate
#%%

def calculate_statistics(y_prior, y0, y, y_pre, im, i183):

    bia      = S.calculate_bias(y_prior, y0, y, y_pre[:, 3], im, i183)
    std      = S.calculate_std(y_prior, y0, y, y_pre[:, 3], im, i183)
    ske      = S.calculate_skew(y_prior, y0, y, y_pre[:, 3], im, i183)
    mae      = S.calculate_mae(y_prior, y0, y, y_pre[:, 3], im, i183)
    
    return bia, std, ske, mae

if __name__ == "__main__":
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
    qrnn_dirs = ["C89+150", "C89+150+118", "C150", "C89+150+183" ]
 #   qrnn_dirs = ["C89+150"]
    
    
    
    #%%
    for target in targets:
        BIA = []
        MAE = []
        SKE = []
        STD = []
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
            im = (np.abs(y_pre[:, 3] - y_prior[:, i183] )>= 5) 
            
            print ('rejected obs', (1 - np.sum(im)/im.size)* 100)
            
            bia, std, ske, mae = calculate_statistics(y_prior, y0, y, y_pre, im, i183)
            
        # #%%
            BIA += bia
            MAE += mae
            STD += std
            SKE += ske
        
        # #%%    
            sets = []
    #%% horizontal table    
        jlist = [0, 1, 2, 3]
        
        for i in range(1, len(qrnn_dirs)):
            jlist += [5*i -1 + 3, 5*i -1 + 4 ]
        
        for j in jlist:
            
             l = [BIA[j], MAE[j], STD[j], SKE[j]]  
             sets.append(l)
        sets_names = ['bias', 'mae', 'std', "skewness"]#, 'corrected(1sigma)', 'sreerekha et al', 'filtered(1sigma)']
        
        # #%%
        
        if len(qrnn_dirs) >2:
             table  = [[sets_names[ii], sets[0][ii], \
                                        sets[1][ii],
                                        sets[2][ii],
                                        sets[3][ii],
                                        sets[4][ii],
                                        sets[5][ii],
                                        sets[6][ii],
                                        sets[7][ii],
                                        sets[8][ii],
                                        sets[9][ii],
                    ] for ii in range(4)]
                 
        else:
             table  = [[sets_names[ii], sets[0][ii], \
                                        sets[1][ii],
                                        sets[2][ii],
                                        sets[3][ii],
#                                        sets[4][ii],
#                                        sets[5][ii],
                    ] for ii in range(4)]
             
            
        
        print(tabulate(table
                  ,  tablefmt="latex", floatfmt=".2f"))
 
        
        
        
        
        
        
        
    
    
    
     
