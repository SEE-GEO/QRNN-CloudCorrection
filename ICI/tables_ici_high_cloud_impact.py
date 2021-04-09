#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:57:41 2020

@author: inderpreet

statistics for all cases with cloud correction greater than 10K
gives the results in latex table format
"""


import matplotlib.pyplot as plt
import numpy as np
import stats as S
from ici_mwi import iciData
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


#%%

if __name__ == "__main__":
    # INPUT parameters
    depth     = 4
    width     = 128
    quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
    batchSize = 128
    
    targets = ['I1V', 'I2V', 'I3V']
    test_file = "TB_ICI_test.nc"
    
    iq = np.argwhere(quantiles == 0.5)[0,0]
    
    
    
    
    #%%
    
    for i,target in enumerate(targets):
        inChannels_single = np.array([target, 'I5V' , 'I6V', 'I7V', 'I8V', \
                                      'I9V', 'I10V', 'I11V'])
        file_single = 'qrnn_ici_%s_%s_%s_single.nc'%(depth, width, target)
        print (file_single)
        i183, = np.argwhere(inChannels_single == target)[0]
        
        y_pre, y_prior, y0, y, y_pos_mean = read_qrnn(file_single,\
                                                      inChannels_single, target )
        im = np.abs(y_pre[:, 3] - y_prior[:, i183]) >= 10
        print (( np.sum(im)/im.size)* 100)
        
        
        bia      = stats.calculate_bias(y_prior, y0, y, y_pre[:, 3], im, i183)
        std      = stats.calculate_std(y_prior, y0, y, y_pre[:, 3], im, i183)
        ske      = stats.calculate_skew(y_prior, y0, y, y_pre[:, 3], im, i183)
        mae      = stats.calculate_mae(y_prior, y0, y, y_pre[:, 3], im, i183)
        
        
        
        inChannels_all = np.array(['I1V', 'I2V', 'I3V', 'I5V' , 'I6V', 'I7V', \
                                   'I8V', 'I9V', 'I10V', 'I11V'])
        file_all = 'qrnn_ici_%s_%s_%s.nc'%(depth, width, target)
        print (file_all)
        i183, = np.argwhere(inChannels_all == target)[0]
        y_pre, y_prior1, y01, y, y_pos_mean1 = read_qrnn(file_all,\
                                                         inChannels_all, target )
        im = np.abs(y_pre[:, 3] - y_prior[:, i183]) >= 10
        print (( np.sum(im)/im.size)* 100)
        bia_A      = stats.calculate_bias(y_prior1, y0, y, y_pre[:, 3], im, i183)
        std_A      = stats.calculate_std(y_prior1, y0, y, y_pre[:, 3], im, i183)
        ske_A      = stats.calculate_skew(y_prior1, y0, y, y_pre[:, 3], im, i183)
        mae_A      = stats.calculate_mae(y_prior1, y0, y, y_pre[:, 3], im, i183)
        
    #%%
        bia = list(bia + bia_A)
        mae = list(mae + mae_A)
        ske = list(ske + ske_A)
        std = list(std + std_A)
    #%%    
        sets = []
        for i in [0, 1, 4, 2, 3, 9, 7, 8]:
            
            l = [bia[i], mae[i], std[i], ske[i]]  
            sets.append(l)
        sets_names = ['bias', 'mae', 'std', "skewness"]
    
        table  = [[sets_names[i],  sets[4][i],
                                   sets[7][i],
               ] for i in range(4)]
    
        print(tabulate(table
                 ,  tablefmt="latex", floatfmt=".2f"))
