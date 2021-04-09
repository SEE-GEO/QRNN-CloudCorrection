#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:35:37 2020

@author: inderpreet

this code plots the PDF of the predicted error and observed error (error  of best estimate (median))
ICI channels
"""
import os
from mwhs import mwhsData
import matplotlib.pyplot as plt
import numpy as np
from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")
import ICI.stats as S
plt.rcParams.update({'font.size': 32})
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")

#%%
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
        im = (np.abs(y_pre[:, 3] - y_prior[:, i183] )< 5) 
               
#        bia, std, ske, mae = S.calculate_statistics(y_prior, y0, y, y_pre[:, 3], im, i183)
        
        SI = np.abs(y_prior[:, 0] - y_prior[:, 1])
        
        im = SI < 5.0

        y_pre = y_prior[im, i183]
        y0    = y0[im]
        y     = y[im]
        y_prior = y_prior[im , :] 
        
        bia = S.calculate_bias(y_prior, y0, y, y_pre[:, 3], im, i183)