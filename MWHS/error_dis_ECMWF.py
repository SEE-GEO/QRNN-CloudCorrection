#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:51:33 2020

@author: inderpreet
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from read_qrnn import read_qrnn
plt.rcParams.update({'font.size': 26})
from mwhs import mwhsData
from tables_SI import get_SI_land, get_SI_ocean
from scipy.stats import norm

#%%
depth     = 3
width     = 128
quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
batchSize = 128

targets = [11, 12, 13, 14, 15]
targets = [15]
test_file = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/MWHS/data/TB_MWHS_test.nc")
path = os.path.expanduser('~/Dendrite/Projects/AWS-325GHz/MWHS/data/')

iq = np.argwhere(quantiles == 0.5)[0,0]

qrnn_dir = "C89+150"
#qrnn_dir = "C150"
#qrnn_dir = "C150+118"
     
        
d = {"C89+150" : [1, 10],
     "C89+150+118" : [1, 10, 6, 7 ],
     "C150" : [10],
     "C89+150+183" : [1, 10, 11, 12, 13, 14, 15]           
    } 

Channels = [[1, 10], [1, 6, 7, 10]]
qrnn_dirs = ["C89+150"]
#qrnn_dirs = ["C89+150", "C89+150+118", "C150" ]
allChannels = np.arange(1, 16, 1)

if __name__ == "__main__":
#%%
    TB_ob = np.load(os.path.join(path, 'TB_obs.npy'))
    TB_fg = np.load(os.path.join(path, 'TB_fg.npy'))
    TB_cl = np.load(os.path.join(path, 'TB_cl.npy'))
    err   = np.load(os.path.join(path, 'obs_err.npy'))
    i89, = np.argwhere(allChannels == 1)[0]
    i150, = np.argwhere(allChannels == 10)[0]
            
           
    for qrnn_dir, target in zip(qrnn_dirs, targets)   :             
        qrnn_path = os.path.expanduser("~/Dendrite/Projects/AWS-325GHz/MWHS/qrnn_output/all_with_flag/%s/"%(qrnn_dir))
        
        
        channels = np.array(d[qrnn_dir])
        
        
        if target not in channels:
            inChannels = np.concatenate([[target], channels])
        else:
            inChannels = channels
        
        print(qrnn_dir, channels, inChannels)
            
        qrnn_file = os.path.join(qrnn_path, "qrnn_mwhs_%s.nc"%(target))
        
        i183, = np.argwhere(inChannels == target)[0]
        
        y_pre, y_prior, y0, y, y_pos_mean = read_qrnn(qrnn_file, test_file, \
                                                      inChannels, target)
        im1 = (np.abs(y_pre[:, 3] - y_prior[:, i183] )< 5) 
        
        data = mwhsData(test_file, 
                   inChannels, target, ocean = False, test_data = True) 
     
    #%%     SI approach
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
    
#%%  calibration of ECMWF error model
    TB_ob = TB_ob[:, data.im]
    obs_err = np.abs(TB_ob[14, im] - y0[im])
    err = err[:, data.im]
#    mean,std = norm.fit(err[14, im])
#    mae = np.mean(err[14, :])
#    sigma_mae  = np.sqrt(np.pi / 2.0) * mae
    interval = norm.interval(quantiles, loc = 0, scale = 1.0)[1]

    y_pre = np.zeros([err[14, im].shape[0], 7])  
    for i in range(7):
        y_pre[:, i] = interval[i] * err[14, im]

    intervals = []    

    for i in range(1, 6):
        intervals.append(quantiles[i] - quantiles[0] )
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    a5 = 0
    a6 = 0
    for i in range(len(obs_err)):
        if  obs_err[i] <= y_pre[i, 1]: 
                   a1 += 1
        if  obs_err[i] <= y_pre[i, 2]:
                   a2 += 1               
        if  obs_err[i] <= y_pre[i, 3]:
                   a3 += 1
        if  obs_err[i] <= y_pre[i, 4]:
                   a4 += 1
        if  obs_err[i] <= y_pre[i, 5]:
                   a5 += 1
        if  obs_err[i] <= y_pre[i, 6]:
                   a6 += 1
            
    
    x = np.arange(0, 1, 0.01)
    y = x
        
    fig, ax = plt.subplots(1, 1, figsize = [8,8])       
    (ax.plot(intervals[:], [ a1/len(y0[im]), a2/len(y0[im]), a3/len(y0[im]), 
                           a4/len(y0[im]), a5/len(y0[im])
                          ], 'r.-', ms = 15, linewidth = 2.5))    
    ax.plot(x, y, '--')
          
    
    
    