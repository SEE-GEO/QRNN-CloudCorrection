#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:57:41 2020

@author: inderpreet

calculate statistics for MWHS point estimates, the results are given in latex format
results from scattering index and beuhler et al
"""

import netCDF4
import os
import matplotlib.pyplot as plt
import numpy as np
import ICI.stats as S
from read_qrnn import read_qrnn
plt.rcParams.update({'font.size': 26})
from tabulate import tabulate
from mwhs import mwhsData
from scipy.stats import skew
#%%


def get_SI_land(y_ob, y_fg, i89, i150):
    """
    compute scattering index over land
    """
    SI_ob = y_ob[i89, :] - y_ob[i150, :]
    SI_fg = y_fg[i89, :] - y_fg[i150, :]
    return (SI_ob + SI_fg)/2

def get_SI_ocean(y_ob, y_fg, y_cl, i89, i150):
    """
    compute scattering index over ocean
    """
    SI_ob = y_ob[i89, :] - y_ob[i150, :] -(y_cl[i89, :] - y_cl[i150, :])
    SI_fg = y_fg[i89, :] - y_fg[i150, :] - (y_cl[i89, :] - y_cl[i150, :])
    return (SI_ob + SI_fg)/2
    
def bias(y , y0):
    return np.mean(y-y0)

def std(y , y0):
    return np.std(y-y0)

def mae(y, y0):
    return np.mean(np.abs(y-y0))


def filter_buehler_19(TB18, TB19):
    """
    Filtering with buehler et al criteria

    Parameters
    ----------
    data : MWI dataset containing testing data

    Returns
    -------
    im : logical array for the filtered data

    """
#   x = data.add_noise(data.x, data.index)
    
    im1 = TB18 < 240.0
    dtb = TB19 - TB18
    im2 = dtb < 0
    
    im = np.logical_or(im1, im2)
    print (np.sum(im1), np.sum(im2))
    return im

if __name__ == "__main__":
#%% input parameters
    depth     = 3
    width     = 128
    quantiles = np.array([0.002, 0.03, 0.16, 0.5, 0.84, 0.97, 0.998])
    batchSize = 128
    
    targets = [11, 12, 13, 14, 15]
    targets = [15]
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
    
    Channels = [[1, 10], [1, 6, 7, 10]]
    qrnn_dirs = ["C89+150"]
    #qrnn_dirs = ["C89+150", "C89+150+118", "C150" ]
    
    path = os.path.expanduser('~/Dendrite/Projects/AWS-325GHz/MWHS/data/')
    
    allChannels = np.arange(1, 16, 1)
    
    #%%
    
    if __name__ == "__main__":
    #%%
        TB_ob = np.load(os.path.join(path, 'TB_obs.npy'))
        TB_fg = np.load(os.path.join(path, 'TB_fg.npy'))
        TB_cl = np.load(os.path.join(path, 'TB_cl.npy'))
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
            
            y_pre, y_prior, y0, y, y_pos_mean = read_qrnn(qrnn_file, test_file, inChannels, target)
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
            
            
            
        #%% Buehler et al approach  
        
            test_file_noise = os.path.join(path, "TB_MWHS_test_noisy_allsky.nc")
        
            file = netCDF4.Dataset(test_file_noise, mode = "r")
            TB_var = file.variables["TB"]
            TB_noise = TB_var[:]
            i18, = np.where(allChannels == 11)[0]
            i19, = np.where(allChannels == 13)[0]
            TB18 = TB_noise[1, i18, data.im].data
            TB19 = TB_noise[1, i19, data.im].data
            
            im18 = np.isfinite(TB18)
            im19 = np.isfinite(TB19)
            im18 = np.logical_and(TB18, TB19)
            im_183 = filter_buehler_19(TB18, TB19)
        #    im_183 = im_183[data.im]
        
            
            #%%     
            
            print ("-----------------channel %s-------------------------"%str(target))
            #        print ("bias uncorr", bias(y_prior[:, i183], y0))
            print ("bias SI", bias(y_fil, y0[im]))
            print ("bias B183", bias(y_prior[~im_183, i183], y0[~im_183]))
            #        print ("bias QRNN", bias(y_prior[im1, i183], y0[im1]))
            #        print ("bias QRNN_corr", bias(y_pre[im1, 3], y0[im1]))
            
            
            #        print ("std uncorr", std(y_prior[:, i183], y0))
            print ("std SI", std(y_fil, y0[im]))
            print ("std B183", std(y_prior[~im_183, i183], y0[~im_183]))
            #        print ("std QRNN", std(y_prior[im1, i183], y0[im1]))
            #        print ("std QRNN_corr", std(y_pre[im1, 3], y0[im1]))
            
            
            #        print ("mae uncorr", mae(y_prior[:, i183], y0))
            print ("mae SI", mae(y_fil, y0[im]))
            print ("mae B183", mae(y_prior[~im_183, i183], y0[~im_183]))
            #        print ("mae QRNN", mae(y_prior[im1, i183], y0[im1]))
            #        print ("mae QRNN_corr", mae(y_pre[im1, 3], y0[im1]))
            
            print ("skew SI", skew(y_fil-y0[im]))
            print ("skew B183", skew(y_prior[~im_183, i183]- y0[~im_183]))
            print ("skew all", skew(y_prior[:, i183]- y0[:]))
            
            print ("% rejected SI", np.sum(~im)/im.shape)
            print ("% rejected B183", np.sum(im_183)/im.shape)
            
            #%%
            bins = np.arange(-30, 20, 0.5)
            hist = np.histogram(y_fil - y0_fil, bins)
            fig, ax = plt.subplots(1, 1)
         #   ax.plot(bins[:-1], hist[0], 'k')
            ax.set_yscale('log')
            
            hist = np.histogram(y_prior[:, i183]- y0 , bins)   
            ax.plot(bins[:-1], hist[0], 'b')
            
            y_pre_fil = y_prior[~im, i183]
            hist = np.histogram(y_pre_fil - y0[~im] , bins)
            ax.plot(bins[:-1], hist[0], 'r')
            
            
            hist = np.histogram(y_pre[im1, 3]- y0[im1], bins)
            ax.plot(bins[:-1], hist[0], 'g')
            
        
            TB_15 = TB_ob[14, data.im]    
            hist = np.histogram(y_prior[im_183, i183] - y0[im_183], bins)
            ax.plot(bins[:-1], hist[0], 'y')
            
            
            
            