#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:24:04 2020

@author: inderpreet
"""
import numpy as np
import matplotlib.pyplot as plt

def calibration(y_pre, y0, im, quantiles):
    """
    

    Parameters
    ----------
    y_pre : all predicted quantiles
    y0 : noise free clear-sky simulations
    im : array containing indices of all cases tobe used in calibration plot
    quantiles : quantiles used

    Returns
    -------
    frequencies (a) for all intervals 

    """
    
    intervals = []
    for i in range(1, 6):
        intervals.append(quantiles[i] - quantiles[0] )
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    a5 = 0
    a6 = 0
    for i in im:
        if np.logical_and(y0[i] > y_pre[i, 0] ,  y0[i] <=y_pre[i, 1]):
                   a1 += 1
        if np.logical_and(y0[i] > y_pre[i, 0] ,  y0[i] <= y_pre[i, 2]):
                   a2 += 1               
        if np.logical_and(y0[i] > y_pre[i, 0] ,  y0[i] <= y_pre[i, 3]):
                   a3 += 1
        if np.logical_and(y0[i] > y_pre[i, 0] ,  y0[i] <= y_pre[i, 4]):
                   a4 += 1
        if np.logical_and(y0[i] > y_pre[i, 0] ,  y0[i] <= y_pre[i, 5]):
                   a5 += 1
        if np.logical_and(y0[i] > y_pre[i, 0] ,  y0[i] <= y_pre[i, 6]):
                   a6 += 1
                   
                   
                   
    return a1, a2, a3, a4, a5, a6, intervals 

