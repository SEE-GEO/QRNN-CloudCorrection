#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:44:19 2020

@author: inderpreet
"""
import numpy as np


def add_gaussian_noise(TB, nedt):
    """
    

    Parameters
    ----------
    TB : Input TB array of shape [cases, channels]
    nedt : array conatining NEDT values for each channel in channels
    
    Adds noise to the TB simulations according to NEDT

    Returns
    -------
    BT_noise : array containing TB values with gaussian noise

    """
    
    TB_noise = TB.copy()
    
    nchannels = len(nedt)
    size_TB = TB.shape[0]

    noise = np.array([np.random.normal(0, nedt[i], size_TB) for i in range(nchannels)]).transpose()
        
    TB_noise += noise        
    return TB_noise