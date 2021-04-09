#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 21:13:37 2020

@author: inderpreet
"""


import numpy as np

def filter_buehler_19(data):
    """
    Filtering with buehler et al criteria

    Parameters
    ----------
    data : MWI dataset containing testing data

    Returns
    -------
    im : logical array for the filtered data

    """
    x = data.add_noise(data.x, data.index)


    TB19 = x[:, 1]
    TB18 = x[:, 2]
    im1 = TB18 < 235.2
    dtb = TB19 - TB18
    im2 = dtb < 0
    
    im = np.logical_or(im1, im2)
    print (np.sum(im1), np.sum(im2))
    return im

def filter_buehler_20(data):
    """
    Filtering with buehler et al criteria

    Parameters
    ----------
    data : MWI dataset containing testing data

    Returns
    -------
    im : logical array for the filtered data

    """
    x = data.add_noise(data.x, data.index)


    TB20 = x[:, 0]
    TB18 = x[:, 2]
    im1 = TB18 < 235.2
    dtb = TB20 - TB18
    im2 = dtb < 0
    
    im = np.logical_or(im1, im2)
    print (np.sum(im1), np.sum(im2))
    return im