#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:06:54 2020

@author: inderpreet
"""

import numpy as np
import ICI.stats as S
from mwhs import mwhsData
from typhon.retrieval.qrnn import set_backend, QRNN
set_backend("pytorch")

def read_qrnn(qrnn_file, test_file,  inChannels, target):

    data = mwhsData(test_file, 
                   inChannels, target, ocean = False, test_data = True)  

    qrnn = QRNN.load(qrnn_file)
    y_pre, y_prior, y0, y, y_pos_mean = S.predict(data, qrnn, \
                                                  add_noise = False)
    
    return y_pre, y_prior, y0, y, y_pos_mean
