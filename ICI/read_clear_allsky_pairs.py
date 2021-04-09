#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:52:21 2020

@author: inderpreet
"""
import os
import xarray

def read_clear_allsky_pairs(files_clearsky):
    """
    find the pairs of clearsky and allsky cases from ARTS simulations 

    Parameters
    ----------
    files_clearsky : list containing clearsky files

    Returns
    -------
    y_ici_cs : TB for clearsky cases
    y_ici_as : TB for all sky cases
    """

    first_iteration = True
    
    for file_clearsky in files_clearsky[:]:
        file_allsky = file_clearsky.replace('_clearsky.nc', '.nc')
        
        if os.path.isfile(file_allsky):
    # check if both files exist      
    #        print (file_allsky)
            y = xarray.open_dataset(file_allsky)
            y_ici_allsky = y.y_ici
            
            y = xarray.open_dataset(file_clearsky)
    #        print(file_clearsky)
            y_ici_clearsky = y.y_ici
            
            
            allsky = y_ici_allsky.shape
            clearsky = y_ici_clearsky.shape
            
            if allsky > clearsky:
    # ('allsky measurements are more')
                y_ici_allsky = y_ici_allsky[:clearsky[0], :]
            
            elif clearsky > allsky:
    # ('clearsky measurements are more')
                y_ici_clearsky = y_ici_clearsky[:allsky[0], :]
                
            if first_iteration:  
    # initialise the xarray DataArray            
                y_ici_cs = y_ici_clearsky
                y_ici_as = y_ici_allsky
                first_iteration = False
            else:
                y_ici_cs = xarray.concat([y_ici_cs, y_ici_clearsky], dim = 'cases')
                y_ici_as = xarray.concat([y_ici_as, y_ici_allsky], dim = 'cases')
    print(y_ici_cs.shape)
    print(y_ici_as.shape)
    
    return y_ici_cs, y_ici_as

def read_clear_allsky_pairs_MWI(files_clearsky):
    
    dict_ici = {"ici_channels":"channels", "ici_stokes_dim":"stokes_dim"}
    dict_mwi = {"mwi_channels":"channels", "mwi_stokes_dim":"stokes_dim"}
    first_iteration = True
    for file_clearsky in files_clearsky[:]:
        file_allsky       = file_clearsky.replace('_clearsky.nc', '.nc')
        
        file_clearsky_mwi = file_clearsky.replace('ICI', 'MWI')
        file_allsky_mwi   = file_allsky.replace('ICI', 'MWI')
        
        files = [file_allsky, file_clearsky_mwi, file_allsky_mwi]
        

        f_exist = [f for f in files if os.path.isfile(f)] 
        if len(f_exist) == 3:
        
#        if os.path.isfile(file_allsky):
    # check if both files exist      
    #        print (file_allsky)
            y = xarray.open_dataset(file_allsky)
            y_ici_allsky = y.y_ici
            
            y = xarray.open_dataset(file_clearsky)
    #        print(file_clearsky)
            y_ici_clearsky = y.y_ici
            
            y = xarray.open_dataset(file_allsky_mwi)
            y_mwi_allsky = y.y_mwi
            
            y = xarray.open_dataset(file_clearsky_mwi)
    #        print(file_clearsky)
            y_mwi_clearsky = y.y_mwi
            
            allsky       = y_ici_allsky.shape[0]
            clearsky     = y_ici_clearsky.shape[0]
            allsky_mwi   = y_mwi_allsky.shape[0]
            clearsky_mwi = y_mwi_clearsky.shape[0]
            
            cases = min(allsky, clearsky, allsky_mwi, clearsky_mwi)
            
            y_ici_allsky = y_ici_allsky[:cases, :]
            y_ici_clearsky = y_ici_clearsky[:cases, :]
            y_mwi_allsky = y_mwi_allsky[:cases, :]            
            y_mwi_clearsky = y_mwi_clearsky[:cases, :]    
        
            
            y_ici_allsky = y_ici_allsky.rename(dict_ici)
            y_ici_clearsky = y_ici_clearsky.rename(dict_ici)
            y_mwi_allsky = y_mwi_allsky.rename(dict_mwi)
            y_mwi_clearsky = y_mwi_clearsky.rename(dict_mwi)
            
            
            y_ici_allsky   = xarray.combine_nested([y_ici_allsky, y_mwi_allsky], concat_dim = ["channels"])            
          
            y_ici_clearsky = xarray.combine_nested([y_ici_clearsky, y_mwi_clearsky], concat_dim = ["channels"])     
            
            if first_iteration:  
    # initialise the xarray DataArray            
                y_ici_cs = y_ici_clearsky
                y_ici_as = y_ici_allsky
                first_iteration = False
            else:
                y_ici_cs = xarray.concat([y_ici_cs, y_ici_clearsky], dim = 'cases')
                y_ici_as = xarray.concat([y_ici_as, y_ici_allsky], dim = 'cases')
    print(y_ici_cs.shape)
    print(y_ici_as.shape)
    
    return y_ici_cs, y_ici_as
