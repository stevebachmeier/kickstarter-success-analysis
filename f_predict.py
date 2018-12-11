# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:25:31 2018

@author: steve
"""
    
#==============================================================================
#
# IMPORT LIBRARIES
#
#==============================================================================
import dill
import time

#==============================================================================
#
# PICKLING
#
#==============================================================================
sc_X = dill.load(open("sc_X.pkl", "rb"))
classifier_rf_opt = dill.load(open("classifier_rf_opt.pkl", "rb"))

#==============================================================================
#
# FUNCTION - PREDICT
#
#==============================================================================
def predictLaunchState(X):
    '''
    This function accepts a new data dataframe and uses a chosen machine 
    learning model to predict whether each observation will be successfully
    funded ('launch_state' = 1) or not ('launch_state' = 0).
    '''
    
    #-----------------------------------------
    # X/y SPLIT
    
    info_variables = ['id','launched_at','category','country', 
                      'pledged_ratio', 'backers_count']
    X = X.drop(columns=info_variables)
    
    #-----------------------------------------
    # SCALE FEATURES
    X = sc_X.transform(X)   
    
    #-----------------------------------------
    # PREDICT RESULTS
    
    start_clock = time.clock()
    y_pred = classifier_rf_opt.predict(X)
    end_clock = time.clock()
    
    clock_predict = end_clock - start_clock
    
    print('\n')
    print('Runtime, predict: ', round(clock_predict, 2), ' sec', sep='')
    
    return y_pred