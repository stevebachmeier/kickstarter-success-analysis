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
from sklearn.metrics import confusion_matrix, classification_report

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
def predictLaunchState(df):
    '''
    This function accepts a new data dataframe and uses a chosen machine 
    learning model to predict whether each observation will be successfully
    funded ('launch_state' = 1) or not ('launch_state' = 0).
    '''
    
    #-----------------------------------------
    # X/y SPLIT
    
    info_variables = ['id','launched_at','category','country', 
                      'pledged_ratio', 'backers_count']
    X = df.drop(columns=info_variables).drop(columns='launch_state')
    y = df['launch_state']
    
    #-----------------------------------------
    # SCALE FEATURES
    X = sc_X.transform(X)   
    
    #-----------------------------------------
    # PREDICT RESULTS
    
    start_clock = time.clock()
    y_pred = classifier_rf_opt.predict(X)
    end_clock = time.clock()
    
    clock_predict = end_clock - start_clock
    print('Runtime, predict: ', round(clock_predict, 2), ' sec', sep='')
    
    #-----------------------------------------
    # EVALUATE MODEL
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Classification report
    cr = classification_report(y, y_pred)
    
    # Accuracy
    acc = cm.diagonal().sum() / cm.sum()
    
    print("\n")
    print("CONFUSION MATRIX:", sep="")
    print(cm)
    print("\n")
    print("CLASSIFICATION REPORT:", sep="")
    print(cr)
    print("\n")
    print("ACCURACY: ", round(acc, 2), sep="")
    
    return y_pred