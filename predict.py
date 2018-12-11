# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:49:34 2018

@author: steve
"""

#==============================================================================
#
# IMPORT LIBRARIES
#
#==============================================================================
import numpy as np
import pandas as pd
import os
from f_dataImport import dataImport
from f_cleanData import cleanData
from f_predict import predictLaunchState
from sklearn.metrics import confusion_matrix, classification_report

#==============================================================================
#
# RAW DATA IMPORT
#
#==============================================================================
            
while True:
    print('\n')
    try:
        choice = int(input('What sort of raw data do you have? (Enter 1 or 2): \n'
                           '  [1] JSON data  \n'
                           '  [2] Dataframe from previously cleaned raw JSON file: '))
    except:
        print('\n')
        print('Input an integer (1 or 2)')
        continue
    else:
        if (choice == 1):
            df = dataImport()
            break
        elif (choice == 2):
            print('\n')
            new_data = str(input('Input the new dataframe filepath you want to predict: '))
        
            if not os.path.exists(new_data):
                print('\n')
                print('The filepath \'', new_data, '\' does not exist.', sep='')
                continue
            else:
                print('\n')
                yesno = str(input(f'Confirm that \'{new_data}\' is the correct '
                                  'filepath (\'y\' or \'n\'): '))
                
                if (yesno[0].lower() == "y"):
                    df = pd.read_csv(new_data, sep=',', na_filter=False, index_col=0)  
                    break
                elif (yesno[0].lower() == 'n'):
                    print('\n')
                    continue
                else:
                    print('\n')
                    print('Improper input')
                    continue
            break
        else:
            print('\n')
            print('Must input 1 or 2')
            continue

#==============================================================================
#
# CLEAN DATA
#
#==============================================================================

df = cleanData(df)

#==============================================================================
#
# EXTRACT OUTCOME
#
#==============================================================================

X = df.drop(columns = 'launch_state')
y = df['launch_state']


#==============================================================================
#
# PREDICT
#
#==============================================================================

y_pred = predictLaunchState(X)

#-----------------------------------------
# WRITE OUT PREDICTIONS
np.savetxt('y_pred.csv', y_pred, delimiter=',')

#==============================================================================
#
# EVALUATE IF APPLICABLE
#
#==============================================================================

if (y.unique().any() == None):
    print('\n')
    print('This appears to be new data. The prediction vector has been created '
          'and is called \'y_pred\'. A comma-separated value copy has been '
          'saved as \'y_pred.csv\'.')
else:
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
    print('\n')
    print('The prediction vector has been created'
          'and is called \'y_pred\'. A comma-separated value copy has been '
          'saved as \'y_pred.csv\'.')
