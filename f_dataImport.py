# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:06:31 2018

@author: steve
"""


#==============================================================================
#
# IMPORT LIBRARIES
#
#==============================================================================
import pandas as pd
import os
from pandas.io.json import json_normalize 
import json

#==============================================================================
#
# FUNCTION - RAW DATA IMPORT
#
#==============================================================================

def dataImport():
    '''
    This function imports the raw json data downloaded from 
    https://webrobots.io/kickstarter-datasets/ and extracts the dictionaries
    into a usable dataframe format.
    
    OUTPUTS:
        * 'df_raw.csv': raw data dataframe
    '''
    
    #-----------------------------------------
    # READ IN RAW DATA
    
    print('\n')
    print('***')
    print('NOTE:')
    print('The input file must be JSON with the same format as those'
          'downloaded from https://webrobots.io/kickstarter-datasets/')
    print('***')
        
    while True:
    
        print('\n')
        new_data = str(input('Input the new data JSON filepath you want to predict: '))
        
        if not os.path.exists(new_data):
            print('\n')
            print('The filepath \'', new_data, '\' does not exist.', sep='')
            continue
        else:
            print('\n')
            yesno = str(input(f'Confirm that \'{new_data}\' is the correct '
                              'filepath (\'y\' or \'n\'): '))
            
            if (yesno[0].lower() == "y"):
                with open(new_data, encoding="utf8") as json_file:
                     json_obj = [json.loads(line) for line in json_file]
                break
            elif (yesno[0].lower() == 'n'):
                print('\n')
                continue
            else:
                print('\n')
                print('Improper input')
                continue
    
    #-----------------------------------------
    # UNPACK RAW DATA
    
    json_obj2 = []
    # append 'data' dictionary only
    for x in range(0, len(json_obj)):
        json_obj2.append(json_obj[x]["data"])
    
    # Check
    if (len(json_obj2) - len(json_obj)) != 0:
        print('*** ERROR: Did not extract all json \'data\' entries ***')

    # ---- CONVERT TO DATAFRAME ----
    df_raw_json = pd.DataFrame(json_obj2)
    
    # ---- UNPACK DICTIONARY ENTRIES ----
    # 'category'
    df_category = json_normalize(data=df_raw_json['category'])
    df_category.columns = 'category_' + df_category.columns
    
    # 'creator'
    df_creator = json_normalize(data=df_raw_json['creator'])
    df_creator.columns = 'creator_' + df_creator.columns
    
    # 'location'
    # Must mannually unpack 'location' with pd.Series due to NaN elements
    df_location = df_raw_json['location'].apply(pd.Series)
    df_location.drop(columns=0, inplace=True)
    df_location.columns = 'location_'+df_location.columns
    df_location1 = df_location['location_urls'].apply(pd.Series)
    df_location1.drop(columns=0, inplace=True)
    df_location1.columns = 'location_urls_'+df_location1.columns
    df_location2 = df_location1['location_urls_web'].apply(pd.Series)
    df_location2.drop(columns=0, inplace=True)
    df_location2.columns = 'location_urls_web_'+df_location2.columns
    df_location3 = df_location1['location_urls_api'].apply(pd.Series)
    df_location3.drop(columns=0, inplace=True)
    df_location3.columns = 'location_urls_api_'+df_location3.columns
    # Concat 'location' dataframes
    df_location = pd.concat([df_location, df_location2, df_location3], axis=1)
    df_location.drop(columns='location_urls', inplace=True)
    
    # 'photo'
    df_photo = json_normalize(data=df_raw_json['photo'])
    df_photo.columns = 'photo_' + df_photo.columns
    
    # 'profile'
    df_profile = json_normalize(data=df_raw_json['profile'])
    df_profile.columns = 'profile_' + df_profile.columns
    
    # 'urls'
    df_urls = json_normalize(data=df_raw_json['urls'])
    df_urls.columns = 'urls_' + df_urls.columns
     
    # ---- CONCAT UNPACKED DATAFRAMES ----
    df_raw = pd.concat([df_raw_json, df_category, df_creator, df_location, 
                        df_photo, df_profile, df_urls], axis=1)
    df_raw.drop(columns=['category','creator','location','photo',
                         'profile','urls'], inplace=True)
    
    #-----------------------------------------
    # WRITE OUT
    
    df_raw.to_csv('data/df_raw.csv', sep=",")
    
    return df_raw