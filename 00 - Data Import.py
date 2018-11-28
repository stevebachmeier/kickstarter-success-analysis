# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:06:31 2018

@author: steve
"""

# ========================================
# USER INPUT
# ========================================
file = 'data/Kickstarter_2018-10-18T03_20_48_880Z/Kickstarter_2018-10-18T03_20_48_880Z.json'

# ========================================
# IMPORT LIBRARIES
# ========================================
import pandas as pd
from pandas.io.json import json_normalize 
pd.options.display.max_columns = None # Shows all columns
import json
from sklearn.model_selection import train_test_split

# ========================================
# READ IN RAW DATA
# ========================================
with open(file, encoding="utf8") as json_file:
    json_obj = [json.loads(line) for line in json_file]
    
# ========================================
# UNPACK RAW DATA
# ========================================
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
    
# ========================================
# WORKING/VALIDATION SPLIT
# ========================================
X, X_v, y, y_v = train_test_split(df_raw.drop(columns=['state']), 
                                  df_raw['state'], test_size=0.2, 
                                  random_state=101)

df00 = pd.concat([y, X], axis=1) # Working set
df_v = pd.concat([y_v, X_v], axis=1) # Validation set
    
# ========================================
# WRITE OUT
# ========================================
#df_raw.to_csv('data/df_raw.csv', sep=",")
#df_v.to_csv('data/df_v.csv', sep=",")
#df00.to_csv('data/df00.csv', sep=",")