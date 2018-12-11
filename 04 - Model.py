# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:08:37 2018

@author: steve
"""

#==============================================================================
#
# USER INPUTS
#
#==============================================================================
new_data = 'data/df_v.csv' # New data to predict on

#==============================================================================
#
# PICKLING
#
#==============================================================================
import dill
sc_X = dill.load(open("sc_X.pkl", "rb"))
classifier_rf_opt = dill.load(open("classifier_rf_opt.pkl", "rb"))

#==============================================================================
#
# IMPORT LIBRARIES
#
#==============================================================================

import pandas as pd
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

#==============================================================================
#
# IMPORT DATA FRAMES
#
#==============================================================================

df = pd.read_csv(new_data, sep=',', na_filter=False, index_col=0)

#==============================================================================
#
# CLEAN DATA
#
#==============================================================================

#-----------------------------------------
# COLUMN CLEANUP

drop_vars = ['photo_1024x576', 'photo_1536x864', 'photo_ed', 'photo_full', 
             'photo_key', 'photo_little', 'photo_med', 'photo_small', 
             'photo_thumb', 'slug', 'urls_api.message_creator', 'urls_api.star', 
             'urls_web.message_creator', 'urls_web.project', 'urls_web.rewards', 
             'source_url', 'creator_avatar.medium', 'creator_avatar.small', 
             'creator_avatar.thumb', 'creator_chosen_currency', 'creator_id', 
             'creator_name', 'creator_slug', 'creator_urls.api.user',
             'creator_urls.web.user', 'location_id', 'location_name', 
             'location_slug', 'location_short_name', 'location_displayable_name', 
             'location_localized_name', 'location_type', 'location_is_root', 
             'location_urls_web_discover', 'location_urls_web_location', 
             'location_urls_api_nearby_projects', 'category_color', 
             'category_id', 'category_urls.web.discover', 
             'profile_background_color', 
             'profile_background_image_attributes.id', 
             'profile_background_image_attributes.image_urls.baseball_card', 
             'profile_background_image_attributes.image_urls.default',
             'profile_background_image_opacity', 'profile_blurb', 
             'profile_feature_image_attributes.id', 
             'profile_feature_image_attributes.image_urls.baseball_card',
             'profile_feature_image_attributes.image_urls.default', 'profile_id',
             'profile_link_background_color', 'profile_link_text', 
             'profile_link_text_color', 'profile_link_url', 'profile_name', 
             'profile_project_id', 'profile_should_show_feature_image_section', 
             'profile_show_feature_image', 'profile_state', 
             'profile_state_changed_at', 'profile_text_color', 'currency_symbol',
             'static_usd_rate','converted_pledged_amount','fx_rate',
             'current_currency', 'usd_pledged', 'is_starrable', 'friends', 
             'is_backing', 'is_starred', 'permissions', 'name', 'blurb',
             'location_state', 'location_country', 'currency', 'currency_trailing_code', 
             'state_changed_at', 'category_parent_id', 'category_position', 
             'category_name', 'category_id', 'creator_is_registered', 
             'disable_communication', 'created_at', 'usd_type']

df.drop(columns=drop_vars, inplace=True)

# Rename columns
df.rename(columns={'category_slug':'category', 'state':'launch_state'}, inplace=True)

# Rearrange columns
df = df[['launch_state', 'id', 'category', 'goal', 'backers_count', 
         'pledged', 'country','deadline', 'launched_at', 
         'staff_pick', 'spotlight']]

#-----------------------------------------
# EXTRACT CATEGORIES
df['category'] = [i.split('/')[0] for i in df['category']]

#-----------------------------------------
# REMOVE DUPLICATES
df.drop_duplicates(inplace=True)
# For duplicate IDs leftover, remove the lesser pledged row
df = df.sort_values('pledged', ascending=False).drop_duplicates('id').sort_index()

# Check
if (len(df) - len(df["id"])) != 0:
    print('*** WARNING: There are ',
          len(df) - len(df["id"]), 
          ' duplicate IDs ***', sep='')

# Re-index
df.reset_index(drop=True, inplace=True)

#-----------------------------------------
# CONVERT DATETIMES
df['deadline'] = df['deadline'].apply(datetime.utcfromtimestamp)
df['launched_at'] = df['launched_at'].apply(datetime.utcfromtimestamp)

#-----------------------------------------
# NA IMPUTATION 
# Checks
if df.isnull().sum().sum() != 0:
    print('*** WARNING: There are null values ***')
if df.isna().sum().sum() != 0:
    print('*** WARNING: There are NA values ***')
if (df=='').sum().sum() != 0:
    print('*** WARNING: There are empty string (\'\') values ***')

#-----------------------------------------
# CLEAN UP 'launch_state'
df.query("launch_state == 'failed' | launch_state == 'successful'", inplace=True)
df.reset_index(drop=True, inplace=True)

#-----------------------------------------
# CONVERT CATEGORICAL VARIABLES TO DUMMY VARIABLES 
category = pd.get_dummies(df['category'], drop_first=True)
country = pd.get_dummies(df['country'], drop_first=True)
d_launch_state = dict(zip(['failed','successful'], range(0,2)))
launch_state = df['launch_state'].map(d_launch_state)

# Check
if (df[df['launch_state'] == 'successful'].shape[0] - launch_state.sum() != 0):
    print('*** WARNING: Some launch_states did not map to 0/1 ***')

# Drop the categorical launch_state column 
# (keep 'category' and 'country' for  visualization)
df.drop(['launch_state'],axis=1,inplace=True)

# Add the new dummy variable launch_state column and move it to column index 0 
# and country to column index 3
df = pd.concat([launch_state, df], axis=1)
df = df[['launch_state', 'id', 'category', 'country', 'goal', 'backers_count', 
         'pledged','deadline', 'launched_at', 'staff_pick', 'spotlight']]

# Add the dummy variable country and category columns
df = pd.concat([df, category, country], axis=1)

# Checks
if (df.isnull().sum().sum() != 0):
    print('*** WARNING: Null values introduced with dummy variables ***')
if (df.isna().sum().sum() != 0):
    print('*** WARNING: NA values introduced with dummy variables ***')
if (df=='').sum().sum() != 0:
    print('*** WARNING: Empty string (\'\') values introduced with dummy variables ***')

#-----------------------------------------
# FINAL CLEANUP
# pledged_ratio
pledged_ratio = df['pledged'] / df['goal']
df.insert(loc=df.columns.get_loc("pledged"), column='pledged_ratio', 
          value=pledged_ratio)
df.drop(columns='pledged', inplace=True)

# datetime columns
funding_days = (df['deadline'] - df['launched_at']).dt.days
df.insert(loc=df.columns.get_loc("deadline"), column='funding_days', 
          value=funding_days)
df.drop(columns='deadline', inplace=True)

# ---- MOVE 'LAUNCHED_AT' ----
launched_at = df['launched_at']
df.drop(columns='launched_at', inplace=True)
df.insert(loc=2, column='launched_at', value=launched_at)

# ---- CONVERT 'STAFF_PICK' AND 'SPOTLIGHT' TO DUMMIES ----
d_staff_pick = dict(zip([False,True], range(0,2)))
staff_pick = df['staff_pick'].map(d_staff_pick)

# Check
if (df[df['staff_pick'] == True].shape[0] - staff_pick.sum()) != 0:
    print('*** WARNING: \'staff_pick\' not mapped to 0/1 properly ***')

d_spotlight = dict(zip([False,True], range(0,2)))
spotlight = df['spotlight'].map(d_spotlight)

# Check
if (df[df['spotlight'] == True].shape[0] - spotlight.sum()) != 0:
    print('*** WARNING: \'spotlight\' not mapped to 0/1 properly ***')

df.drop(['staff_pick','spotlight'],axis=1,inplace=True)

df.insert(loc=df.columns.get_loc("comics"), column='staff_pick', value=staff_pick)
df.insert(loc=df.columns.get_loc("comics"), column='spotlight', value=spotlight)

# Checks
if (df.isnull().sum().sum() != 0):
    print('*** WARNING: Null values introduced with \'staff_pick\' and \'spotlight\' dummy variables ***')
if (df.isna().sum().sum() != 0):
    print('*** WARNING: NA values introduced with \'staff_pick\' and \'spotlight\' dummy variables ***')
if (df=='').sum().sum() != 0:
    print('*** WARNING: Empty string (\'\') values introduced with \'staff_pick\' and \'spotlight\' dummy variables ***')

#-----------------------------------------
# VARIABLE REDUCTION
df.drop(columns='spotlight', inplace=True)

#==============================================================================
#
# X/Y SPLIT
#
#==============================================================================

info_variables = ['id','launched_at','category','country', 
                  'pledged_ratio', 'backers_count']
X = df.drop(columns=info_variables).drop(columns='launch_state')
y = df['launch_state']

#==============================================================================
#
# PREDICT
#
#==============================================================================

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

#-----------------------------------------
# SAVE CSV

#df.to_csv('data/df01.csv', sep=",")

