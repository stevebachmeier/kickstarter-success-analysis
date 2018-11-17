# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:31:42 2018

@author: steve
"""

# ========================================
# IMPORT LIBRARIES
# ========================================
import pandas as pd
import numpy as np
import json
from datetime import datetime
import seaborn as sns

# ========================================
# READ/EXTRACT RELEVANT DATA
# ========================================
file = 'data/Kickstarter_2018-10-18T03_20_48_880Z/Kickstarter_2018-10-18T03_20_48_880Z.json'
with open(file, encoding="utf8") as json_file:
    json_obj = [json.loads(line) for line in json_file]

# ---- QUICK EXPLORATION ----
type(json_obj)
len(json_obj)
type(json_obj[0])
len(json_obj[0])
json_obj[len(json_obj)-1] # look at final entry
json_obj[0].keys()

# ---- DATA EXTRACTION ----
# Extract the 'data' key-value pairs
json_obj2 = [] # pre-allocate
# append 'data' dictionary only
for x in range(0, len(json_obj)):
    json_obj2.append(json_obj[x]["data"])
    
len(json_obj2) - len(json_obj) # Check that all rows extracted

# =============================================================================
# We can remove columns that we know or highly suspect will not help us
# predict project status.
#
# Remove probably useless keys as well as nested-dictionary keys (to be added 
# back later if desired): 
#   - Useless: photo, slug, urls, source_url
#   - Nested: creator, location, category, profile (to maybe be added later)
# =============================================================================

keys=('id', 'name', 'blurb', 'goal', 'pledged', 'state', 'disable_communication', 'country', 'currency', 'currency_symbol',
      'currency_trailing_code', 'deadline', 'state_changed_at', 'created_at', 'launched_at', 'staff_pick', 'is_starrable',
      'backers_count', 'static_usd_rate', 'usd_pledged', 'converted_pledged_amount', 'fx_rate', 'current_currency', 'usd_type',
      'spotlight')

# pre-allocate
json_obj3 = []
# Append
for x in range(0, len(json_obj2)):
    json_obj3.append({k:json_obj2[x][k] for k in keys})
    
# Add back useful sub-nested dictionary entries
# Previously removed: 'creator', 'location', 'category', 'profile'

# ---- Explore 'creator' ----
# Grab useful previously removed sub key-value pairs:
json_obj2[0]["creator"].keys()

# Useful keys are: 'is_registered'
# Add 'is_registered' to json_obj3
for x in range(0, len(json_obj3)):
    json_obj3[x]["creator_registered"] = json_obj2[x]["creator"]["is_registered"]

# ---- Explore 'location' ----
json_obj2[0]["location"].keys()

# Useful keys: country, state
for x in range(0, len(json_obj3)):
    # Extract country and state data
    if "location" in json_obj2[x]:
        json_obj3[x]["loc_country"] = json_obj2[x]["location"]["country"]
        json_obj3[x]["loc_state"] = json_obj2[x]["location"]["state"]
    
    # Add None when country/state data does not exist
    else:
        json_obj3[x]["loc_country"] = np.float64('nan')
        json_obj3[x]["loc_state"] = np.float64('nan')

# ---- Explore 'category' ----
# Grab useful previously removed sub key-value pairs:
json_obj2[0]["category"].keys()

# Useful keys: name, slug, position, parent_id
# Check for null entries
x_category = 0
for x in range(0, len(json_obj2)):
    if "category" in json_obj2[x]:
        x_category = x_category + 1
    else:
        break
x_category - len(json_obj2) # Null entries check
 # 0 null entries

x_name = 0
for x in range(0, len(json_obj2)):
    if "name" in json_obj2[x]["category"]:
        x_name = x_name + 1
x_name - len(json_obj2) # Null entries check
 # 0 null entries

x_slug = 0
for x in range(0, len(json_obj2)):
    if "slug" in json_obj2[x]["category"]:
        x_slug = x_slug + 1
x_slug - len(json_obj2) # Null entries check
 # 0 null entries

x_position = 0
for x in range(0, len(json_obj2)):
    if "position" in json_obj2[x]["category"]:
        x_position = x_position + 1
x_position - len(json_obj2) # Null entries check
 # 0 null entries

x_parent_id = 0
for x in range(0, len(json_obj2)):
    if "parent_id" in json_obj2[x]["category"]:
        x_parent_id = x_parent_id + 1
x_parent_id - len(json_obj2)
# 17378 null entries

# Add columns
for x in range(0, len(json_obj3)):
    # Extract name, slug, position, and parent_id data
    json_obj3[x]["category_name"] = json_obj2[x]["category"]["name"]
    json_obj3[x]["category_slug"] = json_obj2[x]["category"]["slug"]
    json_obj3[x]["category_position"] = json_obj2[x]["category"]["position"]
    if "parent_id" in json_obj2[x]["category"]:
        json_obj3[x]["category_parent_id"] = json_obj2[x]["category"]["parent_id"]
    # Add NaN when data does not exist
    else:
        json_obj3[x]["category_parent_id"] = np.float64('nan')
        
# ---- Explore 'profile' ----
# Grab useful previously removed sub key-value pairs:
json_obj2[0]["profile"].keys()

# check to see how many profile states are 'inactive'
x_inactive = 0
for x in range(0, len(json_obj2)):
    if json_obj2[x]["profile"]["state"] == 'inactive':
        x_inactive += 1
x_inactive

# check to see how many profile states are 'active'
x_active = 0
for x in range(0, len(json_obj2)):
    if json_obj2[x]["profile"]["state"] == 'active':
        x_active += 1
x_active

# Check that 'active' and 'inactive' are the only two possible profile states
len(json_obj2) - x_active - x_inactive

# =============================================================================
# There are only two options for state: active and inactive - there are many 
# inactive profile states

# Useful keys: none
# This is a bit of a guess. Most of the profile states are labeled as 'inactive'.
# My guess is that the profiles go latent once a projct is finished (perhaps 
# regardless of whether it was successful or not)
# =============================================================================

# ---- CONVERT TO DATA FRAME ----

# Create initial (dirty) dataframe (for future reference)
df00 = pd.DataFrame.from_records(json_obj3)
df00.to_csv("data/df00.csv", sep=",") # Write out to csv

# Create working dataframe
df = pd.DataFrame.from_records(json_obj3)
df.head()
df.tail()
df.describe()
df.info()
# Note that most of the data frame is already non-null

# ========================================
# CLEAN DATA
# ========================================

# ---- REARRANGE COLUMNS ----
df.columns

# Check the original column arrangement
json_obj3[0].keys()

# Re-order columns
df = df[['id', 'name', 'blurb', 'category_name', 'category_slug', 'category_position', 'category_parent_id', 'goal', 'pledged', 
         'disable_communication', 'loc_country', 'loc_state', 'country', 'currency', 'currency_symbol', 'currency_trailing_code', 'deadline', 
         'state_changed_at', 'created_at', 'launched_at', 'staff_pick', 'is_starrable', 'backers_count', 'static_usd_rate', 
         'usd_pledged', 'converted_pledged_amount', 'fx_rate', 'current_currency', 'usd_type', 'spotlight', 
         'creator_registered', 'state']]

# ---- REMOVE DUPLICATES ----
df.sort_values(by=["backers_count"],ascending=False)[["id","name","backers_count"]]
df.shape
df.drop_duplicates(inplace=True)
df.shape
# We dropped 205696-189240=16456 duplicate rows

# Search for more duplicates (based on id)
len(df["id"].unique()) - len(df)
# We still have 2166 duplicates

# Explore the duplicate ID rows
dupes = pd.concat(g for _, g in df.groupby("id") if len(g) > 1)
dupes

# =============================================================================
# It looks like some of the rows look different due to usd_pledged, 
# converted_pledged_amount, fx_rate (which appears to be exchange rate), and 
# current_currency. Let's just delete the following currency-related columns:
# currency_symbol, static_usd_rate, convertd_pledged_amount, fx_rate, 
# current_currency, usd_type
# =============================================================================

df.drop(columns=['currency_symbol','static_usd_rate','converted_pledged_amount','fx_rate','current_currency','usd_type'], 
        inplace=True)
df.shape

df.drop_duplicates(inplace=True)
df.shape
# We dropped 189240-187367=1873 duplicate rows

# Search for more duplicates
len(df["id"].unique()) - len(df)
# There are still 293 duplicate IDs
# It looks like there are differences with pledged, backers_count and usd_pledged

# usd_pledged is redundant with pledged; remove usd_pledged
df.drop(columns=['usd_pledged'], inplace=True)

df.shape

df.drop_duplicates(inplace=True)

df.shape

# we dropped 187367-187345=22 duplicate rows

# Search for more duplicates
len(df["id"].unique()) - len(df)
# There are 271 duplicate ID rows

# is_starrable is unclear what it is; drop it
df.drop(columns=['is_starrable'], inplace=True)

df.shape
df.drop_duplicates(inplace=True)
df.shape
# We dropped a whopping one row.

# Search for more duplicates
len(df["id"].unique()) - len(df)

# =============================================================================
# With 270 duplicate ID rows left and out of things I'd like to drop, I'm 
# left with simply removing the duplicates. The drop_duplicates method is not
# removing them due to differences between pledged and backers_count. Without 
# some sort of a time stamp we don't know which one is the most updated row. 
# Therefore, let's look for duplicate IDs, then keep the rows with the largest 
# pledge values (which are probably also the most current rows).
# =============================================================================

df = df.sort_values('pledged', ascending=False).drop_duplicates('id').sort_index()

# Search for more duplicates
len(df["id"].unique()) - len(df)
# No more duplicates

# Re-index
df.reset_index(drop=True, inplace=True)

df.info()

# =============================================================================
# ---- FURTHER REFINE VARIABLES ----
# NOTES:
# delete:
# * blurb
# * loc_state (too granular)
# * country (not sure how it differs from loc_country; largely redundant)
# * currency (pledges is in usd)
# * currency_trailing_code (what is it?)
# * state_changed_at
# 
# KEEP:
# * id (primary key)
# * name (for reference)
# * category_name; change to "sub_category"
# * category slug; extract first word, change to "category"
# * category_position; change to "sub_category_id", move before sub_category
# * category_parent_id; change to "category_id", move before category
# * goal
# * pledged
# * disable_communication (but what is it??)
# * loc_country; rename as "country"
# * deadline; convert to datetime
# * created_at; convert to datetime
# * launched_at; convert to datetime
# * staff_pick
# * backers_count; move before pledged
# * spotlight (but what is it?)
# * creator_registered
# * state
# =============================================================================

# delete columns
df.drop(columns=['blurb','loc_state','country','currency',
                 'currency_trailing_code','state_changed_at'], inplace=True)

# Rename columns
df.rename(columns={'category_name':'sub_category', 'category_slug':'category', 
                   'category_position':'sub_category_id', 
                   'category_parent_id':'category_id', 'loc_country':'country', 
                   'state':'launch_state'}, inplace=True)

# Re-arrange columns
df = df[['id', 'name', 'sub_category_id', 'sub_category', 'category_id', 
         'category', 'goal', 'backers_count', 'pledged', 'disable_communication', 
         'country','deadline', 'created_at', 'launched_at', 'staff_pick', 
         'spotlight', 'creator_registered', 'launch_state']]

# There does not seem to be a strong correlation between sub_category_id and 
# sub_category. Let's drop the sub_category_id and (maybe) keep sub_category
df.sort_values('sub_category').sub_category.unique()
df.sort_values('category').category.unique()

# =============================================================================
# CATEGORY NOTES
# So, what do we keep? It doesn't make much sense to keep sub-categories without 
# the parent category. And for now, dealing with so many categories to begin 
# with it might become overwhelming trying to manage sub-categories too. Let's 
# start with just categories and see if we can make some good predictions (and 
# remember that we can always beef up the algorithm later by adding in the 
# sub-category variables).
#
# As for the category variables, do we want to extract the parent category 
# (eg 'technology') or do we want to keep the sub-categories tacked on 
# (eg 'technology/3d printing' and 'technology/apps')? For now, as mentioned 
# above, let's drop the sub-categories (eg 'technology/3d printing' and 
# 'technology/apps' are both just considered 'technology'). 
# 
# We now have two choices: keep the category as categorical or use the IDs. 
# For context, let's keep the categories as categorical and use dummy variables 
# later as needed.
# =============================================================================

df.drop(columns=['sub_category_id','sub_category','category_id'], inplace=True)

# Now let's extract the primary category from the category column
df.category = [i.split('/')[0] for i in df.category]
df.sort_values(by='category').category.unique() # Check

df.info()

# Convert date values to datetime 
# The default format of these values is in unix time format
df['deadline'] = df['deadline'].apply(datetime.utcfromtimestamp)
df['created_at'] = df['created_at'].apply(datetime.utcfromtimestamp)
df['launched_at'] = df['launched_at'].apply(datetime.utcfromtimestamp)
df.info()

# ---- NA IMPUTATION ----
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

df.isnull().sum(axis=0)
# the only null values are the 876 in the 'country' column

# Retrieve list of project IDs that have country=NaN
null_country_IDs = df[df["country"].isnull().values]["id"].values

# Check if country data can be found from the original (dirty) df00
df00[df00['id'].isin(null_country_IDs)].sort_values(by='id').drop_duplicates('id')["country"].unique()
# They are all US! 

df['country'].fillna('US', inplace=True)

df.isnull().sum(axis=0)
# No more null values in the data frame

# ---- FURTHER CLEAN UP OF 'launch_state' ----
df['launch_state'].unique()
# There are five launch states: failed, successful, canceled, live, and suspended
sns.set_style('whitegrid')
sns.countplot(x='launch_state',data=df)

# =============================================================================
# Let's delete projects with launch state of canceled, live, or suspended.
# * The number of canceled, live, and suspended is far less than failed and successful
# * Live projects are not finished and so cannot be used as data to predict success or failure
# * It is unknown why projects were canceled or suspended or whether or not they were eventually re-launched or unsuspended.
# =============================================================================

df.query("launch_state == 'failed' | launch_state == 'successful'", inplace=True)
df.reset_index(drop=True, inplace=True)

sns.set_style('whitegrid')
sns.countplot(x='launch_state',data=df)

# ========================================
# SAVE CSV
# ========================================
df.to_csv('data/df01.csv', sep=",")

