# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:43:36 2018

@author: steve
"""

# ========================================
# IMPORT LIBRARIES
# ========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ========================================
# IMPORT DATAFRAME
# ========================================
df = pd.read_csv('data/df01.csv', sep=',', na_filter=False, index_col=0, 
                 parse_dates=['deadline','created_at','launched_at'])

df.shape
df.info()

# ========================================
# EXPLORATORY DATA ANALYSIS
# ========================================
fig=plt.figure(figsize=(8, 4))
sns.set_style('whitegrid')
sns.countplot(x='launch_state',data=df)

# ---- FURTHER CLEAN-UP ----
df.columns
df['disable_communication'].unique()
# all values are False; delete it
df['staff_pick'].unique()
df['spotlight'].unique()
df['creator_registered'].unique()
# all values are True; delete it
# Also drop 'name' since it's not useful

df.drop(columns=['name', 'disable_communication', 'creator_registered'], inplace=True)
df.shape

# ---- DUMMY VARIABLES ----
# Convert the categorical variables to dummy variables
df.info()
category = pd.get_dummies(df['category'], drop_first=True)
country = pd.get_dummies(df['country'], drop_first=True)
d_launch_state = dict(zip(['failed','successful'], range(0,2)))
launch_state = df['launch_state'].map(d_launch_state)

# Check that launch state got convertec correctly
df[df['launch_state'] == 'successful'].shape[0] - launch_state.sum()

# Drop current categorical columns
df.drop(['category','country','launch_state'],axis=1,inplace=True)

# Add the dummy variables
df = pd.concat([df, country, category, launch_state],axis=1)
df.shape
df.info()
df.isnull().sum().sum() # Check for nulls

# =============================================================================
# Now we need to consider what exactly some of those columns mean:
# * id - primary key
# * pledged - somewhat useless except as a comparison to the goal. Drop 'pledged'
#   but add a 'pledged_ratio' column (pledged/goal)
# * deadline - useless except in comparison to launched_at. Drop deadline but 
#   add funding_days.
# * created_at - useless; launched_at is more applicable.
# * launched_at - keep to maybe create some time series plots.
# * staff_pick - not exactly sure but convert to 0 (false) and 1 (true)
# * spotlight - not exactly sure but covert to 0 (false) and 1 (true)
# =============================================================================

pledged_ratio = df['pledged'] / df['goal']
df.insert(loc=df.columns.get_loc("pledged"), column='pledged_ratio', 
          value=pledged_ratio)
df.drop(columns='pledged', inplace=True)
df.shape
df.columns

funding_days = (df['deadline'] - df['launched_at']).dt.days
df.insert(loc=df.columns.get_loc("deadline"), column='funding_days', 
          value=funding_days)
df.drop(columns='deadline', inplace=True)
df.drop(columns='created_at', inplace=True)
df.shape
df.columns

launched_at = df['launched_at']
df.drop(columns='launched_at', inplace=True)
df.insert(loc=df.columns.get_loc("funding_days"), column='launched_at', 
          value=launched_at)

d_staff_pick = dict(zip([False,True], range(0,2)))
staff_pick = df['staff_pick'].map(d_staff_pick)
df[df['staff_pick'] == True].shape[0] - staff_pick.sum() # check mapping

d_spotlight = dict(zip([False,True], range(0,2)))
spotlight = df['spotlight'].map(d_spotlight)
df[df['spotlight'] == True].shape[0] - spotlight.sum() # check mapping

df.drop(['staff_pick','spotlight'],axis=1,inplace=True)
df.insert(loc=df.columns.get_loc("AF"), column='staff_pick', value=staff_pick)
df.insert(loc=df.columns.get_loc("AF"), column='spotlight', value=spotlight)

df['staff_pick'].unique() # unique check
df['spotlight'].unique() # unique check
df.isnull().sum().sum() # Null value check
df.isna().sum().sum() # NA value check

# ---- EXPLORE ----

# Pairplot of everything EXCEPT for the dummy variables (country and country)
sns.set_style('whitegrid')
sns.pairplot(data=df.drop(df.columns[8:216], axis=1).drop(columns='id'), 
             diag_kind='kde', hue='launch_state')

# ========================================
# LOAD/SAVE CSV
# ========================================
df.to_csv('data/df02.csv', sep=",")
