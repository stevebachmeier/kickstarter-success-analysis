# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:19:42 2018

@author: steve
"""

# ========================================
# IMPORT LIBRARIES
# ========================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

# ========================================
# IMPORT DATAFRAME
# ========================================
df = pd.read_csv('data/df01.csv', sep=',', na_filter=False, index_col=0, 
                 parse_dates=['launched_at'])

df.shape
df.info()
df.isnull().sum().sum()
df.isna().sum().sum()

# ========================================
# VARIABLE REDUCTION
# ========================================
info_variables = ['id','launched_at','category','country']

# ---- ZERO-VARIANCE ----
sel = VarianceThreshold(threshold=0.0)
print('Number of zero-variance variables: ',
      (sel.fit_transform(X=df.drop(columns=info_variables)).shape[1])-
          (df.drop(columns=info_variables).shape[1]))

# ---- LOW-VARIANCE ----
# =============================================================================
# There is much discussion regarding whether or not low-variance variables
# should be removed from the model; there are times where even near-zero
# variance predictors have a strong influence on the outcome. We will not
# remove such variables.
# =============================================================================

# ---- VARIABLE-OUTCOME CORRELATIONS ----
corr_threshold = 0.5 # Correlation lower threshold
df_corr_variable_outcome = pd.DataFrame(abs(df.drop(
        columns=info_variables).corr().drop(
                columns='launch_state').iloc[0])).reset_index()
df_corr_variable_outcome.columns = ['variable','corr_launch_state']

df_corr_variable_outcome.sort_values(by='corr_launch_state', ascending=False).head()

plt.scatter(x=df_corr_variable_outcome.index,
            y=df_corr_variable_outcome['corr_launch_state'],s=200)
plt.axhline(y=0.5, color='r', linestyle='--', lw=4)

df_corr_variable_outcome[
        df_corr_variable_outcome['corr_launch_state'] > 0.5]['variable']

# =============================================================================
# The only variable of interest here is spotlight. From 
# https://techcrunch.com/2015/03/25/kickstarter-spotlight/, we see that 
# spotlight happens for successfully funded projects and acts as a way to 
# update the project timeline. It clearly does nothing in helping predict 
# funding success; drop it.
# =============================================================================

df.drop(columns='spotlight', inplace=True)

# ---- VARIABLE-VARIABLE CORRELATIONS ----

