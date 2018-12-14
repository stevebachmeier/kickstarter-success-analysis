# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:43:36 2018

@author: steve
"""

#-----------------------------------------
# USER INPUTS


#-----------------------------------------
# IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------
# IMPORT DATAFRAME

df = pd.read_csv('data/df02.csv', sep=',', na_filter=False, index_col=0, 
                 parse_dates=['launched_at'])

# Checks
if (df.isnull().sum().sum() != 0):
    print('*** WARNING: Null values introduced with read_csv ***')
if (df.isna().sum().sum() != 0):
    print('*** WARNING: NA values introduced with read_csv ***')
if (df=='').sum().sum() != 0:
    print('*** WARNING: Empty string (\'\') values introduced with read_csv ***')

#-----------------------------------------
# EXPLORATORY DATA ANALYSIS

# ---- EXPLORE launch_state ----
fig=plt.figure(figsize=(8,4))
sns.set_style('whitegrid')
sns.countplot(x='launch_state', data=df, palette='viridis')

# Success rate is proving difficult to calculate due to dirty (especially
# duplicate data. TBD)

# =============================================================================
# It looks like a huge portion of projects ultimately fail to launch! We will
# look into this more later, but for now consider the following from 
# https://www.kickstarter.com/help/stats (as of 2018-11-16 10:40):
#  * Overall success rate: 36.53%
#      - Why is this different than the 53.2% success rate calculted above?
#  * "While 13% of projects finished having never received a single pledge 78%
#    of projects that raised more than 20% of their goal were successfully 
#    funded."
# =============================================================================

# ---- DUMMY VARIABLE PAIRPLOT ----
sns.set_context("paper", rc={"axes.labelsize":20, 
                             "xtick.labelsize": 15, "ytick.labelsize": 15})
sns.set_style('whitegrid')
plt.figure(figsize=(14,5))
sns.pairplot(data=df.drop(df.columns[10:], axis=1).drop(
        columns=['id','launched_at','category','country']), 
             diag_kind='kde', hue='launch_state', palette='viridis')
# Observation: little insight to be gained here

# ---- FUNDING_DAYS VS GOAL ----
sns.set_style('whitegrid')
sns.set(font_scale=3)
sns.lmplot("goal", "funding_days", data=df, hue='launch_state', size=12,
           fit_reg=False, scatter_kws={'alpha':0.1, 's':500}, palette='viridis')

sns.set_style('whitegrid')
sns.set(font_scale=3)
sns.lmplot("goal", "funding_days", data=df, hue='launch_state', size=12,palette='viridis',
           fit_reg=False, scatter_kws={'alpha':0.5, 's':500}).set(xlim=(0,250000))

# =============================================================================
# Observations: 
#  * Successful launches seem loosely clustered around funding_days = [0,60]
#    and goal < $100k
# =============================================================================

# ---- BACKERS_COUNT VS GOAL ----
sns.set_style('whitegrid')
sns.set(font_scale=3)
sns.lmplot("goal", "backers_count", data=df, hue='launch_state', size=12,
           fit_reg=False, scatter_kws={'alpha':0.1, 's':500}, palette='viridis')

sns.set_style('whitegrid')
sns.set(font_scale=3)
sns.lmplot("goal", "backers_count", data=df, hue='launch_state', size=12,
           fit_reg=False, scatter_kws={'alpha':0.1, 's':500}, palette='viridis').set(
    xlim=(0,250000), ylim=(0,15000))

sns.set_style('whitegrid')
sns.set(font_scale=3)
sns.lmplot("goal", "backers_count", data=df, hue='launch_state', size=12,
           fit_reg=False, scatter_kws={'alpha':0.1, 's':500}, palette='viridis').set(
    xlim=(0,100000), ylim=(0,2000))

# =============================================================================
# Observations: backers_count is not a good predictor; it's obvious that
# with a high number of backers the project is more likely to succeed. More
# important for this project, though, is the fact that it cannot be used a priori.
# =============================================================================

# ---- PLEDGED_RATIO vs GOAL ----
sns.set_style('whitegrid')
sns.set(font_scale=3)
sns.lmplot("goal", "pledged_ratio", data=df, hue='launch_state', size=12,
           fit_reg=False, scatter_kws={'alpha':0.1, 's':500}, palette='viridis')

sns.set_style('whitegrid')
sns.set(font_scale=3)
sns.lmplot("goal", "pledged_ratio", data=df, hue='launch_state', size=12,
           fit_reg=False, scatter_kws={'alpha':0.1, 's':500}, 
           palette='viridis').set(xlim=(0,250000), ylim=(0,2))

# =============================================================================
# Observations: As expected, pledged_ratio < 1 means failure and all
# pledged_ratio >= 1 means success
# =============================================================================

# ---- STAFF_PICK vs GOAL ----
sns.set_style('whitegrid')
sns.set(font_scale=3)
sns.lmplot("goal", "staff_pick", data=df, hue='launch_state', size=12,
           fit_reg=False, scatter_kws={'alpha':0.1, 's':500}, palette='viridis')

sns.set_style('whitegrid')
sns.set(font_scale=3)
sns.lmplot("goal", "staff_pick", data=df, hue='launch_state', size=12,
           fit_reg=False, scatter_kws={'alpha':0.1, 's':500},
           palette='viridis').set(xlim=(0,250000))

# Observations: Staff_pick seems to be a decent indicator in launch_state

# ---- EXPLORE STAFF_PICK AND LAUNCH_STATE
df_staff_picks = df[['launch_state','staff_pick']].groupby(
        ['staff_pick'], as_index=False).count()
df_staff_picks.columns = ['staff_pick','freq']
df_staff_picks['ratio'] = df[['launch_state','staff_pick']].groupby(
        ['staff_pick'], as_index=False).mean()['launch_state']

plt.figure()
sns.set_style('whitegrid')
ax = sns.barplot(data = df[['launch_state','staff_pick']].groupby(
        ['staff_pick'], as_index=False).mean(), x='staff_pick', y='launch_state')
ax.set(xlabel='staff_pick', ylabel='Average launch_state')

# =============================================================================
# Observations: 
#   * 13.5% of projects are chosen as staff picks
#   * staff_pick seems to correlate with launch_state:
#     - 52.3% of projects not chosen as staff picks succeed
#     - 88.9% of projects chosen as staff picks succeed 
#
# From https://www.kickstarter.com/blog/how-to-get-featured-on-kickstarter,
# it appears as if projects are featured when they catch the eye of the
# Kickstarter staff via creativity, a nice and visually appealing site, etc. 
# ie, they are NOT just picked due to them being funded well.
# =============================================================================

# =============================================================================
# So what have we learned so far?
# * goal - use it
# * backers_count - do not use it (it's not known beforehand)
# * pledged_ratio - do not use it (it's just an indicator of success)
# * funding_days - use it
# * staff_pick - use it
# =============================================================================

# ---- VISUALIZE CATEGORIES ----
df_categories = df[['launch_state','category']].groupby(
        ["category"]).describe().reset_index()
df_categories.sort_values(by=[('launch_state','mean')], ascending=False)
print(df_categories)

# Frequency plot
sns.set_style('whitegrid')
sns.factorplot(x='category', data=df, kind='count', size=10)

# Success ratio plot
plt.figure()
sns.set_style('whitegrid')
sns.barplot(x='category',y='launch_state',data=df)

# Observations: clearly, some categories are more successful than others.

# Heatmap
# Note that I fill in empty cells with 0.5 so as not to bias towards
# 0 (failure) and 1 (success)
plt.figure()
sns.set_style('whitegrid')
ax = sns.heatmap(
        df.pivot_table(values='launch_state', columns='category', 
                       index='funding_days', fill_value=0.5), xticklabels=True)
ax.invert_yaxis()
plt.title('Heatmap, launch_state - category vs funding_days')

# Clustermap
sns.set_style('whitegrid')
sns.clustermap(df.pivot_table(values='launch_state', columns='category', 
                              index='funding_days', fill_value=0.5),
                xticklabels=True)
plt.title('Clustermap, launch_state - category vs funding_days')

