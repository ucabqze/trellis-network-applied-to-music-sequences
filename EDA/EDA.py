import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import torch
import os
from sklearn import preprocessing
from sklearn.externals import joblib

###############################################################################
# import data 
###############################################################################

data_path = '/Users/cengqiqi/Desktop/project/data/'

track_features = pd.read_csv(data_path + 'tf_mini.csv')
sessions =pd.read_csv(data_path + 'log_mini.csv')

###############################################################################
# contact sessions and tracks
###############################################################################

sessions.rename(columns={'track_id_clean':'track_id'}, inplace = True)
sessions_new = pd.merge(left=sessions,right=track_features,how='left',on = 'track_id')

###############################################################################
# track analysis
###############################################################################

# the distribution of release year
print('-' * 89)
print('Tracks Featrues: the release year distribution')

plt.figure(figsize=(12,6))
sns.set(color_codes=True)
track_features['release_year'].plot.hist(grid=True, bins=50, rwidth=0.9, color='#607c8e')
plt.title('Release Year Analysis',fontsize = 20)
plt.xlabel('Years', fontsize = 18)
plt.ylabel('Frequency', fontsize = 18)
plt.grid(axis='y', alpha=0.75)

plt.savefig(os.path.join('figures/tracks_features', 'tf_1_release_year_distributions.png'), format='png', dpi=300)

# the distribution of duration
print('-' * 89)
print('Tracks Featrues: the duration distribution')

plt.figure(figsize=(12,6))
sns.set(color_codes=True)
track_features['duration'].plot.hist(grid=True, bins=100, rwidth=0.9, color='#607c8e')
plt.title('duration Analysis',fontsize = 20)
plt.xlabel('seconds', fontsize = 18)
plt.ylabel('Frequency', fontsize = 18)
plt.grid(axis='y', alpha=0.75)

plt.savefig(os.path.join('figures/tracks_features', 'tf_2_duration_distributions.png'), format='png', dpi=300)

print('*'*70)
num_of_long_songs = track_features[track_features['duration']>500].count()['duration']
print('The songs which are more than 500 seconds: ', num_of_long_songs)
print('*'*70)

###############################################################################
# session analysis
###############################################################################

# the distribution of sessions length

# the relationship between skip behaviour and context_switch

# the relationship between hist_user_behavior_n_seekfwd, hist_user_behavior_n_seekback and skip

# the premium analysis

# the distribution of relase year within a session (std)

# the popuarity of us_popularity_estimate within a session (std) 

# the music style

# the relationship between hour_of_day and music property

# 