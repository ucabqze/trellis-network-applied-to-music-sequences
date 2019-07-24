import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.externals import joblib

__author__ = 'Qiqi Zeng'

###############################################################################
# import data 
###############################################################################

data_path = '/Users/cengqiqi/Desktop/project/data/'

track_features = pd.read_csv(data_path + 'tf_mini.csv')
sessions =pd.read_csv(data_path + 'log_mini.csv')

#print(track_features.columns)
#print(track_features.iloc[0])
#print(track_features.size)


###############################################################################
# define functions
###############################################################################


def feature_normalisation(feature_group, track_features_orign=track_features, 
                          track_features_new=track_features):
    """ This function standardized features to have mean = 0 with std = 1 """
    for name in feature_group:
        tf = track_features_orign[name]
        mean = np.mean(tf)
        std = np.std(tf)
        track_features_new[name] = (tf-mean)/std
    return track_features_new

def feature_one_hot(feature_group, track_features_orign=track_features, 
                    track_features_new = track_features):
    """ This function encode labels with value between 0 and n_classes-1"""
    le = preprocessing.LabelEncoder()
    for name in feature_group:
        le.fit(track_features_orign[name])
        track_features_new[name] = le.transform(track_features_orign[name])
    return track_features_new


def batchify(data, bsz, cuda=True):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


###############################################################################
# feature processing(build track dictionary)
###############################################################################

feature_group1 = ['us_popularity_estimate','acousticness','beat_strength',
                  'bounciness','danceability','dyn_range_mean','energy',
                  'flatness', 'instrumentalness','liveness','loudness',
                  'mechanism','organism','speechiness','tempo','valence']
feature_group2 = ['duration','release_year','key','time_signature']
feature_group3 = ['mode']
feature_group4 = ['acoustic_vector_0','acoustic_vector_1','acoustic_vector_2',
                  'acoustic_vector_3','acoustic_vector_4','acoustic_vector_5',
                  'acoustic_vector_6','acoustic_vector_7']


#track_features_new = []
track_features = feature_normalisation(feature_group1)
track_features = feature_normalisation(feature_group2)
track_features = feature_one_hot(feature_group3)

#print(track_features.iloc[0])

###############################################################################
# get feature vector (track_dic)
###############################################################################

# set track_feature_id_index
le_track_id = preprocessing.LabelEncoder()
track_features['track_id'] = le_track_id.fit_transform(track_features['track_id'])
#joblib.dump(le_track_id, 'le_track_id.pkl')
#le_track_id = joblib.load('le_track_id.pkl')
sessions['track_id_clean'] = le_track_id.transform(sessions['track_id_clean'])

le_session = preprocessing.LabelEncoder()
sessions['session_id'] = le_session.fit_transform(sessions['session_id'])
track_dic = {}
for i in range(track_features.shape[0]):
    track_id = int(track_features.iloc[i]['track_id'])
    track_vec = torch.from_numpy(track_features.iloc[i][1:].values) # exclude track id
    track_dic[track_id] = track_vec
        
def get_track_dic():
    return track_dic
       


###############################################################################
# session id segmentation and prepare train data, validation data and test data
###############################################################################

#input_seq = []
#
#sessions.loc[sessions['session_id'] == i,'track_id_clean']
#
#for i in sessions['session_id'].unique():
#    input_vec = torch.Tensor(np.array(sessions.loc[sessions['session_id'] == i,'track_id_clean']))

def get_data(batch_size):
    
    input_raw = torch.Tensor(np.array(sessions['track_id_clean']))  
    input = batchify(input_raw, batch_size, cuda = False)
    train_data = input[0:6000]
    val_data = input[6000:7500]
    test_data = input[7500:8395]
    
    
    label_raw = torch.Tensor(np.array([sessions['skip_1'],sessions['skip_2'],sessions['skip_3']]))
    label_raw = label_raw.sum(0)
    label = batchify(label_raw, batch_size, cuda = False)
    train_label = label[0:6000]
    val_label = label[6000:7500]
    test_label = label[7500:8395]
    
    return train_data,val_data,test_data,train_label,val_label,test_label

###############################################################################
# get data and label
###############################################################################


