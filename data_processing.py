import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.externals import joblib

__author__ = 'Qiqi Zeng'

###############################################################################
# import data 
###############################################################################

#data_path = '/Users/cengqiqi/Desktop/project/data/'
#
#track_features = pd.read_csv(data_path + 'tf_mini.csv')
#sessions =pd.read_csv(data_path + 'log_mini.csv')


###############################################################################
# feature processing(build track dictionary)
###############################################################################

def feature_normalisation(feature_group, track_features_orign, track_features_new):
    """ This function standardized features to have mean = 0 with std = 1 """
    for name in feature_group:
        tf = track_features_orign[name]
        mean = np.mean(tf)
        std = np.std(tf)
        track_features_new[name] = (tf-mean)/std
    return track_features_new

def feature_one_hot(feature_group, track_features_orign, track_features_new):
    """ This function encode labels with value between 0 and n_classes-1"""
    le = preprocessing.LabelEncoder()
    for name in feature_group:
        le.fit(track_features_orign[name])
        track_features_new[name] = le.transform(track_features_orign[name])
    return track_features_new

def feature_all(track_features):
    """ This function """
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
    track_features = feature_normalisation(feature_group1,track_features,track_features)
    track_features = feature_normalisation(feature_group2,track_features,track_features)
    track_features = feature_one_hot(feature_group3,track_features,track_features)
    
    #print(track_features.iloc[0])
    return track_features

def change_feature_weight(track_features):
    return track_features

###############################################################################
# process track id
###############################################################################

def process_id_t(track_features):
    # set track_feature_id_index
    le_track_id = preprocessing.LabelEncoder()
    track_features['track_id'] = le_track_id.fit_transform(track_features['track_id'])
    joblib.dump(le_track_id, 'le_track_id.pkl')
    return track_features
    
def process_id_s(sessions):
    
    le_track_id = joblib.load('le_track_id.pkl')
    sessions['track_id_clean'] = le_track_id.transform(sessions['track_id_clean'])
    
    le_session = preprocessing.LabelEncoder()
    sessions['session_id'] = le_session.fit_transform(sessions['session_id'])
    
    return sessions
        

###############################################################################
# session id segmentation and prepare train data, validation data and test data
###############################################################################

def batchify(data, bsz, cuda=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1,1).contiguous().squeeze().t()
    
    if cuda:
        data = data.cuda()
    return data

def sessions_segmentation(sessions):    
    
#    sessions_list = []
#    label_list = []
#    
#    i = 0
#    while i < len(sessions):
#        from_ = i
#        to_ = i+sessions.iloc[i].session_length
#        
#        tracks_in_a_session = sessions.iloc[from_:to_].track_id_clean
#        skips1_in_a_session = sessions.iloc[from_:to_].skip_1
#        skips2_in_a_session = sessions.iloc[from_:to_].skip_2
#        skips3_in_a_session = sessions.iloc[from_:to_].skip_3
#        label_in_a_session = (torch.Tensor(np.array(skips1_in_a_session))+
#                              torch.Tensor(np.array(skips2_in_a_session))+
#                              torch.Tensor(np.array(skips3_in_a_session)))
#        sessions_list.append(torch.Tensor(np.array(tracks_in_a_session)).long())
#        label_list.append(label_in_a_session.long())
#        i = to_
     
    gp_track = sessions['track_id_clean'].groupby(sessions['session_id'])
    data_list = [torch.Tensor(np.array(gp_track.get_group(x))).long() for x in gp_track.groups]
    
    gp_skip = sessions[['skip_1','skip_2','skip_3']].groupby(sessions['session_id'])
    label_list = [torch.Tensor(np.array(gp_skip.get_group(x))).sum(1).long() for x in gp_skip.groups]

    return data_list,label_list
# the max len of tracks_in_a_session in 'training_set1/log_0_20180715_000000000000.csv' is 20
    

def get_track_dic(track_features):
    # process track features
    track_features = feature_all(track_features)
    track_features = change_feature_weight(track_features) 
    track_features = process_id_t(track_features)
    
    # build track dict
    track_dic = {-1:torch.Tensor(track_features.shape[1]-1)} ### making padding element = 0 after converting to index for embedding
    # track_features.shape[1]-1 = 29 (track_features.shape[1] include track_id)
    for i in range(track_features.shape[0]):
        track_id = int(track_features.iloc[i]['track_id'])
        track_vec = torch.from_numpy(track_features.iloc[i][1:].values) # exclude track id
        track_dic[track_id] = track_vec
    return track_dic
          
def get_data(batch_size, sessions, maxlen):
    
    """ Note: you should build track dictionary using function get_track_dic before get data"""
    # process id
    try:
        sessions = process_id_s(sessions)
    except FileNotFoundError:
        print('*'*95)
        print('you should  build track dictionary using function get_track_dic before get data')
        print('*'*95)
        return None, None
    else:
#        input_raw = torch.Tensor(np.array(sessions['track_id_clean']))  
#        input_ = batchify(input_raw, batch_size, cuda = False).long()
#      
#        label_raw = torch.Tensor(np.array([sessions['skip_1'],sessions['skip_2'],sessions['skip_3']]))
#        label_raw = label_raw.sum(0)
#        label = batchify(label_raw, batch_size, cuda = False).long()
#        
#        return input_, label

        sessions_list,label_list = sessions_segmentation(sessions)
        
        # convert sequences to same length
        #args.seqlen = 20
        from tensorflow.contrib.keras import preprocessing
        sessions_list = preprocessing.sequence.pad_sequences(sessions_list,maxlen, padding='post',truncating='post',value=-1)
        sessions_list = torch.Tensor(sessions_list).long()
        label_list = preprocessing.sequence.pad_sequences(label_list,maxlen, padding='post',truncating='post',value=-1)
        label_list = torch.Tensor(label_list).long()
        #sessions_list = torch.nn.utils.rnn.pad_sequence(sessions_list, batch_first=True) 
        #label_list = torch.nn.utils.rnn.pad_sequence(label_list,batch_first=True) 
     
        # convert input shape
        input_ = batchify(sessions_list, batch_size, cuda = False).long()
#      
        label = batchify(label_list, batch_size, cuda = False).long()
        ### note: return scalar instead of float
        #return sessions_list,label_list
        return input_, label

    
