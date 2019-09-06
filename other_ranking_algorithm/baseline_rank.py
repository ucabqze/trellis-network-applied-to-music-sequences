import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import data
import sys
from utils import *
#from setproctitle import setproctitle
from tqdm import tqdm

sys.path.append("../")
from data_processing_edaadded import get_data,get_track_dic

__author__ = 'Qiqi Zeng'

###############################################################################
# define parameters
###############################################################################
batch_size = 12
emsize = 29
seq_len = 20 #### should be same with the pad_sequence maxlen in data_processing.py 's def get_data(batch_size, sessions)
log_interval =100
saved_model = 'music_rnn.pt'
x = 8

###############################################################################
# Load data
###############################################################################

# load origin data and label
data_path = '/Users/cengqiqi/Desktop/project/data/'

track_features = pd.read_csv(data_path + 'tf_mini.csv')
sessions =pd.read_csv(data_path + 'log_mini.csv')
 
# get track dict
track_dic = get_track_dic(track_features)
ntracks = len(track_dic)

# get train data, validation data and test data
# Note: you should build track dictionary using function get_track_dic before get data  
sessions_train = sessions[0:150000]
sessions_val = sessions[150000:160000]
sessions_test = sessions[160000:167880]

train_data_raw, train_label = get_data(batch_size,sessions_train,seq_len)
val_data_raw, val_label = get_data(batch_size,sessions_val,seq_len)
test_data_raw, test_label = get_data(batch_size,sessions_test,seq_len)


###############################################################################
# prepare data for track embedding in pytorch
###############################################################################

# for word embedding, find the track position in track_dic
track_dic_index = {}
tracks = list(track_dic.keys())
for i in range(len(tracks)):
    track_dic_index[tracks[i]] = i


# convert data: represented by index
# padding is -1, now Let's make it zero in this step
def convert_to_index(data_raw, track_dic):
    data = []
    for session in data_raw:
        b=[]
        for track in session:
            try:
                b.append(track_dic_index[int(track)])
            except ValueError:
                b
        data.append(b)
    data = torch.Tensor(np.array(data)).long()

    return data

train_data = convert_to_index(train_data_raw, track_dic)#
val_data = convert_to_index(val_data_raw, track_dic)#[0:150]
test_data = convert_to_index(test_data_raw, track_dic)#[0:150]
train_label = train_label#[0:500]
val_label = val_label#[0:150]
test_label = test_label#[0:150]

# produce track_dict_weights
track_weight=np.zeros((len(list(track_dic.keys())), emsize))
for i, track in enumerate(list(track_dic.keys())):
    try:
        track_weight[i]=track_dic[track]
    except KeyError:
        track_weight[i] = np.random.normal(scale=0.6, size=(emsize, ))
track_weight=torch.FloatTensor(track_weight)       


###############################################################################
# Get Rank
###############################################################################     

def get_batch_past(source,label, i, seq_len, evaluation=False):
    """get the first 10 tracks in a session"""
        
    from tensorflow.contrib.keras import preprocessing
    # add mask to ignore skipped track and padding tracks
    skip_mask = ((label>=0) *(label<2)).long()
    source = source*skip_mask

    # reshape
    seq_len = min(seq_len, source.size(0) - 1 - i)
    data = source[i:int(i + seq_len/2)] 
    
    # move 0 to left
    data = data.t()
    sessions_list = []
    #pack_len = []
    for session in data: # loop of batch size
        session_remove0 = session[session!=0]
        sessions_list.append(session_remove0)
#        # length filter
#        if len(session_remove0)>=5:
#            sessions_list.append(session_remove0)
        #pack_len.append(len(session_remove0)-1) # length 10 to 9 for next word
    data = preprocessing.sequence.pad_sequences(sessions_list,len(session),padding='pre',truncating='pre')
    data = torch.Tensor(data).long().t()
    if evaluation:
        data.requires_grad = False
    target = torch.cat([data[1:],torch.zeros(data.shape[1]).long().unsqueeze(0)])

    return data, target

def get_batch_future(data_source, label, i, seq_len, evaluation=False):
    """get the last 10 tracks in a session and their skip info"""
    seq_len = min(seq_len, data_source.size(0) - 1 - i)
    data = data_source[i + int(seq_len/2):i + seq_len]
    if evaluation:
        data.requires_grad = False
    target = label[i + int(seq_len/2):i + seq_len]
    return data, target
 
    

def dcg_score(skip_info):
    # not skipped tracks get higher score
    skip_info = 3-skip_info
    
    # count dcg
    gain = 2 ** skip_info-1
    position_disccount = 1/np.log2(np.arange(len(skip_info))+2)
    dcg = np.sum(gain*position_disccount)
    
    return dcg

track_features = pd.read_csv(data_path + 'tf_mini.csv')    
# get rank vector
def rank(data_source, label):
    model.eval() 
    
    with torch.no_grad():
        ndcg_acc = 0
        ndcg_count = 0
        ntokens = len(track_dic)
        batch_size = data_source.size(1)

        for i in range(0, data_source.size(0) - 1, seq_len):

            data, targets = get_batch_past(data_source, label, i, seq_len, evaluation=True)
            data = data.t()
            targets = targets.t()
           
            tracks_future, targets_future = get_batch_future(data_source, label, i, seq_len, evaluation=True)
            tracks_future = tracks_future.t()
            targets_future = targets_future.t()
     
            for j in range(batch_size):
                track_f = tracks_future[j]
                # remove padding elements
                track_f = track_f[track_f!=0]
                score = []
                for ii in tracks_future[j]:
                    if ii!=0:
                        score.append(track_features['us_popularity_estimate'][int(ii)])
                
                # get data frame without padding element
                df_future = pd.DataFrame({'track':np.array(track_f),'score':np.array(score),'skip_info':np.array(targets_future[j][0:len(track_f)])})
                # remove padding elements
                df_future = df_future.loc[df_future['track']!=0]
                
                # sort by popularity
                df_future = df_future.sort_values(by = 'score',ascending=False) #0.8090114383851303
                #sort to the worst case
                #df_future = df_future.sort_values(by = 'skip_info',ascending=False)  #0.6693316113866979
                
                # NDCG
                actual = dcg_score(df_future['skip_info'])
                best = dcg_score(df_future['skip_info'].sort_values(ascending=True))
                
                if best: #best might be 0, while skip_info is 3,3,3,....
                    ndcg = actual/best
                    ndcg_acc = ndcg_acc + ndcg
                else: # avoid nan
                    ndcg_acc = ndcg_acc + 1
                ndcg_count = ndcg_count+1
         
               
        ndcg_avg = ndcg_acc/ndcg_count
    
    return ndcg_avg
                
 
ndcg_avg= rank(test_data, test_label)
print('The NDCG Score for whole test data: ', ndcg_avg)
