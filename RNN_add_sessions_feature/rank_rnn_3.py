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
from data_processing_3 import get_data,get_track_dic,get_session_matrix

__author__ = 'Qiqi Zeng'

###############################################################################
# define parameters
###############################################################################
batch_size = 12
emsize = 29
seq_len = 20 #### should be same with the pad_sequence maxlen in data_processing.py 's def get_data(batch_size, sessions)
log_interval =100
saved_model = 'music_rnn.pt'

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

data_raw,label,sf = get_data(batch_size, sessions, seq_len)


###############################################################################
# prepare data for track embedding in pytorch
###############################################################################

# for word embedding, find the track position in track_dic
track_dic_index = {}
tracks = list(track_dic.keys())
for i in range(len(tracks)):
    track_dic_index[tracks[i]] = i


# convert data: represented by index
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

data = convert_to_index(data_raw, track_dic)
label = label

# produce track_dict_weights
track_weight=np.zeros((len(list(track_dic.keys())), emsize))
for i, track in enumerate(list(track_dic.keys())):
    try:
        track_weight[i]=track_dic[track]
    except KeyError:
        track_weight[i] = np.random.normal(scale=0.6, size=(emsize, ))
track_weight=torch.FloatTensor(track_weight)       

# produce sessions_feature_weights    
sessions_weights = get_session_matrix(sessions) 

###############################################################################
# build model
###############################################################################

class RNN(nn.Module):
    def __init__(self, emsize, nhid, ntoken, nout,nlayers, batch_size, track_weight, sessions_weights):
        super(RNN, self).__init__()
        self.nhid = nhid
        self.batch_size = batch_size
        self.nlayers = nlayers
        
        self.hidden=self.init_hidden()
        
        self.encoder = nn.Embedding.from_pretrained(track_weight, freeze=True)
        self.sessions_encoder = nn.Embedding.from_pretrained(sessions_weights, freeze=True)
        self.rnn=nn.RNN(
            input_size=emsize+sessions_weights.shape[1],
            hidden_size=nhid,
            num_layers=nlayers,
            batch_first=True)
        self.decoder=nn.Linear(nhid,ntoken)
        
        self.init_weights()
        
    def init_hidden(self):
        return None
    def init_weights(self):
        initrange = 0.1
        #self.hidden = (Variable(torch.zeros(self.nlayers,self.batch_size,self.nhid)),Variable(torch.zeros(self.nlayers,self.batch_size,self.nhid)))
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, input_,sf):
        embeds_music = self.encoder(input_)
        sessions_feature = self.sessions_encoder(sf)
        embeds_music = torch.cat([embeds_music,sessions_feature],dim = 2)
        #x=embeds_music.view(self.batch_size,len(input_),-1)
        rnn_out,self.hidden=self.rnn(embeds_music,self.hidden)
        output=self.decoder(rnn_out[:,:,:]) ### batch,sentence length, embedding_dim
        return output
        

###############################################################################
# Load the trained model
###############################################################################
with open(saved_model, 'rb') as f:
    model = torch.load(f)
     
###############################################################################
# Get Rank
###############################################################################     

def get_batch_past(source,label, sessions_feature, i, seq_len, evaluation=False):
    """`source` has dimension (L, N)"""
    seq_len = min(seq_len, source.size(0) - 1 - i)
    data = source[i:int(i + seq_len/2)-1] #first 10 but -1 for target
    if evaluation:
        data.requires_grad = False
    target = source[i + 1:int(i + seq_len/2)]  # CAUTION: This is un-flattened!
    sf = sessions_feature[i:int(i + seq_len/2)-1] 
    return data, target, sf



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
    
# get rank vector
def rank(data_source, label, sessions_feature):
    model.eval() 
    
    with torch.no_grad():
        ndcg_acc = 0
        ndcg_count = 0
        total_loss = 0
        ntokens = len(track_dic)
        batch_size = data_source.size(1)

        for i in range(0, data_source.size(0) - 1, seq_len):

            data, targets, sf = get_batch_past(data_source, label, sessions_feature,i, seq_len, evaluation=True)
            data = data.t()
            sf = sf.t()
            
            tracks_future, targets_future = get_batch_future(data_source, label, i, seq_len, evaluation=True)
            tracks_future = tracks_future.t()
            targets_future = targets_future.t()
            
            #music_rnn; music_lstm
            rank_vec = model(data,sf)[:,-1,:] # batch, ntokens [12, 1, 50704]
            #rank_vec = torch.nn.functional.softmax(rank_vec)
            # advoid for loop
            for j in range(batch_size):
                track_f = tracks_future[j]
                # remove padding elements
                #track_f = track_f[track_f!=0]
                score = rank_vec[j][track_f]
                # get data frame without padding element
                df_future = pd.DataFrame({'track':np.array(track_f),'score':np.array(score),'skip_info':np.array(targets_future[j][0:len(track_f)])})
                # remove padding elements
                df_future = df_future.loc[df_future['track']!=0]
                # sort tracks_future according to score
                df_future = df_future.sort_values (by = 'score',ascending=False) #0.8154440681444343 #0.8227163023038474
                #df_future = df_future.sample(frac=1) # 0.8115378563756852 #0.7787248338261271 
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
                
 
ndcg_avg = rank(data, label,sf)
print(ndcg_avg)

#
#        output = None
#        targets = None
#        final_decoded = None
#
#        return total_loss.item() / processed_data_size
