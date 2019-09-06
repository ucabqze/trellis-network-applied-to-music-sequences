import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import time
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
nout = 29
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

data_raw,label,sf_pos = get_data(batch_size, sessions, seq_len)

train_data_raw = data_raw[0:15000]
val_data_raw = data_raw[15000:16000]
test_data_raw = data_raw[16000:16660]

train_label = label[0:15000]
val_label = label[15000:16000]
test_label = label[16000:16660]

train_sf = sf_pos[0:15000]
val_sf = sf_pos[15000:16000]
test_sf = sf_pos[16000:16660]

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

# produce sessions_feature_weights    
sessions_weights = get_session_matrix(sessions) 

###############################################################################
# Build the model
###############################################################################

class LSTM(nn.Module):
    def __init__(self, emsize, nhid, ntoken, nout,nlayers, batch_size, track_weight,sessions_weights):
        super(LSTM, self).__init__()
        self.nhid = nhid
        self.batch_size = batch_size
        self.nlayers = nlayers
        
        self.hidden=self.init_hidden(batch_size)
        
        self.encoder = nn.Embedding.from_pretrained(track_weight, freeze=True)
        self.sessions_encoder = nn.Embedding.from_pretrained(sessions_weights, freeze=True)
        self.lstm=nn.LSTM(emsize+sessions_weights.shape[1],nhid,nlayers,batch_first=True)
        self.decoder=nn.Linear(nhid,nout+1)
        
        self.init_weights()
        
    def init_hidden(self,batch_size_new):
        return (Variable(torch.zeros(self.nlayers,batch_size_new,self.nhid)),Variable(torch.zeros(self.nlayers,batch_size_new,self.nhid)))
    def init_weights(self):
        initrange = 0.1
        #self.hidden = (Variable(torch.zeros(self.nlayers,self.batch_size,self.nhid)),Variable(torch.zeros(self.nlayers,self.batch_size,self.nhid)))
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, input_, sf):
        self.batch_size = input_.shape[0] # some short rows are removed
        session_feature = self.sessions_encoder(sf)
        embeds_music = self.encoder(input_)
        embeds_music = torch.cat([embeds_music,session_feature],dim = 2)
        #x=embeds_music.view(self.batch_size,len(input_),-1)
        lstm_out,self.hidden=self.lstm(embeds_music,self.hidden)
        output=self.decoder(lstm_out[:,:,:]) ### batch,sentence length, embedding_dim
        return output       
 
model = LSTM(29,500,len(track_dic),29,3,12,track_weight,sessions_weights)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
lr = 0.2
optimizer = torch.optim.SGD(model.parameters(),lr=lr,weight_decay=0)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, lr_decay=0, weight_decay=0, initial_accumulator_value=0) 
###############################################################################
# Training code
###############################################################################

def get_batch_train(source,label, session_feature, i, seq_len, evaluation=False):
    """`source` has dimension (L, N)"""
    seq_len = min(seq_len, source.size(0) - 1 - i)
    data = source[i:int(i + seq_len/2)-1] 
    if evaluation:
        data.requires_grad = False
    target = source[i + 1:int(i + seq_len/2)]  # CAUTION: This is un-flattened!
    label = label[i:int(i + seq_len/2)-1] 
    sf = session_feature[i:int(i + seq_len/2)-1] 
    return data, target, label, sf


def train(epoch,data_source,label_, session_feature):
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(track_dic)
    
    for batch, i in enumerate(range(0, data_source.size(0) - 1, seq_len)):

        data, targets, label, sf = get_batch_train(data_source, label_,session_feature, i, seq_len)
        data = data.t() 
        sf = sf.t()
        
    
        #if data.shape[0]: # to prevent length filter in get_batch_train romove all batches                       
        optimizer.zero_grad()
        model.hidden=model.init_hidden(data.shape[0]) ### This is important, need to be init everytime
        output = model(data,sf)
        output = output.transpose(0, 1)
        targets = targets.contiguous().view(-1)
        label = label.contiguous().view(-1)
        final_decoded = output.contiguous().view(-1, nout+1)
        
        # remove padding rows 
#        mask_targets = targets!=0
#        targets = targets[targets!=0]
#        label = label[label!=-1]
#        loc = torch.ByteTensor(mask_targets) #<IndexBackward>  <ViewBackward>
#        final_decoded = final_decoded[loc]
# 
#        mask_decoded = mask_targets.unsqueeze(1).repeat(1, final_decoded.shape[1])
#        final_decoded = final_decoded*mask_decoded.float()
      

        if final_decoded.shape[0]:#
            
            final_targets = torch.cat([track_weight[targets],label.unsqueeze(1).float()],dim = 1)
        
            loss = criterion(final_decoded, final_targets)
            loss.backward(retain_graph=True) 
            optimizer.step()

            
            total_loss += loss.data
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} |'
                  'raw_loss {:5.2f} | ppl {:8.2f}'.format(
                   epoch, batch, len(data_source) // seq_len, lr,elapsed * 1000 / log_interval,
                   cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        

    output = None
    targets = None
    final_decoded = None
    
    return loss


        
def evaluate(data_source, label_, session_feature):
    model.eval() 
    with torch.no_grad(): 
        total_loss = 0
        ntokens = len(track_dic)
        batch_size = data_source.size(1)

        processed_data_size = 0
        for i in range(0, data_source.size(0) - 1, seq_len):

            data, targets, label, sf = get_batch_train(data_source, label_,session_feature, i, seq_len)
            data = data.t() 
            sf = sf.t()
        
            model.hidden=model.init_hidden(data.shape[0])
            output = model(data,sf)
            output = output.transpose(0, 1)
            targets = targets.contiguous().view(-1)
            label = label.contiguous().view(-1)
            final_decoded = output.contiguous().view(-1, nout+1)
            
            # remove padding rows 
#            mask_targets = targets!=0
#            targets = targets[targets!=0]
#            label= label[label!=-1]
#            loc = torch.ByteTensor(mask_targets) #<IndexBackward>  <ViewBackward>
#            final_decoded = final_decoded[loc]
            
            if final_decoded.shape[0]:#
                final_targets = torch.cat([track_weight[targets],label.unsqueeze(1).float()],dim = 1)
                loss = criterion(final_decoded, final_targets) ######
                loss = loss.data
    
                total_loss += data.size(1)* loss
                processed_data_size += data.size(1)

        output = None
        targets = None
        final_decoded = None

        return total_loss.item() / processed_data_size


def inference(epoch, epoch_start_time, train_loss = 'not avaliable'):
    val_loss = evaluate(val_data,val_label,val_sf)
    test_loss = evaluate(test_data,test_label,test_sf)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                     val_loss, math.exp(val_loss)))
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                    test_loss, math.exp(test_loss)))
    print('-' * 89)
    return val_loss, test_loss

best_val_loss = None
all_val_losses = []
all_test_losses = []
epochs = 50

try:
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        train_loss = train(epoch, train_data, train_label, train_sf)
        val_loss, test_loss = inference(epoch, epoch_start_time, train_loss)
        
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
                with open('music_lstm_4.pt', 'wb') as f:
                    torch.save(model, f)
                    # model.save_weights('weights/pretrained.pkl')
                    print('Saving model (new best validation) in ' + 'music_lstm_4.pt')
                best_val_loss = val_loss


        all_val_losses.append(val_loss)
        all_test_losses.append(test_loss)
        
except KeyboardInterrupt:
    print('-' * 89)
    print('KeyboardInterrupt!!!!!!!!!!!!!!!!!!!!')


# Load the best saved model
with open('music_lstm_4.pt', 'rb') as f:
    model = torch.load(f)
    #model.save_weights('pretrained_music.pkl')


# Run on test data
test_loss = evaluate(test_data,test_label, test_sf)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)