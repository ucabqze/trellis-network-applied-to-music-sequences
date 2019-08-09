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
from data_processing import get_data,get_track_dic

__author__ = 'Qiqi Zeng'

###############################################################################
# define parameters
###############################################################################
batch_size = 12
emsize = 29
seq_len = 20 #### should be same with the pad_sequence maxlen in data_processing.py 's def get_data(batch_size, sessions)
log_interval =100
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
sessions_train = sessions[0:6000]
sessions_val = sessions[6000:7500]
sessions_test = sessions[7500:8395]

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

train_data = convert_to_index(train_data_raw, track_dic)#[0:500]
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
# Build the model
###############################################################################

class RNN(nn.Module):
    def __init__(self, emsize, nhid, ntoken, nout,nlayers, batch_size, track_weight):
        super(RNN, self).__init__()
        self.nhid = nhid
        self.batch_size = batch_size
        self.nlayers = nlayers
        
        self.hidden=self.init_hidden()
        
        self.encoder = nn.Embedding.from_pretrained(track_weight, freeze=True)
        self.rnn=nn.RNN(
            input_size=emsize,
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
        
    def forward(self, input_):
        embeds_music = self.encoder(input_)
        #x=embeds_music.view(self.batch_size,len(input_),-1)
        rnn_out,self.hidden=self.rnn(embeds_music,self.hidden)
        output=self.decoder(rnn_out[:,:,:]) ### batch,sentence length, embedding_dim
        return output
        
 
model = RNN(29,100,len(track_dic),29,3,12,track_weight)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,weight_decay=0)
 
###############################################################################
# Training code
###############################################################################


def get_batch(data_source, label, i, seq_len, evaluation=False):
    """`data_source` has dimension (L, N)"""
    seq_len = min(seq_len, data_source.size(0) - 1 - i)
    data = data_source[i:i + seq_len]
    if evaluation:
        data.requires_grad = False
    target = label[i:i + seq_len] 
    return data, target

       
def train(epoch):
    model.train()
    total_loss = 0
    ntokens = len(track_dic)
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):

        data, targets = get_batch(train_data, train_label, i, seq_len)
        data = data.t()

        optimizer.zero_grad()
        model.hidden=model.init_hidden() ### This is important, need to be init everytime
        output = model(data)
        output = output.transpose(0, 1)
        targets = targets.contiguous().view(-1)
        final_decoded = output.contiguous().view(-1, ntokens)
        
        loss = criterion(final_decoded, targets)

        loss.backward(retain_graph=True) 
        optimizer.step()
        
        
        total_loss += loss.data
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'raw_loss {:5.2f} | ppl {:8.2f}'.format(
                   epoch, batch, len(train_data) // seq_len, lr,
                   cur_loss, math.exp(cur_loss)))
            total_loss = 0
            

    output = None
    targets = None
    final_decoded = None
    
    return loss


        
def evaluate(data_source, label):
    model.eval() 
    with torch.no_grad(): 
        total_loss = 0
        ntokens = len(track_dic)
        batch_size = data_source.size(1)

        processed_data_size = 0
        for i in range(0, data_source.size(0) - 1, seq_len):

            data, targets = get_batch(data_source, label, i, seq_len, evaluation=True)
            data = data.t()
            
            model.hidden=model.init_hidden()
            output = model(data)
            output = output.transpose(0, 1)
            targets = targets.contiguous().view(-1)
            final_decoded = output.contiguous().view(-1, ntokens)
            
            loss = criterion(final_decoded, targets) ######
            loss = loss.data

            total_loss += data.size(1)* loss
            processed_data_size += data.size(1)

        output = None
        targets = None
        final_decoded = None

        return total_loss.item() / processed_data_size


def inference(epoch, train_loss = 'not avaliable'):
    val_loss = evaluate(val_data,val_label)
    test_loss = evaluate(test_data,test_label)
    print('-' * 89)
    print('| end of epoch {:3d} | train loss {:5.2f} | '
          'test ppl {:8.2f}'.format(epoch, train_loss, math.exp(test_loss)))
    print('| end of epoch {:3d} | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, val_loss, math.exp(val_loss)))
    print('| end of epoch {:3d} | test loss {:5.2f} | '
          'test ppl {:8.2f}'.format(epoch, test_loss, math.exp(test_loss)))
    print('-' * 89)
    return val_loss, test_loss


lr = 0.01
best_val_loss = None
all_val_losses = []
all_test_losses = []
epochs = 500

try:
    for epoch in range(1, epochs + 1):
        
        train_loss = train(epoch)
        val_loss, test_loss = inference(epoch, train_loss)
        
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss

#        if len(all_val_losses) > args.anneal and val_loss > min(all_val_losses[:-args.anneal]):
#            print("\n" + "*" * 89)
#            if args.asgd and 't0' not in optimizer.param_groups[0]:
#                print('Switching to ASGD')
#                optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
#                args.save = args.name + "_asgd.pt"
#            elif lr > 0.02:
#                print('Annealing learning rate')
#                lr /= 4.0
#                optimizer.param_groups[0]['lr'] = lr
#            print("*" * 89 + "\n")

        all_val_losses.append(val_loss)
        all_test_losses.append(test_loss)
        
except KeyboardInterrupt:
    print('-' * 89)
    print('KeyboardInterrupt!!!!!!!!!!!!!!!!!!!!')


# Run on test data
test_loss = evaluate(test_data,test_label)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)