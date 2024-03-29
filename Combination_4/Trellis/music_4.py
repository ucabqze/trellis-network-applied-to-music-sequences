#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import time
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import data
import sys
from utils import *
#from setproctitle import setproctitle
from tqdm import tqdm

sys.path.append("../")
from model_music_4 import TrellisNetModel
from data_processing_3 import get_data,get_track_dic,get_session_matrix


__author__ = 'Shaojie Bai and Qiqi Zeng'
# this is Adapted from the code of Shaojie Bai, Avaliable at https://github.com/locuslab/trellisnet


parser = argparse.ArgumentParser(description='PyTorch TrellisNet Language Model')
#parser.add_argument('--data', type=str, default='./data/penn',
#                    help='location of the data corpus')
parser.add_argument('--name', type=str, default='music_trellis_4',
                    help='name of the process')
parser.add_argument('--emsize', type=int, default=29,  ### the shape of a track in track_dic
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=600,  ### 500
                    help='number of hidden units per layer')
parser.add_argument('--nout', type=int, default=29,
                    help='number of output units')
parser.add_argument('--lr', type=float, default=2,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.225,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, 
                    help='upper epoch limit (default: 500)')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='batch size')

# For most of the time, you should change these two together
parser.add_argument('--nlevels', type=int, default=5, ### 4
                    help='levels of the network')
parser.add_argument('--horizon', type=int, default=5, ### 4
                    help='The effective history size')

parser.add_argument('--dropout', type=float, default=0.2,
                    help='output dropout (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='input dropout (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.1,
                    help='dropout applied to weights (0 = no dropout)')
parser.add_argument('--emb_dropout', type=float, default=0,
                    help='dropout applied to embedding layer (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.1,
                    help='dropout applied to hidden layers (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=0.1,
                    help='dropout applied to latent layer in MoS (0 = no dropout)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights (default: True)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--anneal', type=int, default=10,
                    help='learning rate annealing criteria (default: 10)')

parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')

parser.add_argument('--wnorm', action='store_true',
                    help='use weight normalization (default: False)')
parser.add_argument('--temporalwdrop', action='store_false',
                    help='only drop the temporal weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use')
parser.add_argument('--asgd', action='store_true',
                    help='use ASGD when learning plateaus (follows from Merity et al. 2017) (default: False)')
parser.add_argument('--repack', action='store_true',
                    help='use repackaging (default: False)')
parser.add_argument('--eval', action='store_true',
                    help='evaluation only mode')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on outputs (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on outputs (beta = 0 means no regularization)')
parser.add_argument('--aux', type=float, default=0.5,
                    help='use auxiliary loss (default: 0.05), -1 means no auxiliary loss used')
parser.add_argument('--aux_freq', type=float, default=3,
                    help='auxiliary loss frequency (default: 3)')
parser.add_argument('--seq_len', type=int, default=20, ### should be same with the pad_sequence maxlen in data_processing.py 's def get_data(batch_size, sessions)
                    help='total sequence length, including effective history (default: 110)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--when', nargs='+', type=int, default=[-1],
                    help='When to decay the learning rate')
parser.add_argument('--ksize', type=int, default=2,
                    help='conv kernel size (default: 2)')
parser.add_argument('--dilation', nargs='+', type=int, default=[1],
                    help='dilation rate (default: [1])')
parser.add_argument('--n_experts', type=int, default=0,
                    help='number of softmax experts (default: 0)')
parser.add_argument('--load', type=str, default='',
                    help='path to load the model')
parser.add_argument('--load_weight', type=str, default='',
                    help='path to load the model weights (please only use --load or --load_weight)')

args = parser.parse_args()
### by qiqi
args.cuda = False
args.wnorm = True

args.save = args.name + ".pt"

torch.set_default_tensor_type('torch.FloatTensor')


###############################################################################
# Load data
###############################################################################

data_path = '/Users/cengqiqi/Desktop/project/data/'

# load origin data and label
track_features = pd.read_csv(data_path + 'tf_mini.csv')
sessions =pd.read_csv(data_path + 'log_mini.csv')
 
# get track dict
track_dic = get_track_dic(track_features)
ntracks = len(track_dic)

# get train data, validation data and test data
# Note: you should build track dictionary using function get_track_dic before get data  

data_raw,label,sf_pos = get_data(args.batch_size, sessions, args.seq_len)

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

train_data = convert_to_index(train_data_raw, track_dic)#[0:1000]
val_data = convert_to_index(val_data_raw, track_dic)#[0:500]
test_data = convert_to_index(test_data_raw, track_dic)#[0:500]
train_label = train_label#[0:1000]
val_label = val_label#[0:500]
test_label = test_label#[0:500]

# produce track_dict_weights
track_weight=np.zeros((len(list(track_dic.keys())), args.emsize))
for i, track in enumerate(list(track_dic.keys())):
    try:
        track_weight[i]=track_dic[track]
    except KeyError:
        track_weight[i] = np.random.normal(scale=0.6, size=(args.emsize, ))
track_weight=torch.FloatTensor(track_weight)       

# produce sessions_feature_weights    
sessions_weights = get_session_matrix(sessions) 

###############################################################################
# Build the model
###############################################################################
#When using the tied flag, nhid must be equal to emsize
model = TrellisNetModel(track_weight = track_weight,
                        sessions_weights =sessions_weights,
                        ntoken=ntracks,
                        ninp=args.emsize + sessions_weights.shape[1],
                        nhid=args.nhid,
                        nout=args.nout + 1,
                        nlevels=args.nlevels,
                        kernel_size=args.ksize,
                        dilation=args.dilation,
                        dropout=args.dropout,
                        dropouti=args.dropouti,
                        dropouth=args.dropouth,
                        dropoutl=args.dropoutl,
                        emb_dropout=args.emb_dropout,
                        wdrop=args.wdrop,
                        temporalwdrop=args.temporalwdrop,
                        tie_weights=False,
                        repack=args.repack,
                        wnorm=args.wnorm,
                        aux=(args.aux > 0),
                        aux_frequency=args.aux_freq,
                        n_experts=args.n_experts,
                        load=args.load_weight)

criterion = nn.MSELoss() 
optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wdecay)


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




def train(epoch,data_source, label_, sessions_feature):
    model.train()
    total_loss = 0
    total_aux_losses = 0
    start_time = time.time()
    ntokens = len(track_dic)
    hidden = model.init_hidden(args.batch_size)
    eff_history_mode = (args.seq_len > 0 or not args.repack)

    if eff_history_mode:
        validseqlen = args.seq_len - args.horizon
        seq_len = args.seq_len
    else:
        validseqlen = args.horizon
        seq_len = args.horizon

    for batch, i in enumerate(range(0, data_source.size(0) - 1, args.seq_len)):
        # When not using repackaging mode, we DISCARD the first arg.horizon outputs in backprop (which are
        # the "effective history".
        eff_history = args.horizon if eff_history_mode else 0
        if i + eff_history >= data_source.size(0) - 1: 
            continue
        data, targets, label, sf = get_batch_train(data_source, label_, sessions_feature, i, seq_len)
        data = data.t()
        sf = sf.t()

        #if data.shape[0]:# to prevent length filter in get_batch_train romove all batches                       
        
        if args.repack: # repack is not work in my new model since each batch batch_size is different
            hidden = repackage_hidden(hidden)
        else:
            hidden = model.init_hidden(data.size(0))

        optimizer.zero_grad()
        net = nn.DataParallel(model) if data.size(0) > 10 else model
        (raw_output, output, decoded), hidden, all_decoded = net(data,sf,hidden) ##########################slow
        # raw_output.shape  = [12, 20, 29]
        # output.shape = [12, 20, 29]
        # decoded.shape = [12, 20, 50704]
        # hidden[0].shape = hidden[1].shape = [12, 1029, 1]
        # all_decoded.shape = [12, 1, 20, 50704]
        decoded = decoded.transpose(0, 1)
        # decoded.shape = [20, 12, 50704]
        targets = targets[eff_history:].contiguous().view(-1)
        # targets.shape = torch.Size([180])
        label = label[eff_history:].contiguous().view(-1)
        final_decoded = decoded[eff_history:].contiguous().view(-1, args.nout+1)
        #final_decoded.shape = torch.Size([180, 50704])
              
                        
            # remove padding rows 
#        mask_targets = targets!=0
#        targets = targets[targets!=0]
#        label = label[label!=-1]
#        loc = torch.ByteTensor(mask_targets) #<IndexBackward>  <ViewBackward>
#        final_decoded = final_decoded[loc]
#            
        if final_decoded.shape[0]:
            # Loss 1: CE loss
            try:
                final_targets = torch.cat([track_weight[targets],label.unsqueeze(1).float()],dim = 1)
                raw_loss = criterion(final_decoded, final_targets)
            except:
                print(final_decoded.shape)
                print(final_targets.shape)
                print(label.shape)
                print(targets.shape)
                print(output.shape)
                print(decoded.shape)
                print(batch, '   ', i)
#            #qiqi's check 
#            if raw_loss>20:
#            print(raw_loss)
#            print(batch)
#            print(i)
#            print(final_decoded.shape)
#            print(targets.shape)

    
            # Loss 2: Aux loss
            aux_losses = 0
            if args.aux > 0:
                all_decoded = all_decoded[:, :, eff_history:].permute(1, 2, 0, 3).contiguous()  # (N, M, L, C) --> (M, L, N, C)
                aux_size = all_decoded.size(0)
                all_decoded = all_decoded.view(aux_size, -1, args.nout+1)
                
                # remove padding rows
#                all_decoded = all_decoded.transpose(0,1)[loc].transpose(0,1)

                aux_losses = args.aux * sum([criterion(all_decoded[i], final_targets) for i in range(aux_size)])
    

            # Combine losses
            loss = raw_loss + aux_losses #+ alpha_loss + beta_loss
            loss.backward(retain_graph=True) #####################################slow
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
    
            total_loss += raw_loss.data
            if args.aux:
                total_aux_losses += aux_losses.data
                
#        if batch % args.log_interval == 0 and batch > 0:
#            cur_loss = total_loss.item() / args.log_interval
#            cur_aux_loss = total_aux_losses.item() / args.log_interval if args.aux else 0
#            elapsed = time.time() - start_time
#            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
#                  'raw_loss {:5.2f} | aux_loss {:5.2f} | ppl {:8.2f}'.format(
#                   epoch, batch, len(data_source) // validseqlen, lr,
#                   elapsed * 1000 / args.log_interval, 
#                   cur_loss, cur_aux_loss, math.exp(cur_loss)))
#            total_loss = 0
#            total_aux_losses = 0
#            start_time = time.time()
#    
        
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            cur_aux_loss = total_aux_losses.item() / args.log_interval if args.aux else 0
            elapsed = time.time() - start_time
            
            try: 
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'raw_loss {:5.2f} | aux_loss {:5.2f} | ppl {:8.2f}'.format(
                       epoch, batch, len(data_source) // args.seq_len, lr,
                       elapsed * 1000 / args.log_interval, 
                       cur_loss, cur_aux_loss, math.exp(cur_loss)))
            except: 
                print(batch)
                print(i)
                print(cur_loss)
                
            total_loss = 0
            total_aux_losses = 0
            start_time = time.time()

    #            sys.stdout.flush()
    
    raw_output = None
    output = None
    decoded = None
    targets = None
    final_output = None
    final_decoded = None
    all_decoded = None
    all_outputs = None
    final_raw_output = None
        
    return None


def evaluate(data_source, label_, sessions_feature):
    # Turn on evaluation mode which disables dropout.
    model.eval() # will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
    with torch.no_grad(): # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).
        total_loss = 0
        ntokens = len(track_dic)
        batch_size = data_source.size(1)
        hidden = model.init_hidden(batch_size)
        eff_history_mode = (args.seq_len > args.horizon and not args.repack)

        if eff_history_mode:
            validseqlen = args.seq_len - args.horizon
            seq_len = args.seq_len
        else:
            validseqlen = args.horizon
            seq_len = args.horizon

        processed_data_size = 0
        for i in range(0, data_source.size(0) - 1, args.seq_len):
            eff_history = args.horizon if eff_history_mode else 0
            if i + eff_history >= data_source.size(0) - 1: 
                continue
            
            data, targets, label, sf = get_batch_train(data_source, label_, sessions_feature, i, seq_len, evaluation=True)
            data = data.t()
            sf = sf.t()
            
            if args.repack:# repack is not work in my new model since each batch batch_size is different
                hidden = repackage_hidden(hidden)
            else:
                hidden = model.init_hidden(data.size(0))

            net = nn.DataParallel(model) if batch_size > 10 else model
            (_, output, decoded), hidden, _ = net(data, sf, hidden) #######  output; decoded => vector
            decoded = decoded.transpose(0, 1)####
            targets = targets[eff_history:].contiguous().view(-1)
            label = label[eff_history:].contiguous().view(-1)
            final_decoded = decoded[eff_history:].contiguous().view(-1, args.nout+1)
            
                         
            #remove padding rows 
#            mask_targets = targets!=0
#            targets = targets[targets!=0]
#            label = label[label!=-1]
#            loc = torch.ByteTensor(mask_targets) #<IndexBackward>  <ViewBackward>
#            final_decoded = final_decoded[loc]
#            
            if final_decoded.shape[0]:#
     
                final_targets = torch.cat([track_weight[targets],label.unsqueeze(1).float()],dim = 1)
                loss = criterion(final_decoded, final_targets)
                loss = loss.data
        
                total_loss += (data.size(1) - eff_history) * loss
                processed_data_size += data.size(1) - eff_history

        output = None
        decoded = None
        targets = None
        final_output = None
        final_decoded = None

        return total_loss.item() / processed_data_size



def inference(epoch, epoch_start_time, train_loss = 'not avaliable'):
    val_loss = evaluate(val_data,val_label, val_sf)
    test_loss = evaluate(test_data,test_label, test_sf)
    
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                    test_loss, math.exp(test_loss)))
    print('-' * 89)
    
#    try:
#        print('-' * 89)
##        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | '
##              'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
##                                        train_loss, math.exp(train_loss)))
#        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
#              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
#                                         val_loss, math.exp(val_loss)))
#        print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
#              'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
#                                        test_loss, math.exp(test_loss)))
#        print('-' * 89)
#    except:
#        print(val_loss)
#        
    return val_loss, test_loss

if args.eval:
    print("Eval only mode")
    inference(-1, time.time())
    sys.exit(0)

lr = args.lr
best_val_loss = None
all_val_losses = []
all_test_losses = []

try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(epoch,train_data,train_label, train_sf)

        if 't0' in optimizer.param_groups[0]:
            # Average SGD, see (Merity et al. 2017).
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss, test_loss = inference(epoch, epoch_start_time,train_loss)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    # model.save_weights('weights/pretrained.pkl')
                    print('ASGD Saving model (new best validation) in ' + args.save)
                best_val_loss = val_loss
            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss, test_loss = inference(epoch, epoch_start_time, train_loss)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    # model.save_weights('weights/pretrained.pkl')
                    print('Saving model (new best validation) in ' + args.save)
                best_val_loss = val_loss

            if len(all_val_losses) > args.anneal and val_loss > min(all_val_losses[:-args.anneal]):
                print("\n" + "*" * 89)
                if args.asgd and 't0' not in optimizer.param_groups[0]:
                    print('Switching to ASGD')
                    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                    args.save = args.name + "_asgd.pt"
                elif lr > 0.02:
                    print('Annealing learning rate')
                    lr /= 4.0
                    optimizer.param_groups[0]['lr'] = lr
                print("*" * 89 + "\n")

        all_val_losses.append(val_loss)
        all_test_losses.append(test_loss)
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    delete_cmd = input('DO YOU WANT TO DELETE THIS RUN [YES/NO]:')
    if delete_cmd == "YES":
        import os
        os.remove('logs/' + args.name + ".log")
        print("Removed log file")
        os.remove('logs/' + args.name + ".pt")
        print("Removed pt file")

# Load the best saved model
with open(args.save, 'rb') as f:
    model = torch.load(f)
    #model.save_weights('pretrained_music.pkl')

# Run on test data
test_loss = evaluate(test_data,test_label, test_sf)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)