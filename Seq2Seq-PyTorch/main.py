# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys, re, time, random, logging
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np 

from model import Seq2Seq, MaskedSeq2Seq, GlobalAttentionSeq2Seq
from enc_dec_iter import generate_buckets, read_dict, get_enc_dec_text_id, EncoderDecoderIter

def find_free_gpu():
    print('Begin detecting free GPU')
    os.system('nvidia-smi > gpuinfo')
    f = open('gpuinfo', 'r')
    lines = f.readlines()
    if len(lines) < 3:
        return -1
    f.close()
    gpu0 = float(lines[8].split()[12][:-1])
    try:
        gpu1 = float(lines[11].split()[12][:-1])
    except:
        gpu1 = 1.0
    if gpu0 < 25:
        print('GPU0 is free -- take it!')
        return 0
    elif gpu1 < 25:
        print('GPU1 is free -- take it!')
        return 1
    else:
        print('No GPU is free!')
        return -1

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(1)

mode = 'train'
data_dir = '/slfs1/users/zjz17/github/data/couplet/data/final'
share_embed_weight = True
if share_embed_weight:
    enc_vocab_file = 'alllist.txt'
    dec_vocab_file = 'alllist.txt'
else:
    enc_vocab_file = 'shanglist.txt'
    dec_vocab_file = 'xialist.txt'
train_file = 'train.txt'
valid_file = 'valid.txt'
test_file = 'test.txt'

enc_word2idx = read_dict(os.path.join(data_dir, enc_vocab_file))
dec_word2idx = read_dict(os.path.join(data_dir, dec_vocab_file))
ignore_label = dec_word2idx.get('<pad>')


if mode == 'train':
    # ----------------- 1. Configure logging module  ---------------------
    # This is needed only in train mode
    logging.basicConfig(
        level = logging.DEBUG,
        format = '%(asctime)s %(message)s', 
        datefmt = '%m-%d %H:%M:%S %p',  
        filename = 'Log',
        filemode = 'w'
    )
    logger = logging.getLogger()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
 

def train_on_epoch(data_iter, model, optimizer, criterion, is_train=True):
    
    frequent = data_iter.data_len / data_iter.batch_size / 10 # log frequency

    total_loss = 0.
    start_time = time.time()
    for (i, (enc_data, dec_data)) in enumerate(data_iter):
        enc_data = Variable(enc_data, volatile=not is_train) 
        dec_data = Variable(dec_data, volatile=not is_train)
        if use_cuda:
            enc_data, dec_data = enc_data.cuda(), dec_data.cuda()
        target = dec_data[:, 1:].contiguous().view(-1)
        softmax = model(enc_data, dec_data[:, :-1])
        loss = criterion(softmax, target)
        total_loss += loss.data[0]
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i % frequent == 0:
            perplexity = math.exp(total_loss/(i+1))
            elapsed_time = time.time() - start_time
            if i == 0:
                speed = enc_data.size()[0] / elapsed_time
            else:
                speed = frequent * enc_data.size()[0] / elapsed_time
            start_time = time.time()
            if is_train:
                logger.info('Batch [%d]\tTrain-Perplexity: %.2f\tSpeed: %.2f samples/sec' % 
                    (i, perplexity, speed))
    return math.exp(total_loss / (i+1))

def train():
    # -----------------2. Params Defination --------------------------------

    params_dir =  'checkpoints'
    params_prefix = 'seq2seq'
    num_buckets = 3
    batch_size = 32

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    # ----------------------3. Data Iterator Defination ---------------------
    enc_train, dec_train = get_enc_dec_text_id(os.path.join(data_dir, train_file), enc_word2idx, dec_word2idx)
    enc_valid, dec_valid = get_enc_dec_text_id(os.path.join(data_dir, valid_file), enc_word2idx, dec_word2idx)
    sequence_length = []
    # +2 because the decoder sequence will add eos in its beginning and ending
    for i in range(len(enc_train)):
        sequence_length.append((len(enc_train[i]), len(dec_train[i])+2)) 
    for i in range(len(enc_valid)):
        sequence_length.append((len(enc_valid[i]), len(dec_valid[i])+2))
    buckets = generate_buckets(sequence_length, num_buckets)
    train_iter = EncoderDecoderIter(
        enc_data = enc_train, 
        dec_data = dec_train, 
        batch_size = batch_size, 
        buckets = buckets, 
        shuffle = True,
        pad = enc_word2idx.get('<pad>'), 
        eos = enc_word2idx.get('<eos>')
    )
    valid_iter = EncoderDecoderIter(
        enc_data = enc_valid, 
        dec_data = dec_valid, 
        batch_size = batch_size, 
        buckets = buckets,
        shuffle = False, 
        pad = enc_word2idx.get('<pad>'), 
        eos = enc_word2idx.get('<eos>')
    )
    # ----------------------3. Model Training Details ---------------------
    ## Model Definition
    model = Seq2Seq(len(enc_word2idx), len(dec_word2idx))
    if use_cuda:
        model = model.cuda()
    ## Optimizer Definition
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ## Model Initialization

    ## If pretrained, initial with the saved parameters
    pretrained = True
    begin_epoch = 0 
    best_ppl = 0xffffffff
    if pretrained:
        try:
            checkpoint = torch.load('%s/%s' % (params_dir, 'best_model.pt'))
            begin_epoch = checkpoint['epoch'] + 1
            best_ppl = checkpoint['best_ppl']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info('Loading checkpoints at epoch %d' % (begin_epoch-1))
        except:
            pass
    ## Criterion Definition: eliminate the influence of the padding
    weight =  torch.ones(len(dec_word2idx))
    weight[ignore_label] = 0 
    criterion = nn.NLLLoss(weight, size_average=True)
    if use_cuda:
        criterion = criterion.cuda()

    ## Training Process
    epochs = 5
    for epoch in xrange(begin_epoch, epochs):
        logger.info('Epoch [%d]\tbegin training............' % epoch)
        start_time = time.time()
        train_ppl = train_on_epoch(train_iter, model, optimizer, criterion, is_train=True)
        elapsed_time = time.time() - start_time
        logger.info('Epoch [%d]\tTrain Perplexity: %.2f' % (epoch, train_ppl))
        logger.info('Epoch [%d]\tElapsed time: %.2f' % (epoch, elapsed_time))
        valid_ppl = train_on_epoch(valid_iter, model, optimizer, criterion, is_train=False)
        logger.info('Epoch [%d]\tValid Perplexity: %.2f' % (epoch, valid_ppl))
        saved_dict = {'epoch': epoch, 'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 'best_ppl': best_ppl}
        torch.save(saved_dict, '%s/%s-%d.pt' % (params_dir, params_prefix, epoch))
        logger.info('Saving paramters in %s/%s-%d.pt' % (params_dir, params_prefix, epoch+1))
        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            torch.save(saved_dict, '%s/%s.pt' % (params_dir, 'best_model'))
            logger.info('Updating best paramters in %s/%s.pt' % (params_dir, 'best_model'))
        logger.info('=================================================')
        train_iter.reset()
        valid_iter.reset()

train()
