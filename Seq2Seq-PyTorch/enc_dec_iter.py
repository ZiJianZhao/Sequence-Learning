# -*- coding: utf-8 -*-

import logging, codecs, re, random
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np 
from sklearn.cluster import KMeans

def generate_buckets(enc_dec_data, num_buckets):
    enc_dec_data = np.array(enc_dec_data)
    kmeans = KMeans(n_clusters = num_buckets, random_state = 1) # use clustering to decide the buckets
    assignments = kmeans.fit_predict(enc_dec_data) # get the assignments
    # get the max of every cluster
    clusters = np.array([np.max( enc_dec_data[assignments==i], axis=0 ) for i in range(num_buckets)])

    buckets = []
    for i in xrange(num_buckets):
        buckets.append((clusters[i][0], clusters[i][1]))
    return buckets 

def read_dict(path):
    word2idx = {'<pad>' : 0, '<eos>' : 1, '<unk>' : 2}
    idx = 3
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip(' ').strip('\n')
            if len(line) == 0:
                continue
            if word2idx.get(line) == None:
                word2idx[line] = idx
            idx += 1
    return word2idx


def get_enc_dec_text_id(path, enc_word2idx, dec_word2idx):
    enc_data = []
    dec_data = []
    white_spaces = re.compile(r'[ \n\r\t]+')
    index = 0
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid.readlines():
            line = line.strip()
            line_list = line.split('\t=>\t')
            length = len(line_list)
            for i in xrange(1, length):
                enc_list = line_list[0].strip().split()
                dec_list = line_list[i].strip().split()
                enc = [enc_word2idx.get(word) if enc_word2idx.get(word) is not None else enc_word2idx.get('<unk>') for word in enc_list]
                dec = [dec_word2idx.get(word) if dec_word2idx.get(word) is not None else  dec_word2idx.get('<unk>') for word in dec_list]
                enc_data.append(enc)
                dec_data.append(dec)
                if index == 0:
                    print 'Text2digit Preprocess Example:'
                    print line_list[0].strip().encode('utf-8'), '\t=>\t', line_list[1].strip().encode('utf-8')
                    print enc, '\t=>\t', dec
                index += 1
    return enc_data, dec_data



class EncoderDecoderIter(object):
    """This iterator is specially defined for the Couplet Generation
    
    """
    def __init__(self, enc_data, dec_data, batch_size, shuffle=True, pad=0, eos=1):
        
        super(EncoderDecoderIter, self).__init__()
        # initilization
        self.enc_data = enc_data
        self.dec_data = dec_data
        self.data_len = len(self.enc_data)
        self.pad = pad
        self.eos = eos
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = range(self.data_len)
        self.reset()

    def reset(self):
        self.idx = 0
        if self.shuffle:
            random.shuffle(self.indices)
    
    def __iter__(self):
        return self

    def next(self):
        if self.idx > self.data_len - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size] 
        enc = [self.enc_data[i] for i in index]
        dec = [[self.eos] + self.dec_data[i] + [self.eos] for i in index]
        enc_dec = [(enc[i], dec[i]) for i in range(self.batch_size)]
        enc_dec.sort(key = lambda x:len(x[0]), reverse = True)
        enc = [enc_dec[i][0] for i in range(self.batch_size)]
        dec = [enc_dec[i][1] for i in range(self.batch_size)]
        enc_length = [len(enc[i]) for i in range(self.batch_size)]
        enc_len = max([len(l) for l in enc])
        dec_len = max([len(l) for l in dec])
        enc = [l + [self.pad for i in range(enc_len - len(l))] for l in enc]
        dec = [l + [self.pad for i in range(dec_len - len(l))] for l in dec]
        enc = np.asarray(enc, dtype='int64')
        dec = np.asarray(dec, dtype='int64')
        enc_data = torch.from_numpy(enc)
        dec_data = torch.from_numpy(dec)   
        self.idx += self.batch_size
        return enc_data, enc_length, dec_data

