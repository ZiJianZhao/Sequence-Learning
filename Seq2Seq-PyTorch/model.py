# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys, re, time
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()

class MaskedRNN(nn.Module):
    ''' MaskedRNN '''
    def __init__(self, input_size, embed_size, hidden_size, bidirectional=False, n_layers=1):
        super(MaskedRNN, self).__init__()
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.rnn_cell = nn.GRUCell(embed_size, hidden_size)
        if self.bidirectional:
            self.b_rnn_cell = nn.GRUCell(embed_size, hidden_size)
        self.init_weights()

    def rnn_forward(self, embedded, mask, init_state, direction_forward=True):
        input_len = embedded.size()[0]
        if direction_forward:
            rnn_cell = self.rnn_cell
        else:
            rnn_cell = self.b_rnn_cell
        outputs = []
        hiddens = [init_state for l in range(self.n_layers)]
        for i in range(input_len):
            if direction_forward:
                idx = i
            else:
                idx = input_len - 1 - i
            output = embedded[idx, :, :]
            mask_i = mask[idx, :].contiguous().view(-1, 1).expand_as(hiddens[l]) 
            for l in range(self.n_layers):
                h = rnn_cell(output, hiddens[l])
                #h = mask_i * h + (1 - mask_i) * hiddens[l]
                hiddens[l] = h
            if direction_forward:
                outputs.append(h.expand(1, h.size()[0], h.size()[1]))        
            else:
                outputs.insert(0, h.expand(1, h.size()[0], h.size()[1]))
        hiddens = [h.expand(1, h.size()[0], h.size()[1]) for h in hiddens]
        return torch.cat(outputs, 0), torch.cat(hiddens, 0)


    def forward(self, input, hidden=None, pad=0):
        batch_size = input.size(0)
        mask = (input != pad).float().contiguous().transpose(0,1)
        if hidden is None:
            init_state = self.init_hidden(batch_size)
        else:
            init_state = hidden
        embedded = self.embedding(input).transpose(0,1)
        output, hidden = self.rnn_forward(embedded, mask, init_state, direction_forward=True)
        if self.bidirectional:
            b_output, b_hidden = self.rnn_forward(embedded, mask, init_state, direction_forward=True)
            output = torch.cat([output, b_output], 2)
        return output, hidden

    def init_weights(self, init_range=0.05):
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.rnn_cell.weight_ih.data.uniform_(-init_range, init_range)
        self.rnn_cell.weight_hh.data.uniform_(-init_range, init_range)
        self.rnn_cell.bias_ih.data.uniform_(-init_range, init_range)
        self.rnn_cell.bias_hh.data.uniform_(-init_range, init_range)
        if self.bidirectional:
            self.b_rnn_cell.weight_ih.data.uniform_(-init_range, init_range)
            self.b_rnn_cell.weight_hh.data.uniform_(-init_range, init_range)
            self.b_rnn_cell.bias_ih.data.uniform_(-init_range, init_range)
            self.b_rnn_cell.bias_hh.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if USE_CUDA:
            return hidden.cuda()
        else:
            return hidden



class MaskedSeq2Seq(nn.Module):
    '''Masked Sequence to sequence learning '''
    def __init__(self, enc_input_size, dec_input_size):
        super(MaskedSeq2Seq, self).__init__()
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.enc_num_layer = 1
        self.dec_num_layer = 1
        self.enc_embed_size = 4
        self.dec_embed_size = 4
        self.enc_hidden_size = 8
        self.dec_hidden_size = 8
        self.bidirectional = False
        self.share_embed_weight = True
        self.encoder = MaskedRNN(self.enc_input_size, self.enc_embed_size, 
            self.enc_hidden_size, self.bidirectional, self.enc_num_layer)
        self.encoder2decoder = nn.Linear(self.enc_num_layer * self.enc_hidden_size, self.dec_hidden_size)
        self.decoder = MaskedRNN(self.dec_input_size, self.dec_embed_size, 
            self.dec_hidden_size, False, self.dec_num_layer)
        self.decoder2vocab = nn.Linear(self.dec_hidden_size, self.dec_input_size)
        self.softmax = nn.LogSoftmax()
        if self.share_embed_weight:
            self.encoder.embedding.weight = self.decoder.embedding.weight
        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, enc_input, dec_input):
        '''
            enc_input: batch_size * enc_len
            dec_input: batch_size * dec_len
        '''
        batch_size = enc_input.size()[0]
        _, hidden = self.encoder.forward(enc_input)
        hidden = hidden.transpose(0, 1).view(batch_size, -1)
        hidden = self.encoder2decoder(hidden)
        output, _ = self.decoder.forward(dec_input, hidden)
        output = output.transpose(0,1).contiguous().view(-1, self.dec_hidden_size)
        output = self.decoder2vocab(output)
        softmax = self.softmax(output)
        return softmax

class Seq2Seq(nn.Module):
    '''Sequence to sequence learning '''
    def __init__(self, enc_input_size, dec_input_size):
        super(Seq2Seq, self).__init__()
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.enc_num_layers = 1
        self.dec_num_layers = 1
        self.enc_embed_size = 4
        self.dec_embed_size = 4
        self.enc_hidden_size = 8
        self.dec_hidden_size = 8
        self.bidirectional = False
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.share_embed_weight = True
        self.encoder_embedding = nn.Embedding(self.enc_input_size, self.enc_embed_size, padding_idx=0)
        self.encoder_rnn = nn.GRU(self.enc_embed_size, self.enc_hidden_size, self.enc_num_layers,
                            bidirectional=self.bidirectional, batch_first=False)
        self.encoder2decoder = nn.Linear(self.enc_num_layers * self.num_directions * self.enc_hidden_size, self.dec_hidden_size)
        self.decoder_embedding = nn.Embedding(self.dec_input_size, self.dec_embed_size, padding_idx=0)
        self.decoder_rnn = nn.GRU(self.dec_embed_size, self.dec_hidden_size, self.dec_num_layers)
        self.decoder2vocab = nn.Linear(self.dec_hidden_size, self.dec_input_size)
        self.softmax = nn.LogSoftmax()
        if self.share_embed_weight:
            self.encoder_embedding.weight = self.decoder_embedding.weight
        self.init_weights()

    def forward(self, enc_input, dec_input):
        '''
            enc_input: batch_size * enc_len
            dec_input: batch_size * dec_len
        '''
        batch_size = enc_input.size(0)
        enc_embedded = self.encoder_embedding(enc_input).transpose(0,1)
        enc_init_hidden = self.encoder_init_hidden(batch_size)
        _, hidden = self.encoder_rnn(enc_embedded, enc_init_hidden)
        hidden = hidden.transpose(0, 1).view(batch_size, -1)
        hidden = self.encoder2decoder(hidden).view(1, batch_size, -1).expand(self.dec_num_layers, batch_size, self.dec_hidden_size)
        dec_embedded = self.decoder_embedding(dec_input).transpose(0,1)
        output, _ = self.decoder_rnn(dec_embedded, hidden)
        output = output.transpose(0,1).contiguous().view(-1, self.dec_hidden_size)
        output = self.decoder2vocab(output)
        softmax = self.softmax(output)
        return softmax

    def beam_search(self, enc_input):
        """Beam Search: default pad=0, eos=1, unk=2
        Args:
            inputs (list): a list of ints
        """
        global USE_CUDA 
        USE_CUDA = False
        beam_width = 10
        max_len = 25
        eos = 1
        unk = 2
        pad = 0
        enc_input = Variable(torch.LongTensor([enc_input]), volatile=True)
        enc_embedded = self.encoder_embedding(enc_input).transpose(0,1)
        enc_init_hidden = self.encoder_init_hidden(1)
        _, hidden = self.encoder_rnn(enc_embedded, enc_init_hidden)
        hidden = hidden.transpose(0, 1).view(1, -1)
        hidden = self.encoder2decoder(hidden).view(1, 1, -1).expand(self.dec_num_layers, 1, self.dec_hidden_size)

        beam = [(0, [eos], hidden)]
        finished = []
        for step in xrange(max_len):
            new_beam = []
            for (score, sent, state) in beam:
                dec_input = Variable(torch.LongTensor([[sent[-1]]]), volatile=True)
                dec_embed = self.decoder_embedding(dec_input).transpose(0, 1)
                output, hidden = self.decoder_rnn(dec_embed, state)
                output = output.transpose(0,1).contiguous().view(-1, self.dec_hidden_size)
                output = self.decoder2vocab(output)
                softmax = self.softmax(output)
                top_values, top_indices = softmax.topk(beam_width)
                top_values = top_values.data[0].numpy()
                top_indices = top_indices.data[0].numpy()
                for i in xrange(beam_width):
                    if (top_indices[i] == pad or top_indices[i] == unk):
                        continue
                    elif top_indices[i] == eos:
                        finished.append((score + top_values[i], sent + [top_indices[i]]))
                    else:
                        tmp_hidden = hidden.clone()
                        new_beam.append((score + top_values[i], sent + [top_indices[i]], tmp_hidden))
            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

        for (score, sent, state) in beam:
            finished.append((score, sent))
        return sorted(finished, key=lambda x: x[0], reverse=True)[:beam_width]
    
    def encoder_init_hidden(self, batch_size):
        num = self.enc_num_layers
        if self.bidirectional:
            num = num * 2
        hidden = Variable(torch.zeros(num, batch_size, self.enc_hidden_size))
        if USE_CUDA:
            return hidden.cuda()
        else:
            return hidden

    def init_weights(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

class NolinearAttention(nn.Module):
    '''Neural Machine Translation By Jointly Learning to Align and Translate
        e = V^T * tanh(W * [enc_hiddens; dec_hidden])
    '''
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(NolinearAttention, self).__init__()
        self.W = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size, bias=False)
        self.V = nn.Linear(dec_hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax()
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, enc_hiddens, dec_hidden):
        """attention calculation
        
        Args:
            enc_hiddens: batch_size * seq_len * hidden_size
            dec_hidden: batch_size * hidden_size
        """
        batch_size, hidden_size = dec_hidden.size()
        seq_len = enc_hiddens.size(1)
        dec_hiddens = dec_hidden.view(batch_size, 1, -1).expand(batch_size, seq_len, hidden_size)
        hiddens = (torch.cat([enc_hiddens, dec_hiddens], 2)).view(batch_size * seq_len, -1)
        scores = self.V(self.tanh(self.W(hiddens))).view(batch_size, -1) # batch_size * seq_len
        attn = self.sm(scores).view(batch_size, 1, -1)
        ctx = torch.bmm(attn, enc_hiddens).squeeze(1)
        return ctx, attn  

class GeneralAttention(nn.Module):
    '''Effective Approaches to Attention-based Neural Machine Translation
        e = enc_hiddens * W * dec_hidden
    '''
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(GeneralAttention, self).__init__()
        self.W = nn.Linear(dec_hidden_size, enc_hidden_size, bias=False)    
        self.sm = nn.Softmax()
        self.init_weights()           
    
    def init_weights(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, enc_hiddens, dec_hidden):
        """attention calculation
        
        Args:
            enc_hiddens: batch_size * seq_len * hidden_size
            dec_hidden: batch_size * hidden_size
        """
        batch_size = dec_hidden.size(0)
        dec_atten = self.W(dec_hidden).unsqueeze(2)  
        scores = torch.bmm(enc_hiddens, dec_atten).squeeze(2)  
        attn = self.sm(scores).view(batch_size, 1, -1)
        ctx = torch.bmm(attn, enc_hiddens).squeeze(1)
        return ctx, attn

class DotAttention(nn.Module):
    '''Effective Approaches to Attention-based Neural Machine Translation
        e = enc_hiddens^T * dec_hidden

        This class need that the enc_hiddens num_features equals to the dec_hidden num_features 
    '''
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(DotAttention, self).__init__()
        assert enc_hidden_size == dec_hidden_size    
        self.sm = nn.Softmax()
        self.init_weights()           
    
    def init_weights(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, enc_hiddens, dec_hidden):
        """attention calculation
        
        Args:
            enc_hiddens: batch_size * seq_len * hidden_size
            dec_hidden: batch_size * hidden_size
        """
        batch_size = dec_hidden.size(0)
        dec_hidden = dec_hidden.unsqueeze(2) 
        scores = torch.bmm(enc_hiddens, dec_hidden).squeeze(2)  
        attn = self.sm(scores).view(batch_size, 1, -1)
        ctx = torch.bmm(attn, enc_hiddens).squeeze(1)
        return ctx, attn

class ConcatAttention(nn.Module):
    '''Effective Approaches to Attention-based Neural Machine Translation
        e = W * [enc_hiddens; dec_hidden]
    '''
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(ConcatAttention, self).__init__()
        self.W = nn.Linear(enc_hidden_size + dec_hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax()
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def forward(self, enc_hiddens, dec_hidden):
        """attention calculation
        
        Args:
            enc_hiddens: batch_size * seq_len * hidden_size
            dec_hidden: batch_size * hidden_size
        """
        batch_size, hidden_size = dec_hidden.size()
        seq_len = enc_hiddens.size(1)
        dec_hiddens = dec_hidden.view(batch_size, 1, -1).expand(batch_size, seq_len, hidden_size)
        hiddens = (torch.cat([enc_hiddens, dec_hiddens], 2)).view(batch_size * seq_len, -1)
        scores = self.W(hiddens)
        attn = self.sm(scores).view(batch_size, 1, -1)
        ctx = torch.bmm(attn, enc_hiddens).squeeze(1)
        return ctx, attn  

class GlobalAttentionSeq2Seq(nn.Module):
    '''Sequence to sequence learning '''
    def __init__(self, enc_input_size, dec_input_size):
        super(GlobalAttentionSeq2Seq, self).__init__()
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.enc_num_layers = 1
        self.dec_num_layers = 1 
        self.enc_embed_size = 4
        self.dec_embed_size = 4
        self.enc_hidden_size = 8
        self.dec_hidden_size = 8
        self.bidirectional = False
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.share_embed_weight = True
        self.encoder_embedding = nn.Embedding(self.enc_input_size, self.enc_embed_size, padding_idx=0)
        self.encoder_rnn = nn.GRU(self.enc_embed_size, self.enc_hidden_size, self.enc_num_layers,
                            bidirectional=self.bidirectional, batch_first=False)
        self.atten_func = GeneralAttention(self.enc_hidden_size*self.num_directions, self.dec_hidden_size)
        self.encoder2decoder = nn.Linear(self.enc_num_layers * self.num_directions * self.enc_hidden_size, self.dec_hidden_size)
        self.decoder_embedding = nn.Embedding(self.dec_input_size, self.dec_embed_size, padding_idx=0)
        self.decoder_rnn_cell = nn.GRUCell(self.dec_embed_size+self.enc_hidden_size*self.num_directions, self.dec_hidden_size)
        self.decoder_layer_linear = nn.Linear(self.dec_hidden_size, self.dec_embed_size)
        self.decoder2vocab = nn.Linear(self.dec_hidden_size, self.dec_input_size)
        self.softmax = nn.LogSoftmax()
        if self.share_embed_weight:
            self.encoder_embedding.weight = self.decoder_embedding.weight
        self.init_weights() 

    def forward(self, enc_input, dec_input):
        '''
            enc_input: batch_size * enc_len
            dec_input: batch_size * dec_len
        '''
        # Encoder
        batch_size = enc_input.size(0)
        dec_len = dec_input.size(1)
        enc_embedded = self.encoder_embedding(enc_input).transpose(0,1)
        enc_init_hidden = self.encoder_init_hidden(batch_size)
        enc_outputs, hidden = self.encoder_rnn(enc_embedded, enc_init_hidden)
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        enc_outputs = enc_outputs.transpose(0, 1)
        # Decoder
        outputs = []
        hidden = self.encoder2decoder(hidden)
        hiddens = [hidden for i in range(self.dec_num_layers)]
        dec_embedded = self.decoder_embedding(dec_input)
        for emb_t in dec_embedded.chunk(dec_len, dim=1):
            inpu = emb_t.squeeze(1)
            for l in range(self.dec_num_layers):
                c_t, _ = self.atten_func(enc_outputs, hiddens[l])  # attention
                hidden = self.decoder_rnn_cell(torch.cat([inpu, c_t], 1), hiddens[l])
                hiddens[l] = hidden
                inpu = self.decoder_layer_linear(hidden)
            outputs.append(hidden)
        output = torch.stack(outputs, dim=0)
        output = output.transpose(0,1).contiguous().view(-1, self.dec_hidden_size)
        output = self.decoder2vocab(output)
        softmax = self.softmax(output)
        return softmax

    def beam_search(self, enc_input):
        """Beam Search: default pad=0, eos=1, unk=2
        Args:
            inputs (list): a list of ints
        """
        global USE_CUDA 
        USE_CUDA = False
        beam_width = 10
        max_len = 25
        eos = 1
        unk = 2
        pad = 0
        enc_input = Variable(torch.LongTensor([enc_input]), volatile=True)
        enc_embedded = self.encoder_embedding(enc_input).transpose(0,1)
        enc_init_hidden = self.encoder_init_hidden(1)
        enc_outputs, hidden = self.encoder_rnn(enc_embedded, enc_init_hidden)
        hidden = hidden.transpose(0, 1).contiguous().view(1, -1)
        hidden = self.encoder2decoder(hidden)
        hiddens = [hidden for i in range(self.dec_num_layers)]
        enc_outputs = enc_outputs.transpose(0, 1)

        beam = [(0, [eos], hiddens)]
        finished = []
        for step in xrange(max_len):
            new_beam = []
            for (score, sent, states) in beam:
                new_states = []
                dec_input = Variable(torch.LongTensor([[sent[-1]]]), volatile=True)
                dec_embed = self.decoder_embedding(dec_input).transpose(0, 1).squeeze(1)
                inpu = dec_embed
                for l in xrange(self.dec_num_layers):
                    c_t, _ = self.atten_func(enc_outputs, states[l])  # attention
                    hidden = self.decoder_rnn_cell(torch.cat([inpu, c_t], 1), states[l])
                    new_states.append(hidden)
                    inpu = self.decoder_layer_linear(hidden)    
                output = torch.stack([hidden], dim=0)
                output = output.transpose(0,1).contiguous().view(-1, self.dec_hidden_size)
                output = self.decoder2vocab(output)
                softmax = self.softmax(output)
                top_values, top_indices = softmax.topk(beam_width)
                top_values = top_values.data[0].numpy()
                top_indices = top_indices.data[0].numpy()
                for i in xrange(beam_width):
                    if (top_indices[i] == pad or top_indices[i] == unk):
                        continue
                    elif top_indices[i] == eos:
                        finished.append((score + top_values[i], sent + [top_indices[i]]))
                    else:
                        tmp_states = copy.deepcopy(new_states)
                        new_beam.append((score + top_values[i], sent + [top_indices[i]], tmp_states))
            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

        for (score, sent, state) in beam:
            finished.append((score, sent))
        return sorted(finished, key=lambda x: x[0], reverse=True)[:beam_width]
    
    def encoder_init_hidden(self, batch_size):
        num = self.enc_num_layers
        if self.bidirectional:
            num = num * 2
        hidden = Variable(torch.zeros(num, batch_size, self.enc_hidden_size))
        if USE_CUDA:
            return hidden.cuda()
        else:
            return hidden

    def init_weights(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)