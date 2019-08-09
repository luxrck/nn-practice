import random
import re
import sys
import os
import csv
from functools import reduce

from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import Multi30k
from torchnlp.metrics import bleu

import numpy as np
import spacy

from gensim import corpora

from ash.utils import train, test
from ash.train import App, Trainer, Checkpoint

import pdb


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, d_input, d_attn_weight, d_model):
        super(MultiHeadSelfAttention, self).__init__()
        self.linear_q = nn.Linear(d_input, n_head * d_attn_weight)
        self.linear_k = nn.Linear(d_input, n_head * d_attn_weight)
        self.linear_v = nn.Linear(d_input, n_head * d_attn_weight)
        self.out = nn.Linear(n_head * d_attn_weight, d_model)

    def forward(self, inputs):
        #pdb.set_trace()
        q = self.linear_q(inputs)
        k = self.linear_k(inputs)
        v = self.linear_v(inputs)
        e = q.matmul(k.transpose(-1,-2)) / k.var().sqrt()
        a = F.softmax(e, dim=2)
        x = a.matmul(v)
        x = self.out(x)
        return x


class NerBiLSTMAttn(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, out_classes, dropout_p=0.0, attention=False):
        super(NerBiLSTMAttn, self).__init__()
        self.attention = attention
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm= nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=False, dropout=dropout_p, bidirectional=True)
        self.attn= MultiHeadSelfAttention(n_head=8, d_input=2*hidden_dim, d_attn_weight=2*hidden_dim, d_model=2*hidden_dim)
        if self.attention:
            out_in_dim = 4*hidden_dim
        else:
            out_in_dim = 2*hidden_dim
        self.nn = nn.Sequential(
                nn.Linear(out_in_dim, out_in_dim),
                    nn.Dropout(p=dropout_p),
                    nn.ReLU(),
                    nn.Linear(out_in_dim, out_classes),
                    nn.LogSoftmax(dim=2))
    def forward(self, inputs):
        # pdb.set_trace()
        inputs, lengths = inputs
        x = self.emb(inputs)
        lengths_list = lengths.view(-1).tolist()
        packed_x = pack(x, lengths_list)
        packed_oh, _ = self.lstm(packed_x)
        oh, lengths_list = unpack(packed_oh)
        oh = oh.transpose(0,1)
        if self.attention:
            wh = self.attn(oh)
            # wh = wh.transpose(0,1)
            x = torch.cat([oh, wh], dim=2)
        else:
            x = oh
        # x: (seq_len, batch, n_directions * hidden_size + wh.size(2))
        x = x.transpose(0,1)
        x = self.nn(x)
        return x


if __name__ == "__main__":
    lr = 0.0004
    adam_eps = 1e-4
    batch_size = 64
    epochs = 10
    num_layers = 1
    embedding_dim = 200
    hidden_size = 200
    #max_padding_sentence_len = 20
    dropout_p = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEN = Field(include_lengths=True,
                pad_token="<pad>",
                unk_token="<unk>",
                tokenize=lambda x: x.split(),
                lower=False)
    TAG = Field(include_lengths=True,
                pad_token="o",
                unk_token=None,
                tokenize=lambda x: x.split(),
                lower=True)
    _train, _test = TabularDataset.splits(path="data/conll/CoNLL-2003", root="data", train="train.tsv", test="testa.tsv", format='tsv', skip_header=False, fields=[("sen", SEN), ("tag", TAG)], csv_reader_params={"quoting": csv.QUOTE_NONE})
    SEN.build_vocab(_train)
    TAG.build_vocab(_train, min_freq=1)
    # pdb.set_trace()
    # btrain, btesta, dictionary = preprocess()
    train_iter = BucketIterator(_train, batch_size=batch_size, train=True,
                                 sort_within_batch=True,
                                 sort_key=lambda x: len(x.sen), repeat=False,
                                 device=device)
    test_iter = BucketIterator(_test, batch_size=1, train=False,
                                 sort_key=None,
                                 sort=False,
                                 repeat=False,
                                 device=device)

    model = NerBiLSTMAttn(num_embeddings=len(SEN.vocab), embedding_dim=embedding_dim, hidden_dim=hidden_size, out_classes=8, attention=True)
    model = model.to(device)
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)
    # model = model.half()
    criterion = nn.NLLLoss(ignore_index = 1)
    criterion = criterion.to(device)
    # criterion = criterion.half()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=adam_eps)
    #pdb.set_trace()

    def itos(s):
        s.transpose(0,1)
        return [" ".join([SEN.vocab.itos[ix] for ix in sent]) for sent in s]

    @train(model, criterion, optimizer, train_iter, epochs, device=device, immediate=False, checkpoint=True, verbose=False)
    def ner_train(model, criterion, data):
        sent, label = data.sen, data.tag
        y_predict = model(sent)
        loss = torch.tensor([0.0]).cuda()
        label = label[0]
        # pdb.set_trace()
        try:
            for i in range(label.size(0)):
               loss += criterion(y_predict[i], label[i])
        except Exception as e:
            pdb.set_trace()
        #import pdb; pdb.set_trace()
        return y_predict, loss

    def ner_eval(model, criterion, optimizer):
        state = torch.load(f"checkpoint/NerBiLSTMAttn.{4}.torch")
        model.load_state_dict(state["model"])
        model = model.to(device)
        criterion = criterion.to(device)
        optimizer.load_state_dict(state["optim"])

        with torch.no_grad():
            for i,data in tqdm(enumerate(test_iter)):
                #pdb.set_trace()
                sent, label = data.sen, data.tag
                # y_predict: (seq_len, batch, classes)
                y_predict = model(sent)
                y_predict = y_predict.transpose(0,1)
                sent = sent[0]
                label = label[0]
                sent = sent.view(sent.size(1), -1)
                label = label.view(label.size(1), -1)
                _, y_predict = torch.max(y_predict, dim=2)
                for i,out in enumerate(y_predict):
                    s_in = [SEN.vocab.itos[ix] for ix in sent[i]]
                    s_out = [TAG.vocab.itos[ix] for ix in out]
                    t_out = [TAG.vocab.itos[ix] for ix in label[i]]
                    print("S: " + " ".join(s_in))
                    print("T: " + " ".join(t_out))
                    print("D: " + " ".join(s_out))
    
    ner_train()
    ner_eval(model, criterion, optimizer)
