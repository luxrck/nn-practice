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


class GeneralAttention(nn.Module):
    def __init__(self):
        super(GeneralAttention, self).__init__()
    def forward(self, Q, K, V, mask=None):
        e = Q.matmul(K.transpose(-1, -2))
        e = e / K.var().sqrt()
        if mask is not None:
            e = e.dot(mask)
        a = F.softmax(e, dim=-1)
        y = a.matmul(V)
        return y

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, d_input, d_attn, d_out):
        super(MultiHeadSelfAttention, self).__init__()
        self.linear_q = nn.Linear(d_input, n_head * d_attn)
        self.linear_k = nn.Linear(d_input, n_head * d_attn)
        self.linear_v = nn.Linear(d_input, n_head * d_attn)
        self.attn = GeneralAttention()
        self.out = nn.Linear(n_head * d_attn, d_out)
    def forward(self, inputs):
        #pdb.set_trace()
        q = self.linear_q(inputs)
        k = self.linear_k(inputs)
        v = self.linear_v(inputs)

        x = self.attn(q, k, v)
        x = self.out(x)
        return x


class NerBiLSTMAttn(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, out_classes, dropout_p=0.0, attention=False):
        super(NerBiLSTMAttn, self).__init__()
        self.attention = attention
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm= nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=False, dropout=dropout_p, bidirectional=True)
        self.attn= MultiHeadSelfAttention(n_head=8, d_input=2*hidden_dim, d_attn=2*hidden_dim, d_out=2*hidden_dim)
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
    #model = model.to(device)
    #def init_weights(m):
    #    for name, param in m.named_parameters():
    #        nn.init.uniform_(param.data, -0.08, 0.08)
    #model.apply(init_weights)
    ## model = model.half()
    #criterion = nn.NLLLoss(ignore_index = 1)
    #criterion = criterion.to(device)
    ## criterion = criterion.half()
    #optimizer = optim.Adam(model.parameters(), lr=lr, eps=adam_eps)
    #pdb.set_trace()

    app = Trainer(App(model=model, criterion=nn.NLLLoss(ignore_index=SEN.vocab.stoi["<pad>"])))

    def itos(s):
        s.transpose(0,1)
        return [" ".join([SEN.vocab.itos[ix] for ix in sent]) for sent in s]

    #@train(model, criterion, optimizer, train_iter, epochs, device=device, immediate=False, checkpoint=True, verbose=False)
    @app.on("train")
    def ner_train(e):
        e.model.zero_grad()
        sent, label = e.batch.sen, e.batch.tag
        y_predict = e.model(sent)
        loss = 0.0
        label = label[0]
        # pdb.set_trace()
        try:
            for i in range(label.size(0)):
               loss += e.criterion(y_predict[i], label[i])
        except Exception as e:
            pdb.set_trace()
        #import pdb; pdb.set_trace()
        loss.backward()
        e.optimizer.step()

    def ner_eval(app, test_iter):
        with torch.no_grad():
            for i,data in tqdm(enumerate(test_iter)):
                #pdb.set_trace()
                sent, label = data.sen, data.tag
                # y_predict: (seq_len, batch, classes)
                y_predict = app.model(sent)
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

    app.fastforward()   \
       .save_every(iters=1000)  \
       .set_optimizer(optim.Adam, lr=0.0003, eps=1e-4) \
       .to("auto")  \
       .half()  \
       .run(train_iter, max_iters=4000, train=True)

    ner_eval(app, test_iter)
