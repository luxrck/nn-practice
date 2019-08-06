import pdb
import random
import re
import sys
import csv
import os
from functools import reduce

from tqdm import tqdm

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import Multi30k
from torchnlp.metrics import bleu

import numpy as np
import spacy

from gensim import corpora

from sparkle.utils import train, test
from sparkle.app import App



class CharRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, dropout_p=0.5):
        super(CharRNN, self).__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_size,
                           batch_first=False,
                           num_layers=num_layers,
                           bidirectional=True,
                           dropout=dropout_p)
        self.out = nn.Sequential(nn.Linear(2 * hidden_size, 4 * hidden_size),
                                 nn.Dropout(p=dropout_p),
                                 nn.ReLU(),
                                 nn.Linear(4 * hidden_size, num_embeddings),
                                 nn.LogSoftmax(dim=2))

    def forward(self, inputs, hc=None):
        inputs = self.emb(inputs)

        # out: (seq_len, batch, num_directions * hidden_size)
        out, hc = self.rnn(inputs, hc)
        out = self.out(out)

        return out, hc


def preprocess(**kwargs):
    SRC = Field(include_lengths=False,
                init_token=None,
                eos_token="<eos>",
                pad_token="<pad>",
                unk_token="<unk>",
                lower=True,
                batch_first=False,
                tokenize=lambda text: list(text.strip()))
    _train, = TabularDataset.splits(path="data/poetry", root="data", train="poetry.train.tsv", test=None, format='tsv', skip_header=False, fields=[("text", SRC)], csv_reader_params={"quoting": csv.QUOTE_NONE})
    SRC.build_vocab(_train, min_freq=5)
    train_iter = BucketIterator(_train, batch_size=kwargs["batch_size"], train=True,
                                 sort_within_batch=True,
                                 sort_key=lambda x: (len(x.text)), repeat=False,
                                 device=device)
    return train_iter, SRC



def lm_eval(SRC, model, criterion, device, checkpoint, max_len=128):
    def pick(y, topk=5):
        _, topi = y.squeeze().topk(topk)
        return topi[torch.randint(high=topi.size(0), size=(1,)).squeeze().item()]
    state = torch.load(f"checkpoint/CharRNN.{checkpoint}.torch")
    model.load_state_dict(state["model"])
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer.load_state_dict(state["optim"])

    with torch.no_grad():
        while True:
            sent = input("> ").strip()
            if sent == "exit": return

            out = list(sent)

            inputs = [SRC.vocab.stoi[s] for s in sent]
            inputs = torch.tensor(inputs).to(device)
            inputs = inputs.view(inputs.size(0), 1)
            pdb.set_trace()
            y, hc = model(inputs)
            _, y = y[-1].topk(1)
            y = torch.tensor([[y]]).to(device)

            while True:
                y_predict, hc = model(y, hc)
                #pdb.set_trace()
                target = pick(y_predict, topk=5).item()
                if target == SRC.vocab.stoi["<eos>"] or len(out) == max_len:
                    out.append("<eos>"); break
                word_p = SRC.vocab.itos[target]
                out.append(word_p)
                y = torch.tensor([[target]]).to(device)

            print(out)


if __name__ == "__main__":
    lr = 0.0004
    adam_eps = 1e-4
    batch_size = 128
    epochs = 100
    num_layers = 2
    embedding_dim = 256
    hidden_size = 512
    #max_padding_sentence_len = 20
    dropout_p = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iter, dictionary = preprocess(batch_size=batch_size)
    model = CharRNN(len(dictionary.vocab), embedding_dim, hidden_size, num_layers, dropout_p=dropout_p)
    model = model.to(device)
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)
    model = model.half()
    criterion = nn.NLLLoss(ignore_index = dictionary.vocab.stoi['<pad>'])
    criterion = criterion.to(device)
    criterion = criterion.half()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=adam_eps)

    @train(model, criterion, optimizer, train_iter, epochs, device=device, immediate=False, checkpoint=True, verbose=False)
    def lm_train(model, criterion, data):
        # import pdb; pdb.set_trace()
        src = data.text
        batch_size = src.size(1)
        # src, lengths_src = src
        src = src.to(device)
        target = src[1:,:].contiguous()
        pad = torch.tensor([dictionary.vocab.stoi['<pad>'] for _ in range(batch_size)])
        pad = pad.view(1, batch_size).to(device)
        target = torch.cat([target, pad])
        target = target.to(device)
        y_predict, _ = model(src)
        # target = target.transpose(0, 1)
        # pdb.set_trace()
        loss = torch.tensor([0.0]).to(device)
        loss = loss.half()
        for i in range(target.size(0)):
           loss += criterion(y_predict[i], target[i])
        #import pdb; pdb.set_trace()
        return loss

    from nltk.translate.bleu_score import sentence_bleu


    lm_train()
    lm_eval(dictionary, model, criterion, device, 22)
