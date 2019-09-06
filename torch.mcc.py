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

import numpy as np

from ash.train import App, Trainer, Checkpoint



class TextCNNMulti(nn.Module):
    def __init__(self, emb, num_embeddings, embedding_dim, cnn_filter_num, max_padding_sentence_len, out_dim, dropout_p):
        super(TextCNN, self).__init__()
        # padding_idx: If given, pads the output with the embedding vector at
        #              padding_idx (initialized to zeros) whenever it encounters the index.
        # 如果pytorch的Embedding没有提供这个padding的功能，第一时间想到的不应该是扩展Embedding使它加上这个功能吗？
        # self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.emb = emb
        self.conv_w2 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 2),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 2 + 1))
        self.conv_w3 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 3),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 3 + 1))
        self.conv_w4 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 4),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 4 + 1))
        self.conv_w5 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 5),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 5 + 1))
        self.conv_w6 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 6),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 6 + 1))
        self.out = nn.Sequential(
                        nn.Linear(cnn_filter_num * 5, 128),
                        nn.Dropout(p=dropout_p),
                        nn.ReLU(),
                        nn.Linear(128, out_dim))

    def forward(self, x):
        # import pdb; pdb.set_trace()
        batch_size = x.size(0)
        x = self.emb(x)
        x = x.permute(0, 2, 1)
        # x = x.view(batch_size, 1, *x.shape[1:])
        x_w2 = self.conv_w2(x)
        x_w3 = self.conv_w3(x)
        x_w4 = self.conv_w4(x)
        x_w5 = self.conv_w5(x)
        x_w6 = self.conv_w6(x)
        x = torch.cat([x_w2, x_w3, x_w4, x_w5, x_w6], dim=1)
        x = x.view(batch_size, -1)
        x = self.out(x)
        # 我用的是CrossEntropyLoss, 所以这里不需要用softmax
        # x = F.softmax(x)
        return x


def preprocess(**kwargs):
    SRC = Field(include_lengths=False,
                init_token=None,
                pad_token="<pad>",
                unk_token="<unk>",
                lower=True,
                batch_first=False,
                tokenize=lambda text: list(text.strip()))
    _train, _test = TabularDataset.splits(path="data", root="data", train="train.weibo.txt", test="test.weibo.txt", format='tsv', skip_header=False, fields=[("text", SRC), ("label", SRC), ("target", SRC)], csv_reader_params={"quoting": csv.QUOTE_NONE})
    SRC.build_vocab(_train, min_freq=5)
    train_iter = BucketIterator(_train, batch_size=kwargs["batch_size"], train=True,
                                 sort_within_batch=True,
                                 sort_key=lambda x: (len(x.text)), repeat=False,
                                 device=device)
    test_iter = BucketIterator(_test, batch_size=1, train=False, device=device)
    return train_iter, test_iter, SRC



if __name__ == "__main__":
    batch_size = 128
    num_layers = 2
    embedding_dim = 256
    hidden_size = 512
    dropout_p = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iter, test_iter, dictionary = preprocess(batch_size=batch_size)
    model = TextCNNMulti(len(dictionary.vocab), embedding_dim, hidden_size, num_layers, dropout_p=dropout_p)
    #def init_weights(m):
    #    for name, param in m.named_parameters():
    #        nn.init.uniform_(param.data, -0.08, 0.08)
    #model.apply(init_weights)

    app = Trainer(App(model=model,
                      criterion=nn.NLLLoss(ignore_index=dictionary.vocab.stoi['<pad>'])))

    @app.on("train")
    def lm_train(e):
        e.model.zero_grad()
        src = e.batch.text
        batch_size = src.size(1)
        target = src[1:,:].contiguous()
        pad = torch.tensor([dictionary.vocab.stoi['<pad>'] for _ in range(batch_size)])
        pad = pad.view(1, batch_size).to(target.device)
        target = torch.cat([target, pad])
        y_predict, _ = model(src)
        loss = 0.0
        #pdb.set_trace()
        for i in range(target.size(0)):
            loss += e.criterion(y_predict[i], target[i])
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(e.model.parameters(), max_norm=2)
        e.optimizer.step()
        e.a.loss = loss
        #print(loss.item())

    @app.on("iter_completed")
    def logging(e):
        if e.current_iter % 200 == 0:
            #pdb.set_trace()
            print(e.a.loss)

    app.fastforward()   \
       .save_every(iters=2000)  \
       .set_optimizer(optim.Adam, lr=0.0004, eps=1e-4)  \
       .to("auto")  \
       .half()  \
       .run(train_iter, max_iters=20000, train=True)

    lm_eval(dictionary, app)
