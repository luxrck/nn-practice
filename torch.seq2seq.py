import pdb
import random
import re
import sys
import os
from functools import reduce

from tqdm import tqdm

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

import numpy as np
import spacy

from ash.models import Transformer
from ash.train import App, Trainer, Checkpoint


class Seq2Seq(nn.Module):
    def __init__(self, src, trg, src_vocab_size, trg_vocab_size, embedding_dim, hidden_size, dropout_p, teacher_forcing_p=0.6, attention=False):
        super(Seq2Seq, self).__init__()
        self.src = src
        self.trg = trg
        self.teacher_forcing_p = teacher_forcing_p
        self.attention = attention
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb = nn.Embedding(src_vocab_size, embedding_dim)
        self.trg_emb = nn.Embedding(trg_vocab_size, embedding_dim)

        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=False, bidirectional=True, num_layers=4)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=False, bidirectional=True, num_layers=4)

        out_in_dim = 4*2*hidden_size
        if self.attention:
            out_in_dim += 2*hidden_size
        self.out = nn.Sequential(
                        nn.Linear(out_in_dim, out_in_dim),
                        nn.Dropout(p=dropout_p),
                        nn.ReLU(),
                        nn.Linear(out_in_dim, trg_vocab_size),
                        nn.LogSoftmax(dim=1))

    # (seq_len, batch, embedding_dim)
    def forward(self, src, trg, lengths_src, lengths_trg, max_len=128, teacher_forcing_p=0.8):
        is_training = teacher_forcing_p > 0

        src = self.src_emb(src)
        trg = self.trg_emb(trg)

        lengths_src = lengths_src.view(-1).tolist()
        packed_src = pack(src, lengths_src)
        packed_oe, (h, c) = self.encoder(packed_src)
        oe, lengths_src = unpack(packed_oe)
        # oe: (batch, seq_len, num_directions * hidden_size)
        oe = oe.transpose(0,1)

        output = []

        lengths_trg = lengths_trg.view(-1)

        #pdb.set_trace()
        di = torch.tensor([self.src.vocab.stoi["<sos>"]]).to(src.device)
        di = self.trg_emb(di)
        if is_training:
            di = trg[0]
            max_len = trg.size(0)
        di = di.view(1, *di.size())

        for i in range(max_len):
            od, (h, c) = self.decoder(di, (h, c))

            x = h.permute(1,0,2).contiguous()
            # x: (batch, num_directions * hidden_size)
            x = x.view(x.size(0), -1)

            # Soft-Attention [Bahdanau et al. 2015]
            #pdb.set_trace()
            if self.attention:
                e = od.permute(1,0,2).matmul(oe.transpose(-1,-2))
                a = F.softmax(e, dim=2)
                #pdb.set_trace()
                w = a.matmul(oe).transpose(-1, -2)

                #if self.attention:
                x = torch.cat([x, w.view(w.size(0), -1)], dim=1)
            y = self.out(x)

            if random.random() < teacher_forcing_p:
                di = trg[i].view(1, *trg[i].size())
            else:
                _, topi = y.topk(1)
                di = self.trg_emb(topi).permute(1,0,2)

                if is_training == False:
                    if topi.item() == 3:    # <eos>
                        output.append(y.unsqueeze(1))
                        break

            output.append(y.unsqueeze(1))

        return torch.cat(output, dim=1)



if __name__ == "__main__":
    spacy_trg = spacy.load("en")
    spacy_src = spacy.load("de")
    TRG = Field(include_lengths=True,
                init_token="<sos>",
                eos_token="<eos>",
                pad_token="<pad>",
                unk_token="<unk>",
                lower=True,
                batch_first=False,
                tokenize=lambda text: [t.text for t in spacy_trg.tokenizer(text)])
    SRC = Field(include_lengths=True,
                init_token="<sos>",
                eos_token="<eos>",
                pad_token="<pad>",
                unk_token="<unk>",
                lower=True,
                batch_first=False,
                tokenize=lambda text: [t.text for t in spacy_src.tokenizer(text)])
    _train, _vali, _test = Multi30k.splits(root="data", exts=(".de", ".en"), fields=(SRC, TRG))
    # It is important to note that your vocabulary should only be built from the training set
    # and not the validation/test set. This prevents "information leakage" into your model,
    # giving you artifically inflated validation/test scores.
    SRC.build_vocab(_train, min_freq=2)
    TRG.build_vocab(_train, min_freq=2)

    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iter = BucketIterator(_train, batch_size=batch_size, train=True,
                                 sort_within_batch=True,
                                 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                 device=device)
    test_iter = BucketIterator(_test, batch_size=1, train=False, repeat=False, device=device)

    #model = Seq2Seq(SRC, TRG, src_vocab_size=len(SRC.vocab), trg_vocab_size=len(TRG.vocab), embedding_dim=256, hidden_size=512, dropout_p=0.5, attention=True)
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model_t = Transformer(len(SRC.vocab), len(TRG.vocab), d_model=256, n_encoder_layers=3, n_decoder_layers=1, dropout_p=0.1)
    #model_t.apply(init_weights)

    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'], reduction="sum")
    app = Trainer(App(model=model_t,
                      criterion=criterion))
    app.extend(Checkpoint())

    @app.on("train")
    def nmt_train(e):
        e.model.zero_grad()
        (src, lengths_src), (targets, lengths_trg) = e.batch.src, e.batch.trg
        
        # (seq, batch)
        #import pdb; pdb.set_trace()
        #targets, gold = targets[:-1,:], targets[1:,:]
        #lengths_trg -= 1

        #import pdb; pdb.set_trace()
        y_predict = e.model(src, targets, lengths_src, lengths_trg)
        #targets = targets.transpose(0, 1)

        #y_predict = y_predict[:,:-1,:].contiguous().view(-1, y_predict.size(-1))
        #gold = targets[1:,:].view(-1)
        y_predict = y_predict[:,:-1,:].contiguous().view(-1, y_predict.size(-1))
        gold = targets[1:,:].contiguous().transpose(0, 1).contiguous().view(-1)

        #loss = 0.0
        #for i in range(targets.size(0)):
        #    loss += e.criterion(y_predict[i,:-1,:], targets[i,1:])

        loss = e.criterion(y_predict, gold)
        loss.backward()
        e.optimizer.step()
        print(loss.item())

    # test_iter的batch_size=1
    @app.on("evaluate")
    def nmt_eval(e):
        (src, lengths_src), (targets, lengths_trg) = e.batch.src, e.batch.trg
        #import pdb; pdb.set_trace()
        # set `teacher_forcing_p < 0` to indicate that we are in `evaluate` mode.
        decoded = [TRG.vocab.stoi['<sos>']]
        for _ in range(128):
            trg = torch.tensor([decoded]).to(src.device)
            trg = trg.view(-1, 1)
            lengths_trg = torch.tensor([trg.size(0)]).to(src.device)
            #import pdb; pdb.set_trace()
            y_predict = e.model(src, trg, lengths_src, lengths_trg)
            #import pdb; pdb.set_trace()
            y_predict = y_predict.topk(3)[1].view(-1, 3).tolist()
            wo = y_predict[-1][1]
            decoded.append(wo)
            if decoded[-1] == TRG.vocab.stoi['<eos>']:
                break
        decode_output = [TRG.vocab.itos[i] for i in decoded]
        targets = list(targets.view(-1))
        targets = [TRG.vocab.itos[i] for i in targets]
        print("T:", " ".join(targets))
        print("D:", " ".join(decode_output))
        return decode_output, targets

    app.fastforward()   \
       .set_optimizer(optim.Adam, lr=0.0004, eps=1e-6)  \
       .to("auto")  \
       .save_every(iters=2000)  \
       .run(train_iter, max_iters=10000, train=True)   \
       .eval(test_iter)
