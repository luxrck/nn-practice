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
from torchnlp.metrics import bleu

import numpy as np
import spacy

from sparkle.utils import train, test
from sparkle.train import App, Trainer, Checkpoint


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
    def forward(self, src, trg, lengths_src, lengths_trg, teacher_forcing_p=0.8):
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
        di = trg[0].view(1, *trg[0].size())
        for i in range(trg.size(0)):
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

            output.append(y.unsqueeze(1))

        return torch.cat(output, dim=1)

    def evaluate(self, src, max_len=100):
        import pdb
        #pdb.set_trace()
        src = self.src_emb(src)
        oe, (h, c) = self.encoder(src)
        oe = oe.transpose(0,1)

        output = []

        di = torch.tensor([[2]]).cuda()
        di = self.trg_emb(di)

        for i in range(max_len):
            od, (h, c) = self.decoder(di, (h, c))

            x = h.permute(1,0,2).contiguous()
            # x: (batch, num_directions * hidden_size)
            x = x.view(x.size(0), -1)

            if self.attention:
                # Soft-Attention [Bahdanau et al. 2015]
                e = od.permute(1,0,2).matmul(oe.transpose(-1,-2))
                a = F.softmax(e, dim=2)
                w = a.matmul(oe).transpose(-1, -2)
                #pdb.set_trace()

                #if self.attention:
                x = torch.cat([x, w.view(w.size(0), -1)], dim=1)
            y = self.out(x)
            # x = h.view(-1, 4*2*512)

            _, topi = y.topk(1)
            if topi.item() == 3:
                output.append("<eos>")
                break
            else:
                di = self.trg_emb(topi).permute(1,0,2)
                output.append(self.trg.vocab.itos[topi.item()])
        #import pdb; pdb.set_trace()
        return output


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

    train_iter = BucketIterator(_train, batch_size=batch_size, train=True,
                                 sort_within_batch=True,
                                 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                 device=device)
    test_iter = BucketIterator(_test, batch_size=1, train=False, repeat=False, device=device)

    model = Seq2Seq(SRC, TRG, src_vocab_size=len(SRC.vocab), trg_vocab_size=len(TRG.vocab), embedding_dim=embedding_dim, hidden_size=hidden_size, dropout_p=dropout_p, attention=False)
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)

    app = Checkpoint(Trainer(App(model=Seq2Seq(SRC, TRG, src_vocab_size=len(SRC.vocab), trg_vocab_size=len(TRG.vocab),
                                               embedding_dim=256, hidden_size=512, dropout_p=0.5, attention=False),
                                 criterion=nn.NLLLoss(ignore_index=TRG.vocab.stoi['<pad>']))))
    @app.on("iter_started")
    def nmt_train(e):
        model = e.model
        criterion = e.criterion
        optimizer = e.optimizer
        batch = e.batch
        src, targets = batch.src, batch.trg
        src, lengths_src = src
        targets, lengths_trg = targets
        model.zero_grad()
        y_predict = model(src, targets, lengths_src, lengths_trg, teacher_forcing_p=0.9)
        targets = targets.transpose(0, 1)
        loss = torch.tensor([0.0]).type(y_predict.dtype)
        for i in range(target.size(0)):
           loss += criterion(y_predict[i], targets[i])
        optimizer.step()
        loss.backward()

    app.fastforward()   \
       .save_every(iters=1000)  \
       .set_optimizer(optim.Adam, lr=0.0004, eps=1e-4)  \
       .to("auto")  \
       .half()  \
       .run(train_iter, max_iters=20000)

    @app.on("train_completed")
    def nmt_eval(e):
        trainer, data = e.trainer, e.data
        trainer.eval(data, metric=bleu)

    app.run(test_iter, train=False)


    # @train(model, criterion, optimizer, train_iter, epochs, device=device, immediate=False, checkpoint=True, verbose=False)
    # def nmt_train(model, criterion, data):
    #     # import pdb; pdb.set_trace()
    #     # inputs = torch.cat([i.view(1, *i.size()).to(device) for i in inputs], dim=0)
    #     src = data.src
    #     src, lengths_src = src
    #     src = src.to(device)
    #     target = data.trg
    #     target, lengths_trg = target
    #     target = target.to(device)
    #     y_predict = model(src, target, lengths_src, lengths_trg, teacher_forcing_p=0.9)
    #     target = target.transpose(0, 1)
    #     # loss = criterion(y_predict, target)
    #     #import pdb; pdb.set_trace()
    #     loss = torch.tensor([0.0]).cuda().half()
    #     for i in range(target.size(0)):
    #        loss += criterion(y_predict[i], target[i])
    #     #import pdb; pdb.set_trace()
    #     return y_predict, loss

    #@train(model, criterion, optimizer, train_iter, epochs, device=device, immediate=True, checkpoint=True, verbose=False)
    from nltk.translate.bleu_score import sentence_bleu

    def nmt_eval(model, criterion):
        state = torch.load(f"checkpoint/Seq2Seq.{6}.torch")
        model.load_state_dict(state["model"])
        model = model.to(device)
        criterion = criterion.to(device)
        optimizer.load_state_dict(state["optim"])

        score = 0.0

        with torch.no_grad():
            for i,(data) in tqdm(enumerate(test_iter)):
                src, target = (data.src, data.trg)
                src, lengths_src = src
                src = src.to(device)
                target, lengths_trg = target
                target = target.to(device)
                decode_output = model.evaluate(src, max_len=100)
                #src = list(src.view(-1))
                #src = [SRC.vocab.itos[i] for i in src]
                target = list(target.view(-1))
                target = [TRG.vocab.itos[i] for i in target]
                sc = sentence_bleu([target[1:-1]], decode_output[1:-1])
                print("T:", " ".join(target))
                print("D:", " ".join(decode_output))
                #print(sc)
                score += sc
                #import pdb; pdb.set_trace()
        print("BLEU: ", score / 1000)
    nmt_train()
    nmt_eval(model, criterion)
