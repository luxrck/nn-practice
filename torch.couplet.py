import pdb
import random
import re
import csv
import sys
import os
from functools import reduce

from tqdm import tqdm

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import torch
from torch import nn, optim
import torch.nn.functional as F

from torchtext.data import Field, TabularDataset, BucketIterator

import numpy as np
import spacy

from ash.models import Transformer
from ash.train import App, Trainer, Checkpoint


def preprocess_couplet():
    SRC = Field(include_lengths=True,
                init_token="<sos>",
                eos_token="<eos>",
                pad_token="<pad>",
                unk_token="<unk>",
                lower=True,
                batch_first=False,
                tokenize=lambda text: text.split())
    TRG = Field(include_lengths=True,
                init_token="<sos>",
                eos_token="<eos>",
                pad_token="<pad>",
                unk_token="<unk>",
                lower=True,
                batch_first=False,
                tokenize=lambda text: text.split())
    _train, _test = TabularDataset.splits(path="data/couplet", root="data", train="train.tsv", test="test.tsv",
                                    format='csv', skip_header=False, fields=[("src", SRC), ("trg", TRG)],
                                    csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})
    SRC.build_vocab(_train.src, _train.trg, min_freq=1)
    TRG.vocab = SRC.vocab
    return _train, _test, SRC, TRG


if __name__ == "__main__":

    batch_size = 128
    d_model = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    _train, _test, SRC, TRG = preprocess_couplet()
    train_iter = BucketIterator(_train, batch_size=batch_size, train=True,
                                 repeat=False, shuffle=True,
                                 #sort_within_batch=True,
                                 #sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                 device=device)
    test_iter = BucketIterator(_test, batch_size=1, train=False, repeat=False,
            sort_within_batch=False, sort_key=lambda x: (len(x.src), len(x.trg)), device=device)
    #import pdb; pdb.set_trace()

    #model = Seq2Seq(SRC, TRG, src_vocab_size=len(SRC.vocab), trg_vocab_size=len(TRG.vocab), embedding_dim=256, hidden_size=512, dropout_p=0.5, attention=True)
    model_t = Transformer(len(SRC.vocab), len(TRG.vocab), d_model=d_model, n_encoder_layers=6, n_decoder_layers=6, dropout_p=0.1)

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

        y_predict = e.model(src, targets, lengths_src, lengths_trg)
        #if e.current_iter % 50 == 0:
        #    import pdb; pdb.set_trace()
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
        # %% 梯度剪裁对模型训练有这么大的影响？
        torch.nn.utils.clip_grad_norm_(e.model.parameters(), max_norm=2)
        e.optimizer.step()
        e.a.loss = loss.item()

    @app.on("iter_completed")
    def pront_loss(e):
        e.progress.set_postfix(loss=e.a.loss)
    
    #@app.on("iter_completed")
    def adjust_learning_rate(e):
        step_num = e.current_iter
        if step_num % 200: return
        warmup_steps = 4000
        lr = d_model ** (-0.5) * min([step_num ** (-0.5), step_num * warmup_steps ** (-1.5)])
        for param_group in e.optimizer.param_groups:
            param_group['lr'] = lr


    # test_iter的batch_size=1
    @app.on("evaluate")
    def nmt_eval(e):
        (src, lengths_src), (targets, lengths_trg) = e.batch.src, e.batch.trg
        #import pdb; pdb.set_trace()
        # set `teacher_forcing_p < 0` to indicate that we are in `evaluate` mode.
        seq_len = lengths_src.max()
        decoded = [TRG.vocab.stoi['<sos>']]
        for _ in range(seq_len-1):
            trg = torch.tensor([decoded]).to(src.device)
            trg = trg.view(-1, 1)
            lengths_trg = torch.tensor([trg.size(0)]).to(src.device)
            #import pdb; pdb.set_trace()
            y_predict = e.model(src, trg, lengths_src, lengths_trg)
            #import pdb; pdb.set_trace()
            #y_predict = y_predict.topk(3)[1].view(-1, 3).tolist()
            y_predict = y_predict.topk(3)[1].squeeze(dim=0)[-1].tolist()
            wo = y_predict[0]
            if wo == TRG.vocab.stoi['<eos>']:
                wo = y_predict[1]
            decoded.append(wo)
        decoded[-1] = TRG.vocab.stoi['<eos>']
        decode_output = [TRG.vocab.itos[i] for i in decoded]
        src = src.view(-1).tolist()
        src = [SRC.vocab.itos[i] for i in src]
        targets = targets.view(-1).tolist()
        targets = [TRG.vocab.itos[i] for i in targets]
        print("S:", " ".join(src))
        print("T:", " ".join(targets))
        print("D:", " ".join(decode_output))
        return decode_output, targets


    app.fastforward()   \
       .set_optimizer(optim.Adam, lr=0.0001, eps=1e-4, betas=(0.9, 0.98))  \
       .to("auto")  \
       .half()  \
       .save_every(iters=1000)  \
       .run(train_iter, max_iters=1000, train=False)   \
       .eval(test_iter)
