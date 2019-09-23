import csv
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import pandas as pd

from ash.train import App, Trainer, Checkpoint

from torchtext.data import Field, TabularDataset, BucketIterator
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LinearNet(nn.Module):
    def __init__(self, d_input, d_out, dropout_p=0.1):
        super().__init__()
        self.nn = nn.Sequential(
                                nn.Linear(d_input, 2048),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_p),
                                nn.Linear(2048, d_out),
                                nn.Softmax(dim=-1))

    # (batch, d_input)
    def forward(self, inputs):
        y = self.nn(inputs)
        return y


class LSTMNet(nn.Module):
    def __init__(self, vocab, dropout_p=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=dropout_p, bidirectional=True)
        self.out = LinearNet(d_input=2*2*256, d_out=2)

    # (seq, batch, d_emb)
    def forward(self, inputs):
        x = self.emb(inputs)
        _,(h, _) = self.lstm(x)
        # import pdb; pdb.set_trace()
        h = h.transpose(0,1).contiguous().view(x.size(1), -1)
        x = self.out(h)
        return x




class SentenceJudgementDataset(Dataset):
        def __init__(self, csv_file):
                self.frame = pd.read_csv(csv_file, delimiter="\t", header=None)

        def __len__(self):
                return len(self.frame)

        def __getitem__(self, idx):
                if torch.is_tensor(idx):
                        idx = idx.tolist()
                # import pdb; pdb.set_trace()
                input = self.frame.iloc[idx, 2:].tolist()
                label = self.frame.iloc[idx, 0].tolist()
                return {"src": torch.tensor(input), "trg": label}


def preprocess_sent_score():
    SRC = Field(include_lengths=True,
                init_token=None,
                eos_token=None,
                pad_token="<pad>",
                unk_token="<unk>",
                lower=True,
                batch_first=False,
                tokenize=lambda text: list(text))
    TRG = Field(include_lengths=True,
                                use_vocab=False,
                init_token=None,
                eos_token=None,
                pad_token=None,
                unk_token=None,
                # lower=True,
                batch_first=False,
                tokenize=lambda text: [int(i) for i in text])
    _train, _test = TabularDataset.splits(path="data/sent_score", root="data", train="train.tsv", test="test.tsv",
                                    format='csv', skip_header=True, fields=[("trg", TRG), ("src", SRC)],
                                    csv_reader_params={"quoting": csv.QUOTE_NONE, "delimiter": "\t"})
    SRC.build_vocab(_train.src, min_freq=1)
    # TRG.build_vocab(_train.trg, min_freq=1)
    return _train, _test, SRC, TRG



def predict(inputs, labels, w, t):
    predicted = []
    gold = []
    count = 0
    for i in range(len(labels)):
        # import pdb; pdb.set_trace()
        s = inputs.iloc[i].tolist()
        l = labels.iloc[i].tolist()
        p = np.dot(s, w)
        p = sigmoid(p)
        if p >= t:
            p = 1
        else:
            p = 0
        if p == l:
            count += 1
        predicted.append(p)
        gold.append(l)
    return count #, precision_recall_fscore_support(gold, predicted)


if __name__ == "__main__":
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train, _test, SRC, TRG = preprocess_sent_score()
    train_iter = BucketIterator(_train, batch_size=batch_size, train=True,
                                 repeat=False, shuffle=True,
                                 sort_within_batch=True,
                                 sort_key=lambda x: (len(x.src), len(x.trg)),
                                 device=device)
    test_iter = BucketIterator(_test, batch_size=1, train=False, repeat=False, sort_within_batch=False, sort_key=lambda x: len(x.src), device=device)
    model = LinearNet(d_input=4, d_out=1, dropout_p=0.2)
    model = LSTMNet(len(SRC.vocab))

    criterion = nn.CrossEntropyLoss(reduction="sum")
    #criterion = nn.MSELoss()
    app = Trainer(App(model=model, criterion=criterion))
    app.extend(Checkpoint())

    @app.on("train")
    def sent_train(e):
        e.model.zero_grad()
        # import pdb; pdb.set_trace()
        #src, trg = e.batch['src'], e.batch['trg']
        (src, lengths_src), (trg, lengths_trg) = e.batch.src, e.batch.trg
        src = src.to(device)
        trg = trg.to(device)
        y_predict = e.model(src)
        #trg = F.one_hot(trg).to(device).type(torch.float32)
        trg = trg.transpose(0, 1).view(-1)
        loss = e.criterion(y_predict, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(e.model.parameters(), max_norm=2)
        e.optimizer.step()
        e.a.loss = loss.item()

    @app.on("iter_completed")
    def pront_loss(e):
            e.progress.set_postfix(loss=e.a.loss)

    @app.on("iter_completed")
    def validation(e):
        if e.current_iter % 20 == 0:
            p,g = app.eval(test_iter)
            print((p == g).sum())
            e.model.train()

    # test_iter的batch_size=1
    tt = 0.5
    @app.on("evaluate")
    def sent_eval(e):
        (src, lengths_src), (trg, lengths_trg) = e.batch.src, e.batch.trg
        src = src.to(device)
        trg = trg.to(device)
        y_predict = e.model(src)
        y_prob, y_predict_ = y_predict.topk(1)
        #print(y_predict, y_prob, y_predict_)
        #if 2 * y_prob -1 < tt:
        #    y_predict[0,0] = 1 - y_predict[0,0]
        # import pdb; pdb.set_trace()
        return y_predict_, trg.view(1, -1)

         #.set_optimizer(optim.SGD, lr=0.001)    \
    app.fastforward()   \
         .set_optimizer(optim.Adam, lr=0.0001, eps=1e-6, betas=(0.9, 0.98))  \
         .to("auto")  \
         .save_every(iters=10000)  \
         .run(train_iter, max_iters=8000, train=True)
    predicted, gold = app.eval(test_iter)
    print((predicted == gold).sum())
    print(predicted.view(-1).tolist())
    print(gold.view(-1).tolist())
    print(precision_recall_fscore_support(gold.view(-1).tolist(), predicted.view(-1).tolist()))

    # _train = SentenceJudgementDataset("data/sent_score/train.tsv")
    # _test = SentenceJudgementDataset("data/sent_score/test.tsv")
    # inputs = _train.frame.iloc[:,2:]
    # labels = _train.frame.iloc[:, 0]

    # results = []
    # from tqdm import tqdm
    # tq = tqdm(total = 1000000)
    # t = 0.2
    # while t < 1.:
    #     w1 = 0.05
    #     for _ in range(10):
    #         w2 = 0.05
    #         for _ in range(10):
    #             w3 = 0.05
    #             for _ in range(10):
    #                 w4 = 0.05
    #                 for _ in range(10):
    #                     c = predict(inputs, labels, [w1,w2,w3,w4], t)
    #                     results.append([c, [w1,w2,w3,w4], t])
    #                     # print(c,t, w1, w2, w3,w4)
    #                     tq.update(1)
    #                     w4 += 0.1
    #                 w3 += 0.1
    #             w2 += 0.1
    #         w1 += 0.1
    #     t += 0.05
    # import pickle
    # pickle.dump(results, open("out.pkl", "wb+"))

    # oo = pickle.load(open("out.pkl", "rb+"))
    # oo = list(sorted(oo, key=lambda x: x[0]))
    # # print(oo[-32:])

    # for i,w,t in oo[-64:]:
    #     r = predict(inputs, labels, w, t)
    #     print(r, i,w,t)
