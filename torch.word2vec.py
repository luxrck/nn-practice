# !/usr/bin/python

import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms

import numpy as np


from sparkle.utils import train, test


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self._vocab_size = vocab_size
        self.em = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, embedding_dim, bias=False)
        self.fc2 = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, inputs):
        #import pdb; pdb.set_trace()
        embeds = self.em(inputs).view((inputs.size(1), -1))
        embeds = self.fc1(embeds)
        embeds = F.relu(embeds)
        embeds = self.fc2(embeds)
        embeds = F.log_softmax(embeds, dim=1)
        return embeds



class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(SkipGram, self).__init__()
        self._vocab_size = vocab_size
        self.em = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.fc2 = nn.Linear(embedding_dim, context_size * vocab_size, bias=False)

    def forward(self, inputs):
        x = self.onehot(inputs)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = -F.log_softmax(x, dim=1)
        return x

    def onehot(self, idx):
        o = [0 for _ in range(self._vocab_size)]; o[idx] = 1
        o = torch.tensor(o, dtype=torch.long)
        return o



class NGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs



def prepare():
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    # By deriving a set from `raw_text`, we deduplicate the array
    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []
    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1],
                raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))

    return vocab, data, word_to_ix



if __name__ == "__main__":
    vocab, dataset, word_to_ix = prepare()

    vocab_size = len(vocab)
    embedding_dim = 100
    context_size = 4
    batch_size = 1
    epochs = 100

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model = CBOW(vocab_size, embedding_dim, context_size)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    @train(model, loss_function, optimizer, trainloader, epochs)
    def cbow_train(model, criterion, inputs, labels):
        #import pdb; pdb.set_trace()
        context_ids = [[word_to_ix[w] for w in context] for context in inputs]
        context_ids = torch.tensor(context_ids)
        labels = [word_to_ix[l] for l in labels]
        labels = torch.tensor(labels)
        y_predicts = model(context_ids)
        loss = criterion(y_predicts, labels)
        return loss

    @test(model, loss_function, optimizer, testloader)
    def cbow_test(model, criterion, inputs, labels):
        context_ids = [[word_to_ix[w] for w in context] for context in inputs]
        context_ids = torch.tensor(context_ids)
        labels = [word_to_ix[l] for l in labels]
        labels = torch.tensor(labels)
        y_predicts = model(context_ids)
        #import pdb; pdb.set_trace()
        #loss = criterion(y_predicts, labels)
        _, y_predicts = torch.max(y_predicts.data, dim=1)
        return (y_predicts == labels).sum()

    cbow_train()
    cbow_test()