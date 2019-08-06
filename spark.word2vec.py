#!/bin/env python3

import random
import itertools
from collections import OrderedDict, defaultdict
import numpy as np

from sparkle import softmax

#random.seed(2143)
#np.random.seed(1234)

def cosine(u, v):
    return np.dot(u,v)/(np.linalg.norm(u)*(np.linalg.norm(v)))


class SkipGram(object):
    def __init__(self, corpus, embedding_dims, context_size):
        vocab = sorted(set(corpus))
        vocab_size = len(vocab)
        vocab = {word: i for i, word in enumerate(vocab)}
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.corpus = corpus
        self.V = np.random.randn(embedding_dims, vocab_size)
        self.U = np.random.randn(vocab_size, embedding_dims)

    def step(self, centerWord, context):
        loss = 0.
        gradCenterWordVec = 0.
        gradContextWordVecs = 0.

        centerWordIdx = self.vocab[centerWord]
        #x = np.zeros(self.vocab_size).reshape(-1, 1)
        #vc = V * x . (x is one-hot vector for current center word)
        centerWordVec = self.V[:, centerWordIdx].reshape(-1, 1)

        for contextWord in context:
            contextWordIdx = self.vocab[contextWord]

            z = self.U.dot(centerWordVec)
            y_predict = softmax(z)

            # Loss function J = -log(P(O=o|C=c))
            loss += (-np.log(y_predict[contextWordIdx]))

            # y^ - y
            y_predict[contextWordIdx] -= 1

            # dJ/dvc = U^T * (y^ - y) . (vc: current center word vector, a column in self.V)
            gradCenterWordVec += self.U.T.dot(y_predict).reshape(-1, 1)

            # dJ/duc = (y^ - y) * vc . (uc: current context word vector, a row in self.U)
            gradContextWordVecs += y_predict.dot(centerWordVec.T)

        return loss, gradCenterWordVec, gradContextWordVecs


    def train(self, lr, dataset, epochs):
        r"""SGD optimizing method"""
        for e in range(epochs):
            loss = 0.
            for i, (center, context) in enumerate(dataset):
                _loss, gradCenterWordVec, gradContextWordVecs = self.step(center, context)

                centerWordIdx = self.vocab[center]

                self.V[:, centerWordIdx] -= lr * gradCenterWordVec[:, 0]
                self.U -= lr * gradContextWordVecs

                loss += _loss
                print(e, i, loss)#, center, context)
        return self.V, self.U

corpus = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
corpus = open("stanfordSentimentTreebank/SOStr.txt").read().split("|")
trainloader = []
hw = 4
for i in range(4, len(corpus) - 4):
    center = corpus[i]
    context = [corpus[i-4], corpus[i-3], corpus[i-2], corpus[i-1], corpus[i+1], corpus[i+2], corpus[i+3], corpus[i+4]]
    if i % 1000 == 0: print(i)
    trainloader.append((center, context))

word2vec = SkipGram(corpus, 64, 8)
V, U = word2vec.train(lr=0.03, dataset=trainloader, epochs=500)
import pdb; pdb.set_trace()
print(cosine(U[word2vec.vocab["men"]], U[word2vec.vocab["women"]]))