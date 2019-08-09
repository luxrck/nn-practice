def accuracy(predicts, targets):
    count = 0
    for p,t in zip(predicts, targets):
        _, topi = p.topk(1)
        topi = topi.squeeze()
        count += (topi == t).sum()
    return float(count.item()) / len(predicts)


def bleu():
    pass


def f1_score():
    pass


def precision():
    pass
