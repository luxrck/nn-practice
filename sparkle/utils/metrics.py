def accuracy(predicts, targets):
    count = 0
    for p,t in zip(predicts, targets):
        count += (p == t).sum()
    return float(count.item()) / len(predicts)


def bleu():
    pass


def f1_score():
    pass


def precision():
    pass