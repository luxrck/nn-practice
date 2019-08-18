import math

import torch
from torch import nn, optim
import torch.nn.functional as F


def attn_padding_mask(query_len, key_len):
    batch_size = query_len.numel()
    lengths_max = key_len.max().item()
    row_max = query_len.max().item()
    out = []
    for i in range(batch_size):
        a = torch.ones(query_len[i], key_len[i], device=query_len.device).type_as(query_len)
        lengths_pad = lengths_max - key_len[i]
        rows_pad = row_max - query_len[i]
        a = F.pad(a, (0, lengths_pad, 0, rows_pad))
        out.append(a)
    return torch.stack(out)

def attn_subsequence_mask(lengths):
    attn_mask = attn_padding_mask(lengths, lengths)
    for i in range(attn_mask.size(0)):
        a = attn_mask[i]
        for j in range(a.size(0)):
            a[:j+1, j] = 0
    #import pdb; pdb.set_trace()
    return attn_mask


class GeneralAttention(nn.Module):
    def __init__(self):
        super(GeneralAttention, self).__init__()
    def forward(self, Q, K, V, mask=None):
        # mmp! d_k指的是`dimention k`, 不是`D_k(k的方差)`
        d_k = Q.size(-1)
        e = Q.matmul(K.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            masked = -1e-3 if Q.dtype == torch.float16 else -2**30
            e.masked_fill_(mask == 0, -2**15)
        #pdb.set_trace()
        a = F.softmax(e, dim=-1)
        y = a.matmul(V)
        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_attn, d_out):
        super(MultiHeadAttention, self).__init__()
        self.linear_q = nn.Linear(d_input, n_head * d_attn)
        self.linear_k = nn.Linear(d_input, n_head * d_attn)
        self.linear_v = nn.Linear(d_input, n_head * d_attn)
        self.attn = GeneralAttention()
        self.out = nn.Linear(n_head * d_attn, d_out)
    def forward(self, q, k, v, mask=None):
        # import pdb; pdb.set_trace()
        #pdb.set_trace()
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        x = self.attn(q, k, v, mask=mask)
        x = self.out(x)
        return x
