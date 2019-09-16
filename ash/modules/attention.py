import math

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


def attn_padding_mask(query_len, key_len):
    batch_size = query_len.numel()
    k_len_max = key_len.max().item()
    q_len_max = query_len.max().item()
    out = []
    for i in range(batch_size):
        a = torch.zeros(query_len[i], key_len[i], dtype=torch.bool, device=query_len.device)
        k_pad = k_len_max - key_len[i]
        q_pad = q_len_max - query_len[i]
        a = F.pad(a, (0, k_pad, 0, q_pad), value=True)
        out.append(a)
    return torch.stack(out)

def attn_subsequent_mask(lengths):
    q_len_max = lengths.max().item()
    padding_mask = attn_padding_mask(lengths, lengths)
    subsequent_mask = torch.triu(torch.ones(q_len_max, q_len_max, dtype=torch.uint8), diagonal=1).type(torch.bool).to(padding_mask.device)
    #for i in range(attn_mask.size(0)):
    #    a = attn_mask[i]
    #    for j in range(a.size(0)):
    #        a[j, j+1:] = 1.
    ##import pdb; pdb.set_trace()
    return padding_mask + subsequent_mask


class GeneralAttention(nn.Module):
    def __init__(self, dropout_p=0.1):
        super(GeneralAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
    def forward(self, Q, K, V, scale=1, mask=None):
        # mmp! d_k指的是`dimention k`, 不是`D_k(k的方差)`
        #import pdb; pdb.set_trace()
        assert scale > 0
        e = Q.bmm(K.transpose(-1, -2)) / scale
        if mask is not None:
            masked = -2**14 if Q.dtype == torch.float16 else -2**31
            e.masked_fill_(mask, masked)
            #mask = mask.type(e.dtype).to(e.device)
            #import pdb; pdb.set_trace()
            #e += mask
        #pdb.set_trace()
        #import pdb; pdb.set_trace()
        a = F.softmax(e, dim=-1)
        a = self.dropout(a)
        y = a.bmm(V)
        #import pdb; pdb.set_trace()
        return y


class MultiHeadAttention(nn.Module):
    # d_model: 输入和输出的向量维度
    def __init__(self, n_head, d_model, d_k, d_v, dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.linear_q = nn.Linear(d_k, n_head * d_k)    # d_q == d_k
        self.linear_k = nn.Linear(d_k, n_head * d_k)
        self.linear_v = nn.Linear(d_v, n_head * d_v)
        self.attn = GeneralAttention(dropout_p=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        #self.out = nn.Linear(n_head * d_v, d_model)
        self.out = nn.Sequential(
                    nn.Linear(n_head * d_v, d_model),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p),
                    nn.Linear(d_model, d_model))
        
        #nn.init.normal_(self.linear_q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.linear_k.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.linear_v.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
    
    # (b, s, d)
    def forward(self, q, k, v, scale=None, mask=None):
        if scale is not None:
            assert scale != 0

        bs = q.size(0)
        n_head = self.n_head
        
        s_q = q.size(1)
        s_k = k.size(1)
        s_v = v.size(1)

        d_q = q.size(2)
        d_k = k.size(2)
        d_v = v.size(2)

        q = self.linear_q(q).view(bs, s_q, n_head, d_q).contiguous().permute(2, 0, 1, 3).contiguous().view(-1, s_q, d_q)
        k = self.linear_q(k).view(bs, s_k, n_head, d_k).contiguous().permute(2, 0, 1, 3).contiguous().view(-1, s_k, d_k)
        v = self.linear_q(v).view(bs, s_v, n_head, d_v).contiguous().permute(2, 0, 1, 3).contiguous().view(-1, s_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        if not scale:
            scale = math.sqrt(d_k)
        
        x = self.attn(q, k, v, scale=scale, mask=mask)
        x = x.view(n_head, bs, s_q, d_v).contiguous().permute(1, 2, 0, 3).contiguous().view(bs, s_q, -1)
        x = self.dropout(x)
        x = self.out(x)
        return x
