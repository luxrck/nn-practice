import math

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


def attn_padding_mask(query_len, key_len):
    batch_size = query_len.numel()
    lengths_max = key_len.max().item()
    row_max = query_len.max().item()
    out = []
    for i in range(batch_size):
        a = torch.zeros(query_len[i], key_len[i], dtype=torch.float32, device=query_len.device)
        lengths_pad = lengths_max - key_len[i]
        rows_pad = row_max - query_len[i]
        a = F.pad(a, (0, lengths_pad, 0, rows_pad), value=1.)
        out.append(a)
    return torch.stack(out)

def attn_subsequent_mask(lengths):
    attn_mask = attn_padding_mask(lengths, lengths)
    for i in range(attn_mask.size(0)):
        a = attn_mask[i]
        for j in range(a.size(0)):
            a[j, j+1:] = 1.
    #import pdb; pdb.set_trace()
    return attn_mask


class GeneralAttention(nn.Module):
    def __init__(self):
        super(GeneralAttention, self).__init__()
    def forward(self, Q, K, V, mask=None):
        # mmp! d_k指的是`dimention k`, 不是`D_k(k的方差)`
        #import pdb; pdb.set_trace()
        d_k = Q.size(-1)
        e = Q.bmm(K.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            masked = -2**14 if Q.dtype == torch.float16 else -2**31
            e.masked_fill_((mask - 1.).type(torch.bool).eq(False), masked)
            #mask = mask.type(e.dtype).to(e.device)
            #import pdb; pdb.set_trace()
            #e += mask
        #pdb.set_trace()
        #import pdb; pdb.set_trace()
        a = F.softmax(e, dim=-1)
        y = a.bmm(V)
        #import pdb; pdb.set_trace()
        return y


class MultiHeadAttention(nn.Module):
    # d_model: 输入和输出的向量维度
    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.linear_q = nn.Linear(d_k, n_head * d_k)    # d_q == d_k
        self.linear_k = nn.Linear(d_k, n_head * d_k)
        self.linear_v = nn.Linear(d_v, n_head * d_v)
        self.attn = GeneralAttention()
        self.out = nn.Linear(n_head * d_v, d_model)
    def forward(self, q, k, v, mask=None):
        #import pdb; pdb.set_trace()
        batch_size = q.size(0)
        n_head = self.n_head
        
        seq_q = q.size(1)
        seq_k = k.size(1)
        seq_v = v.size(1)

        d_q = q.size(2)
        d_k = k.size(2)
        d_v = v.size(2)

        q = self.linear_q(q).view(batch_size, seq_q, n_head, d_q).contiguous().permute(2, 0, 1, 3).contiguous().view(-1, seq_q, d_q)
        k = self.linear_q(k).view(batch_size, seq_k, n_head, d_k).contiguous().permute(2, 0, 1, 3).contiguous().view(-1, seq_k, d_k)
        v = self.linear_q(v).view(batch_size, seq_v, n_head, d_v).contiguous().permute(2, 0, 1, 3).contiguous().view(-1, seq_v, d_v)

        mask = mask.repeat(n_head, 1, 1)
        x = self.attn(q, k, v, mask=mask)
        x = x.view(n_head, batch_size, seq_q, d_v).contiguous().permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)
        x = self.out(x)
        return x


# https://github.com/jadore801120/attention-is-all-you-need-pytorch
#class ScaledDotProductAttention(nn.Module):
#    ''' Scaled Dot-Product Attention '''
#
#    def __init__(self, temperature, attn_dropout=0.1):
#        super().__init__()
#        self.temperature = temperature
#        self.dropout = nn.Dropout(attn_dropout)
#        self.softmax = nn.Softmax(dim=2)
#
#    def forward(self, q, k, v, mask=None):
#
#        attn = torch.bmm(q, k.transpose(1, 2))
#        attn = attn / self.temperature
#
#        if mask is not None:
#            attn = attn.masked_fill(mask == 0, -2**30)
#
#        attn = self.softmax(attn)
#        attn = self.dropout(attn)
#        output = torch.bmm(attn, v)
#
#        return output, attn
#
#class MultiHeadAttention0(nn.Module):
#    ''' Multi-Head Attention module '''
#
#    def __init__(self, n_head, d_model, d_k, d_v):
#        super().__init__()
#
#        self.n_head = n_head
#        self.d_k = d_k
#        self.d_v = d_v
#
#        self.w_qs = nn.Linear(d_model, n_head * d_k)
#        self.w_ks = nn.Linear(d_model, n_head * d_k)
#        self.w_vs = nn.Linear(d_model, n_head * d_v)
#        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
#
#        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
#        self.layer_norm = nn.LayerNorm(d_model)
#
#        self.fc = nn.Linear(n_head * d_v, d_model)
#        nn.init.xavier_normal_(self.fc.weight)
#
#    def forward(self, q, k, v, mask=None):
#
#        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#
#        sz_b, len_q, _ = q.size()
#        sz_b, len_k, _ = k.size()
#        sz_b, len_v, _ = v.size()
#
#        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
#
#        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
#        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
#        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
#
#        #import pdb; pdb.set_trace()
#        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
#        output, attn = self.attention(q, k, v, mask=mask)
#
#        output = output.view(n_head, sz_b, len_q, d_v)
#        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
#
#        output = self.fc(output)
#
#        return output

