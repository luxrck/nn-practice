import math
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F

from ash.modules.attention import MultiHeadAttention, attn_padding_mask, attn_subsequent_mask
from .encoder_decoder import EncoderDecoder


class PositionalEncoding(nn.Module):
    # max_len: max positions.
    def __init__(self, d_model, dropout_p=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        # d_model += d_model % 2
        self.dropout = nn.Dropout(p=dropout_p)
        pe = torch.zeros(max_len, d_model)

        # pos: (max_len, 1)
        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)
        dterm = torch.exp(-torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(pos * dterm)
        pe[:, 1::2] = torch.cos(pos * dterm)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    # inputs: (batch, seq_len, d_model)
    def forward(self, inputs):
        inputs += torch.autograd.Variable(self.pe[:, :inputs.size(1)], requires_grad=False)
        inputs = self.dropout(inputs)
        return inputs


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_out, d_hidden=2048, dropout_p=0.1):
        super(PositionwiseFFN, self).__init__()
        self.ffn = nn.Sequential(
                        nn.Linear(d_model, d_hidden),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_p),
                        nn.Linear(d_hidden, d_out))

    def forward(self, inputs):
        x = self.ffn(inputs)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head=8, d_ffn=2048, layernorm_eps=1e-3, dropout_p=0.1):
        super(TransformerEncoderLayer, self).__init__()
        ##self.attn = MultiHeadAttention(n_head=8, d_model=d_model, d_k=d_model, d_v=d_model)
        self.attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_model, d_v=d_model)
        self.a_norm = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        self.dropout1 = nn.Dropout(p=dropout_p)
       
        self.pffn = PositionwiseFFN(d_model=d_model, d_out=d_model, d_hidden=d_ffn, dropout_p=dropout_p)
        self.f_norm = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        self.dropout2 = nn.Dropout(p=dropout_p)
    
    def forward(self, x, mask=None):
        #import pdb; pdb.set_trace()
        y = self.attn(x, x, x, mask=mask)
        y = self.dropout1(y)
        x = self.a_norm(x + y)
        y = self.pffn(x)
        y = self.dropout2(y)
        x = self.f_norm(x + y)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head=8, n_layers=6, d_ffn=2048, layernorm_eps=1e-3, dropout_p=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoders = nn.ModuleList([
                            TransformerEncoderLayer(d_model=d_model, n_head=n_head, d_ffn=d_ffn, layernorm_eps=layernorm_eps, dropout_p=dropout_p) for _ in range(n_layers)])

    # x: (seq_len, batch, d_emb)
    def forward(self, x, mask=None):
        for encoder in self.encoders:
            #import pdb; pdb.set_trace()
            x = encoder(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head=8, d_ffn=2048, layernorm_eps=1e-3, dropout_p=0.1):
        super(TransformerDecoderLayer, self).__init__()
        #self.self_attn = MultiHeadAttention(n_head=8, d_model=d_model, d_k=d_model, d_v=d_model)
        self.self_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_model, d_v=d_model)
        #self.self_attn = MultiHeadAttention(n_head=8, d_model=d_model, d_attn=d_model*2, d_out=d_model)
        self.self_norm = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        self.dropout1 = nn.Dropout(p=dropout_p)

        #self.src_attn = MultiHeadAttention(n_head=8, d_model=d_model, d_k=d_model, d_v=d_model)
        self.src_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_model, d_v=d_model)
        #self.src_attn = MultiHeadAttention(n_head=8, d_model=d_model, d_attn=d_model*2, d_out=d_model)
        self.src_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.pffn = PositionwiseFFN(d_model=d_model, d_out=d_model, d_hidden=d_ffn, dropout_p=dropout_p)
        self.f_norm = nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        self.dropout3 = nn.Dropout(p=dropout_p)

    def forward(self, x, memory, trg_mask=None, mem_mask=None):
        #import pdb; pdb.set_trace()
        y = self.self_attn(x, x, x, mask=trg_mask)
        y = self.dropout1(y)
        x = self.self_norm(x + y)
        y = self.src_attn(x, memory, memory, mask=mem_mask)
        y = self.dropout2(y)
        x = self.src_norm(x + y)
        y = self.pffn(x)
        y = self.dropout3(y)
        x = self.f_norm(x + y)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head=8, n_layers=6, d_ffn=2048, layernorm_eps=1e-3, dropout_p=0.1):
        super(TransformerDecoder, self).__init__()
        self.decoders = nn.ModuleList([
                            TransformerDecoderLayer(d_model=d_model, n_head=n_head, d_ffn=d_ffn, layernorm_eps=layernorm_eps, dropout_p=dropout_p) for _ in range(n_layers)])

    def forward(self, x, memory, trg_mask=None, mem_mask=None):
        # import pdb; pdb.set_trace()
        for i,decoder in enumerate(self.decoders):
            #print("transformer:", i)
            #if i == 2:
            #    import pdb; pdb.set_trace()
            #import pdb; pdb.set_trace()
            x = decoder(x, memory, trg_mask, mem_mask)
            # decoded_mask = sequence_mask()
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size,
                d_model, n_head=8, n_encoder_layers=6, n_decoder_layers=6,
                d_ffn=2048, layernorm_eps=1e-3, dropout_p=0.1):
        super(Transformer, self).__init__()
        self.src_embeddings = nn.Embedding(src_vocab_size, d_model)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout_p=dropout_p)
        self.encoder_decoder = EncoderDecoder(encoder = TransformerEncoder(d_model=d_model, n_head=n_head, n_layers=n_decoder_layers, d_ffn=d_ffn, layernorm_eps=layernorm_eps, dropout_p=dropout_p),
                                              decoder = TransformerDecoder(d_model=d_model, n_head=n_head, n_layers=n_decoder_layers, d_ffn=d_ffn, layernorm_eps=layernorm_eps, dropout_p=dropout_p))
        #self.encoder_decoder = nn.Transformer(d_model=d_model, num_encoder_layers=n_encoder_layers, num_decoder_layers=n_decoder_layers)
        self.out = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p),
                    nn.Linear(d_model * 2, trg_vocab_size))

        #self._reset_parameters()

    # %% 参数初始化方式会对模型训练有这么大的影响？
    # 答: 是的！
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # 这是Linear层Weight的默认初始化方式
                nn.init.kaiming_uniform_(p, nonlinearity="leaky_relu")

    def forward(self, src, trg, src_lengths=None, trg_lengths=None):
        src = self.src_embeddings(src)
        trg = self.trg_embeddings(trg)
        src = self.positional_encoding(src)
        trg = self.positional_encoding(trg)
        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)
        #import pdb; pdb.set_trace()
        src_mask = attn_padding_mask(query_len=src_lengths, key_len=src_lengths) if src_lengths is not None else None
        trg_mask = attn_subsequent_mask(trg_lengths) if trg_lengths is not None else None
        mem_mask = attn_padding_mask(query_len=trg_lengths, key_len=src_lengths) if src_lengths is not None else None

        #src_mask = None
        #mem_mask = None
        x = self.encoder_decoder(src, trg, src_mask, trg_mask, mem_mask)

        #trg_mask = self.gen_mask(trg_lengths.topk(1)[0]).to(src_mask.device)
        #import pdb; pdb.set_trace()
        #src_mask = src_mask[:,0]
        #trg_mask_p = trg_mask_p[:,0]
        #mem_mask = mem_mask[:,0]
        #x = self.encoder_decoder(src, trg, tgt_mask=trg_mask, src_key_padding_mask=src_mask, tgt_key_padding_mask=trg_mask_p, memory_key_padding_mask=mem_mask)
        #x = x.transpose(0, 1)
        #import pdb; pdb.set_trace()
        x = self.out(x)
        #import pdb; pdb.set_trace()
        return x
