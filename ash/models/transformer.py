import math
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F

from ash.modules.attention import MultiHeadAttention, attn_padding_mask, attn_subsequent_mask
from .encoder_decoder import EncoderDecoder


class PositionalEncoding(nn.Module):
    # max_len: max positions.
    def __init__(self, d_input, dropout_p=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        # d_input += d_input % 2
        self.dropout = nn.Dropout(p=dropout_p)
        pe = torch.zeros(max_len, d_input)

        # pos: (max_len, 1)
        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)
        dterm = torch.exp(-torch.arange(0, d_input, 2).float() * math.log(10000.0) / d_input)
        pe[:, 0::2] = torch.sin(pos * dterm)
        pe[:, 1::2] = torch.cos(pos * dterm)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    # inputs: (batch, seq_len, d_input)
    def forward(self, inputs):
        inputs += torch.autograd.Variable(self.pe[:, :inputs.size(1)], requires_grad=False)
        inputs = self.dropout(inputs)
        return inputs


class PositionwiseFFN(nn.Module):
    def __init__(self, d_input, d_out, dropout_p):
        super(PositionwiseFFN, self).__init__()
        d_hidden = 2048
        self.ffn = nn.Sequential(
                        nn.Linear(d_input, d_hidden),
                        nn.Dropout(p=dropout_p),
                        nn.ReLU(),
                        nn.Linear(d_hidden, d_out))
    def forward(self, inputs):
        x = self.ffn(inputs)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_input, dropout_p=0.1):
        super(TransformerEncoderLayer, self).__init__()
        ##self.attn = MultiHeadAttention(n_head=8, d_model=d_input, d_k=d_input, d_v=d_input)
        self.attn = MultiHeadAttention(n_head=8, d_model=d_input, d_k=d_input, d_v=d_input)
        self.a_norm = nn.LayerNorm(normalized_shape=d_input)
        self.dropout1 = nn.Dropout(p=dropout_p)
       
        self.pffn = PositionwiseFFN(d_input=d_input, d_out=d_input, dropout_p=dropout_p)
        self.f_norm = nn.LayerNorm(normalized_shape=d_input)
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
    def __init__(self, embeddings, n_layers=6, dropout_p=0.1):
        super(TransformerEncoder, self).__init__()
        self.emb = embeddings
        d_input = self.emb.embedding_dim
        self.positional_encoding = PositionalEncoding(d_input=d_input, dropout_p=dropout_p)
        self.encoders = nn.ModuleList([
                            copy.deepcopy(TransformerEncoderLayer(d_input=d_input, dropout_p=dropout_p)) for _ in range(n_layers)])

    # x: (seq_len, batch, d_emb)
    def forward(self, x, lengths=None):
        x = x.transpose(0, 1)
        x = self.emb(x)
        x = self.positional_encoding(x)
        mask = attn_padding_mask(lengths, lengths) if lengths is not None else None
        for encoder in self.encoders:
            x = encoder(x, mask)
        #import pdb; pdb.set_trace()
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_input, dropout_p=0.1):
        super(TransformerDecoderLayer, self).__init__()
        #self.self_attn = MultiHeadAttention(n_head=8, d_model=d_input, d_k=d_input, d_v=d_input)
        self.self_attn = MultiHeadAttention(n_head=8, d_model=d_input, d_k=d_input, d_v=d_input)
        #self.self_attn = MultiHeadAttention(n_head=8, d_input=d_input, d_attn=d_input*2, d_out=d_input)
        self.self_norm = nn.LayerNorm(normalized_shape=d_input)
        self.dropout1 = nn.Dropout(p=dropout_p)

        #self.src_attn = MultiHeadAttention(n_head=8, d_model=d_input, d_k=d_input, d_v=d_input)
        self.src_attn = MultiHeadAttention(n_head=8, d_model=d_input, d_k=d_input, d_v=d_input)
        #self.src_attn = MultiHeadAttention(n_head=8, d_input=d_input, d_attn=d_input*2, d_out=d_input)
        self.src_norm = nn.LayerNorm(normalized_shape=d_input)
        self.dropout2 = nn.Dropout(p=dropout_p)

        self.pffn = PositionwiseFFN(d_input=d_input, d_out=d_input, dropout_p=dropout_p)
        self.f_norm = nn.LayerNorm(normalized_shape=d_input)
        self.dropout3 = nn.Dropout(p=dropout_p)

    def forward(self, x, src_encoded, src_mask=None, trg_mask=None):
        #import pdb; pdb.set_trace()
        y = self.self_attn(x, x, x, mask=trg_mask)
        y = self.dropout1(y)
        x = self.self_norm(x + y)
        y = self.src_attn(x, src_encoded, src_encoded, mask=src_mask)
        y = self.dropout2(y)
        x = self.src_norm(x + y)
        y = self.pffn(x)
        y = self.dropout3(y)
        x = self.f_norm(x + y)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embeddings, n_layers=6, dropout_p=0.1):
        super(TransformerDecoder, self).__init__()
        self.emb = embeddings
        d_input = self.emb.embedding_dim
        self.positional_encoding = PositionalEncoding(d_input=d_input, dropout_p=dropout_p)
        self.decoders = nn.ModuleList([
                            copy.deepcopy(TransformerDecoderLayer(d_input=d_input, dropout_p=dropout_p)) for _ in range(n_layers)])

    def forward(self, x, src_encoded, src_lengths=None, trg_lengths=None):
        x = x.transpose(0, 1)
        x = self.emb(x)
        x = self.positional_encoding(x)
        # import pdb; pdb.set_trace()
        src_mask = attn_padding_mask(query_len=trg_lengths, key_len=src_lengths) if src_lengths is not None else None
        trg_mask = attn_subsequent_mask(trg_lengths) if trg_lengths is not None else None
        for i,decoder in enumerate(self.decoders):
            #print("transformer:", i)
            #import pdb; pdb.set_trace()
            x = decoder(x, src_encoded, src_mask, trg_mask)
            # decoded_mask = sequence_mask()
        #import pdb; pdb.set_trace()
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, n_encoder_layers, n_decoder_layers, dropout_p=0.1):
        super(Transformer, self).__init__()
        self.src_embeddings = nn.Embedding(src_vocab_size, d_model)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, d_model)
        self.encoder_decoder = EncoderDecoder(encoder = TransformerEncoder(self.src_embeddings, n_layers=n_decoder_layers, dropout_p=dropout_p),
                                              decoder = TransformerDecoder(self.trg_embeddings, n_layers=n_decoder_layers, dropout_p=dropout_p))
        self.out = nn.Sequential(
                nn.Linear(d_model, trg_vocab_size))

    def forward(self, src, trg, src_len=None, trg_len=None):
        # import pdb; pdb.set_trace()
        x = self.encoder_decoder(src, trg, src_len, trg_len)
        x = self.out(x)
        #import pdb; pdb.set_trace()
        return x
