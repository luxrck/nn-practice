import torch
from torch import nn, optim
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg, src_mask=None, trg_mask=None, mem_mask=None):
        #import pdb; pdb.set_trace()
        memory = self.encoder(src, src_mask)
        decoded = self.decoder(trg, memory, trg_mask, mem_mask)
        return decoded
