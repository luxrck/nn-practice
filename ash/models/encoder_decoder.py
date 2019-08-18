import torch
from torch import nn, optim
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg, src_lengths=None, trg_lengths=None):
        # import pdb; pdb.set_trace()
        src_encoded = self.encoder(src, src_lengths)
        decoded = self.decoder(trg, src_encoded, src_lengths, trg_lengths)
        return decoded