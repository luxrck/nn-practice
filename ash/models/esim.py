import torch
from torch import nn, optim
import torch.nn.functional as F



class ESIM(nn.Module):
    def __init__(self, out_dim, num_embeddings, embedding_dim, hidden_dim, dropout_p=0.0, embedding_weight=None):
        super(ESIM, self).__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        if embedding_weight:
            self.emb.weight = nn.Parameter(embedding_weight)
        self.input_encoding = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True, dropout=dropout_p, bidirectional=True)
        self.nn = nn.Sequential(
                    nn.Linear(4 * 2 * embedding_dim, 4 * 2 * embedding_dim),
                    nn.Dropout(p=dropout_p),
                    nn.ReLU(),
                    nn.Linear(4 * 2 * embedding_dim, embedding_dim),
                    nn.Dropout(p=dropout_p),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, out_dim))

    # input shape: (batch, sentence_len)
    # 我的实现是预先将句子padding到统一的长度以方便训练
    def forward(self, inputs):
        # import pdb; pdb.set_trace()
        input_p, input_h = inputs[0], inputs[1]

        batch_size =len(inputs[0])

        # Input Encoding
        xp = self.emb(input_p)
        xh = self.emb(input_h)

        # Sentence embedding
        op, _ = self.input_encoding(xp) # (batch, sentence_len, embedding_dim)
        oh, _ = self.input_encoding(xh)

        # Local Inference Modeling (with soft-align attention)
        E_ph = torch.matmul(op, oh.transpose(-1, -2)) # We calculate the similarity matrix of op and oh
        wp = torch.matmul(F.softmax(E_ph, dim=2), oh)
        wh = torch.matmul(F.softmax(E_ph, dim=1).transpose(-1, -2), op)

        # Enhancement of local inference information
        mp = torch.cat([op, wp, op-wp, op*wp], dim=1)
        mh = torch.cat([oh, wh, oh-wh, oh*wh], dim=1)

        # Pooling
        embedding_dim = mp.size(-1)
        vp_a = F.avg_pool1d(mp.transpose(-1,-2), embedding_dim).view(batch_size, -1)
        vp_m = F.max_pool1d(mp.transpose(-1,-2), embedding_dim).view(batch_size, -1)
        vh_a = F.avg_pool1d(mh.transpose(-1,-2), embedding_dim).view(batch_size, -1)
        vh_m = F.max_pool1d(mh.transpose(-1,-2), embedding_dim).view(batch_size, -1)

        v = torch.cat([vp_a, vp_m, vh_a, vh_m], dim=1)

        y_p = self.nn(v)
        return y_p