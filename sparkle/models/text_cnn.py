import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F



class TextCNN(nn.Module):
    def __init__(self, emb, num_embeddings, embedding_dim, cnn_filter_num, max_padding_sentence_len, dropout_p):
        super(TextCNN, self).__init__()
        # padding_idx: If given, pads the output with the embedding vector at
        #              padding_idx (initialized to zeros) whenever it encounters the index.
        # 如果pytorch的Embedding没有提供这个padding的功能，第一时间想到的不应该是扩展Embedding使它加上这个功能吗？
        # self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.emb = emb
        self.conv_w2 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 2),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 2 + 1))
        self.conv_w3 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 3),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 3 + 1))
        self.conv_w4 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 4),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 4 + 1))
        self.conv_w5 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 5),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 5 + 1))
        self.conv_w6 = nn.Sequential(
                            nn.Conv1d(embedding_dim, cnn_filter_num, 6),
                            # nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.MaxPool1d(max_padding_sentence_len - 6 + 1))
        self.fc1 = nn.Linear(cnn_filter_num * 5, 128)
        self.dropout = nn.Dropout(p=dropout_p)
        self.out = nn.Linear(128, 5)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        batch_size = x.size(0)
        x = self.emb(x)
        x = x.permute(0, 2, 1)
        # x = x.view(batch_size, 1, *x.shape[1:])
        x_w2 = self.conv_w2(x)
        x_w3 = self.conv_w3(x)
        x_w4 = self.conv_w4(x)
        x_w5 = self.conv_w5(x)
        x_w6 = self.conv_w6(x)
        x = torch.cat([x_w2, x_w3, x_w4, x_w5, x_w6], dim=1)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.out(x)
        # 我用的是CrossEntropyLoss, 所以这里不需要用softmax
        # x = F.softmax(x)
        return x