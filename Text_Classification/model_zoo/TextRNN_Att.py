# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh = nn.Tanh()
        self.W = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc2 = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        x, _ = x
        x_embed = self.embedding(x)
        H, _ = self.lstm(x_embed)

        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.W), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)

        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out





# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         if config.embedding_pretrained is not None:
#             self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
#         self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
#                             bidirectional=True, batch_first=True, dropout=config.dropout)
#         self.tanh1 = nn.Tanh()
#         # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
#         self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
#         self.tanh2 = nn.Tanh()
#         self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
#         self.fc = nn.Linear(config.hidden_size2, config.num_classes)
#
#     def forward(self, x):
#         x, _ = x
#         emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
#         H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
#
#         M = self.tanh1(H)  # [128, 32, 256]
#         # M = torch.tanh(torch.matmul(H, self.u))
#         alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
#         out = H * alpha  # [128, 32, 256]
#         out = torch.sum(out, 1)  # [128, 256]
#         out = F.relu(out)
#         out = self.fc1(out)
#         out = self.fc(out)  # [128, 64]
#         return out
