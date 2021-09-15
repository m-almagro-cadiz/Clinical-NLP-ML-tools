#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# Recurrent Convolutional Neural Network from Lai et al. (2015)
class TextRCNN(nn.Module):
   #Initialization
   def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pretrained_embeddings):
      super(TextRCNN, self).__init__()
      self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
      self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
      self.W2 = Linear(2 * hidden_size + embedding_dim, hidden_size * 2)
      self.fc = Linear(hidden_size * 2, output_dim)
      self.dropout = nn.Dropout(dropout)
   
   # Calculate the output according to the input data
   def forward(self, x):
      text, text_lengths = x # [batch size, sent len]
      text = text.permute(1, 0) # [seq_len, batch size]
      embedded = self.dropout(self.embedding(text)) # [seq_len, batch size, emb dim]
      outputs, _ = self.rnn(embedded) # [seq_len， batch_size, hidden_size * bidirectional]
      outputs = outputs.permute(1, 0, 2) # [batch_size, seq_len, hidden_size * bidirectional]
      embedded = embedded.permute(1, 0, 2) # [batch_size, seq_len, embeding_dim]
      x = torch.cat((outputs, embedded), 2) # [batch_size, seq_len, embdding_dim + hidden_size * bidirectional]
      y2 = torch.tanh(self.W2(x)).permute(0, 2, 1) # [batch_size, hidden_size * bidirectional, seq_len]
      y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2) # [batch_size, hidden_size * bidirectional]
      return self.fc(y3).squeeze(-1)