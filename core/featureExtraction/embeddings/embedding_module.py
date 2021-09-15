#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np

# Get pre-trained word vectors
def get_pretrained_weights(path, corpus_vocab, embed_dim):
   corpus_set = set(corpus_vocab)
   pretrained_vocab = set()
   wv_pretrained = torch.zeros(len(corpus_vocab), embed_dim)
   with open(path, 'rb') as f:
      for l in f:
         line = l.decode().split()
         if line[0] in corpus_set:
            pretrained_vocab.add(line[0])
            wv_pretrained[corpus_vocab.index(line[0])] = torch.from_numpy(np.array(line[1:]).astype(np.float))
      var = float(torch.var(wv_pretrained))
      for oov in corpus_set.difference(pretrained_vocab):
         wv_pretrained[corpus_vocab.index(oov)] = torch.empty(embed_dim).float().uniform_(-var, var)
   return wv_pretrained