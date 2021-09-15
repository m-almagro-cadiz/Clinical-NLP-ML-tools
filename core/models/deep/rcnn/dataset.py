#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch, csv
import pandas as pd
from torch.utils.data import Dataset

# Grouping individual documents into batches
class CustomDataset(Dataset):
   # Initialization
   def __init__(self, df, word_map_path, max_doc_length=500):
      self.max_doc_length = max_doc_length
      self.data = dict()
      self.data['data'] = df.text__.tolist()
      self.data['target'] = df.target.tolist()
      self.vocab = pd.read_csv(
            filepath_or_buffer=word_map_path,
            header=None,
            sep=" ",
            quoting=csv.QUOTE_NONE,
            usecols=[0]).values[:500000]
      self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]
   
   # Transform text to ids
   def transform(self, text):
      doc = text.lower().split('\n')
      doc = [self.vocab.index(word) if word in self.vocab else 1 for sent in doc for word in sent.split() + ['<pad>'] if sent.strip()]
      doc = doc[:self.max_doc_length]
      num_words = min(len(doc), self.max_doc_length)
      if num_words == 0:
         return None, -1, None
      return doc, num_words
   
   # Process one item
   def __getitem__(self, i):
      label = self.data['target'][i]
      text = self.data['data'][i]
      doc, num_words = self.transform(text)
      if num_words == -1:
         return None
      return doc, label, num_words
   
   # Get length
   def __len__(self):
      return len(self.data['data'])
   
   @property
   def vocab_size(self):
      return len(self.vocab)
   
   @property
   def num_classes(self):
      return len(targets)

# Function to group data for a batch
def collate_fn(batch):
   batch = filter(lambda x: x is not None, batch)
   docs, labels, doc_lengths = list(zip(*batch))
   bsz = len(labels)
   batch_max_doc_length = max(doc_lengths)
   docs_tensor = torch.zeros((bsz, batch_max_doc_length)).long()
   for doc_idx, doc in enumerate(docs):
      doc_length = doc_lengths[doc_idx]
      docs_tensor[doc_idx, :doc_length] = torch.LongTensor(doc)
   return docs_tensor, torch.LongTensor(labels), torch.LongTensor(doc_lengths)