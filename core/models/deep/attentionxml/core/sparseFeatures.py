#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, re, joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# Function to generate the files needed for the attention model
def generateSparseFeatures(X_train, X_test, y_train, y_test, outputPath):
   # Code labels
   mlb = MultiLabelBinarizer(sparse_output=True)
   mlb.fit(y_train)
   y_train_bin = mlb.transform(y_train)
   y_test_bin = mlb.transform(y_test)
   # Get training vocabulary and save text
   vocab = ['<PAD>', '<UNK>'] + [k for k,v in Counter([w for d in X_train for s in d for w in s]).items() if v > 5] # Discard low frequent words
   ind_ = {vocab[i]:i for i in range(len(vocab))}
   vocab = np.asarray(vocab)
   np.save(outputPath + 'vocab.npy', vocab)
   np.save(outputPath + 'train_labels.npy', np.asarray(y_train, dtype=object))
   np.save(outputPath + 'test_labels.npy', np.asarray(y_test, dtype=object))
   with open(outputPath + 'labels_binarizer', 'wb') as f: joblib.dump(mlb, f)
   # Code text and add padding
   X_train_pad, X_test_pad = [[ind_[w] if w in ind_ else 1 for s in re.split('[ \n]', d) for w in s.split() if w != 'unk'] for d in docs], [[ind_[w] if w in ind_ else 1 for s in re.split('[ \n]', d) for w in s.split() if w != 'unk'] for d in test_docs]
   max_train_pad = max([len(e) for e in X_train_pad])
   max_test_pad = max([len(e) for e in X_test_pad])
   X_train_pad = [d + [0] * (max_train_pad - len(d)) for d in X_train_pad]
   X_test_pad = [d + [0] * (max_test_pad - len(d)) for d in X_test_pad]
   X_train_pad = np.asarray(X_train_pad)
   X_test_pad = np.asarray(X_test_pad)
   np.save(outputPath + 'train_texts.npy', X_train_pad)
   np.save(outputPath + 'test_texts.npy', X_test_pad)
   # Save TF-IDF scores for splitting label space
   X_train_pad, X_test_pad = [' '.join([w if w in ind_ else '<UNK>' for s in re.split('[ \n]', d) for w in s.split()]) for d in docs], [' '.join([w if w in ind_ else '<UNK>' for s in re.split('[ \n]', d) for w in s.split()]) for d in test_docs]
   vectorizer = TfidfVectorizer()
   X_train_f = vectorizer.fit_transform(X_train_pad)
   X_test_f = vectorizer.transform(X_test_pad)
   # Save feature files
   dump_svmlight_file(X_train_f, sparse.csr_matrix(y_train_bin), outputPath + 'train.txt', multilabel=True)
   dump_svmlight_file(X_test_f, sparse.csr_matrix(y_test_bin), outputPath + 'test.txt', multilabel=True)
