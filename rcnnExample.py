#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, re, gc, random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

# Constants
device = 'cuda'
learning_rate = 1e-3
batch_size = 16
fixed_length = 250
embedding_length = 300
num_epochs = 25
hidden_size = 300
bidirectional = True

# seed
seed = 7777
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
gc.enable();

# Module files
utils_module_path = './core/utils'
preprocessing_module_path = './core/dataPreparation'
features_module_path = './core/featureExtraction/embeddings'
models_module_path = './core/models/deep/rcnn'
evaluation_module_path = './core/evaluation'

# Load modules
sys.path.extend([utils_module_path, preprocessing_module_path, features_module_path, models_module_path, evaluation_module_path])
from utils_module import *
from getting_data_module import *
from preprocessing_module import *
from embedding_module import *
from dataset import *
from model import *
from evaluation_module import *

# Data files
goldstandardFileName = "./toy datasets/codiesp/labels.tsv"
corpusDirectoryName = "./toy datasets/codiesp/text_files"
outputDirectoryPath = "./results/models/rcnn/"
emb_file = outputDirectoryPath + "embFile.txt"

# Create output folder
if not os.path.exists(outputDirectoryPath):
   os.makedirs(outputDirectoryPath, 0o777)

# Load data
goldstandard = readOneLineCsv(goldstandardFileName) # Read goldstandard
docs = getTextFromFiles(corpusDirectoryName) # Read documents
X, y, docids = getXY(docs, goldstandard) # Get input and output data

# Prepare and split data
nlp = NLP() # Preprocessing pipeline
X = nlp.preprocessDocuments(X, filter=lambda word: '_xxxx_' not in word, removeParenthesis=False, removeExpressions=False, dealWithNegation=False, groupRelatedwords=False, stemming=False) # Tokenize and clean input data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Code labels
mlb = AdvMultiLabelBinarizer() # Label coding component
y_train_bin = mlb.fit_transform(y_train, min=20) # Discard all labels with less than 20 documents
y_val_bin = mlb.transform(y_val)
y_test_bin = mlb.transform(y_test)
save_obj(mlb, os.path.join(outputDirectoryPath, 'mlb'))

# Show label statistics
statistics = list()
statistics.append(['Training labels'] + list(mlb.getTrainingStatistics()[1]))
statistics.append(['Unique training labels'] + list(mlb.getTrainingStatistics()[0]))
statistics.append(['Test labels'] + list(mlb.getTestStatistics()[1]))
statistics.append(['Unique test labels'] + list(mlb.getTestStatistics()[0]))
print(SEP_LINE.join([TAB.join([str(n) for n in e]) for e in statistics]))
print_file(statistics, os.path.join(outputDirectoryPath, 'Statistics'), header=['All', 'Frequent', 'Infrequent', 'Unseen'])

# Create dataframes and false pre-trained embeddings
df_train = pd.DataFrame(np.array([X_train, y_train_bin]), columns=['data', 'target'])
df_val = pd.DataFrame(np.array([X_val, y_val_bin]), columns=['data', 'target'])
df_test = pd.DataFrame(np.array([X_test, y_test_bin]), columns=['data', 'target'])
# Comment on these lines to load some pre-trained embeddings
vocab = [k for k,v in Counter([w for d in X_train for s in d for w in s]).items() if v > 10] # Discard low frequent words
emb = np.random.randn(len(vocab), embedding_length).astype('float32')
with open(emb_file, 'w', encoding='utf8') as f: f.write([vocab[i] + ' ' + ' '.join(emb[i]) for i in range(len(vocab))])

# Load features and batches
ds_train = CustomDataset(df_train, emb_file, max_doc_length=2350)
ds_val = CustomDataset(df_val, emb_file, max_doc_length=2350)
ds_test = CustomDataset(df_test, emb_file, max_doc_length=2350)
loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, pin_memory=True, shuffle=False, drop_last=True)
vloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, shuffle=False, drop_last=True)
tloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, shuffle=False, drop_last=True)
weights = get_pretrained_weights(emb_file, ds_train.vocab, embedding_length).to(device)

# Define model
model = TextRCNN(embedding_dim=embedding_length, output_dim=mlb.getTrainingStatistics()[0], hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional, dropout=0.3, pretrained_embeddings=weights).to(device)
opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate, betas=(0.75, 0.999), eps=1e-08, weight_decay=0)
loss_fn = FocalLoss(logits=True, reduce=True, pos_weight=(torch.Tensor([])).to(device))

# Train model
for epoch in range(0, num_epochs):      
   y_true_train = np.empty((0, len(targets)))
   y_pred_train = np.empty((0, len(targets)))
   total_loss_train, total_loss_valid = 0, 0          
   model.train()
   nb = len(loader)
   for batch_idx, (docs, labels, doc_lengths) in enumerate(loader):
      if batch_idx % 50 == 0:
         print(str(batch_idx) + ' from ' + str(nb))
      docs = docs.to(device)  # (batch_size, padded_doc_length, padded_sent_length)
      labels = labels.type(torch.FloatTensor).to(device)  # (batch_size)
      doc_lengths = doc_lengths.to(device)  # (batch_size)
      pred = model((docs, doc_lengths))
      opt.zero_grad()
      loss = loss_fn(pred, labels)
      loss.backward()
      opt.step()
      y_true_train = np.concatenate([y_true_train, labels.detach().cpu().numpy()], axis = 0)
      y_pred_train = np.concatenate([y_pred_train, pred.detach().cpu().numpy()], axis = 0)
      total_loss_train += loss.item()
   # Get prediction for validation
   model.eval()
   y_true_val = np.empty((0, len(targets)))
   y_pred_val = np.empty((0, len(targets)))
   for batch_idx, (docs, labels, doc_lengths) in enumerate(vloader):
      with torch.no_grad():
         docs = docs.to(device)  # (batch_size, padded_doc_length, padded_sent_length)
         labels = labels.type(torch.FloatTensor).to(device)  # (batch_size)
         doc_lengths = doc_lengths.to(device)  # (batch_size)
         pred = model((docs, doc_lengths))
         y_true_val = np.concatenate([y_true_val, labels.detach().cpu().data.numpy()], axis = 0)
         y_pred_val = np.concatenate([y_pred_val, pred.detach().cpu().numpy()], axis = 0)
   vloss = total_loss_val/len(vloader)
   y_pred_val[y_pred_val < 0.5] = 0
   y_pred_val[y_pred_val >= 0.5] = 1
   val_num_corrects = (y_pred_val == y_true_val).sum()
   val_tp = ((y_pred_val == y_true_val) & (y_true_val == 1)).sum()
   val_fp = y_pred_val.sum() - val_tp
   val_fn = y_true_val.sum() - val_tp
   val_acc = 100.0 * val_num_corrects/len(y_true_val)
   val_precision = val_tp / (val_tp + val_fp)
   val_recall = val_tp / (val_tp + val_fn)
   val_f = 100.0 * 2 * val_precision * val_recall / (val_precision + val_recall)
   print(f'Epoch {epoch+1}: Val loss: {vloss:.4f}, Val acc: {val_acc:.4f}, Val F: {val_f:.4f}')
   torch.save(model, outputDirectoryPath + 'han_' + str(epoch))
   gc.collect()


# Predict test labels
y_true_test = np.empty((0, len(targets)))
y_pred_test = np.empty((0, len(targets)))
for batch_idx, (docs, labels, doc_lengths) in enumerate(tloader):
   with torch.no_grad():
      docs = docs.to(device)  # (batch_size, padded_doc_length, padded_sent_length)
      labels = labels.type(torch.FloatTensor).to(device)  # (batch_size)
      doc_lengths = doc_lengths.to(device)  # (batch_size)
      pred = model((docs, doc_lengths))
      y_true_test = np.concatenate([y_true_test, labels.detach().cpu().data.numpy()], axis = 0)
      y_pred_test = np.concatenate([y_pred_test, pred.detach().cpu().numpy()], axis = 0)

# Evaluate
order = np.argsort(-y_pred_test)
y_pred_test[y_pred_test < 0.5] = 0
y_pred_test[y_pred_test >= 0.5] = 1
metric = Metric(zip(range(len(y_true_test)), y_true_test), y_train, propensity_score=True)
scores = metric.computeScores(range(len(y_true_test)), y_pred_test, mlb.classes_, order, k=[1,5,10])
print(SEP_LINE.join([TAB.join([str(e) for e in item_]) for item_ in scores.items()]))
print_file(y_true_test, os.path.join(outputDirectoryPath, 'scores.txt'), lambda_=lambda row: row[0:30])