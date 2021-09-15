#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, re, gc, random, time
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Constants
n_models = 5

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
models_module_path = './core/models/deep/attentionxml/core'
evaluation_module_path = './core/evaluation'

# Load modules
sys.path.extend([utils_module_path, preprocessing_module_path, features_module_path, models_module_path, evaluation_module_path])
from utils_module import *
from getting_data_module import *
from preprocessing_module import *
from embedding_module import *
from dataset import MultiLabelDataset
from data_utils import get_data, get_mlb, get_word_emb, output_res
from models import Model
from tree import FastAttentionXML
from networks import AttentionRNN
from sparseFeatures import generateSparseFeatures
from evaluation_module import *

# Data files
goldstandardFileName = "./toy datasets/codiesp/labels.tsv"
corpusDirectoryName = "./toy datasets/codiesp/text_files"
data_cnf = './core/models/deep/attentionxml/configure/datasets/codiesp.yaml'
model_cnf = './core/models/deep/attentionxml/configure/models/AttentionXML-codiesp.yaml'
outputDirectoryPath = "./results/models/attentionxml/"
outputDataDirectoryPath = outputDirectoryPath + "data"

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

# Generate and save sparse features
generateSparseFeatures(X_train, X_test, y_train, y_test, outputDataDirectoryPath)

# Load setting, data, and model
yaml = YAML(typ='safe')
data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
X_train, train_labels = get_data(data_cnf['train']['texts'], data_cnf['train']['labels'])
random_state = data_cnf['valid'].get('random_state', 1240)
X_train, X_valid, train_labels, valid_labels = train_test_split(X_train, train_labels, test_size=data_cnf['valid']['size'], random_state=random_state)
mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((train_labels, valid_labels)))
y_train, y_valid = mlb.transform(train_labels), mlb.transform(valid_labels)
labels_num = len(mlb.classes_)

# Train models
labels, scores = [], []
for tree_id in range(0, n_models):
   tree_id = F'-Tree-{tree_id}' if tree_id is not None else ''
   model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}{tree_id}')
   model = FastAttentionXML(labels_num, data_cnf, model_cnf, tree_id)
   start_time = time.time()
   model.train(X_train, y_train, X_valid, y_valid, mlb)
   print("--- %s seconds for training ---" % (time.time() - start_time))
   X_test, _ = get_data(data_cnf['test']['texts'], None)
   start_time = time.time()
   scores, labels = model.predict(X_test)
   print("--- %s seconds for testing ---" % (time.time() - start_time))
   labels = mlb.classes_[labels]
   output_res(data_cnf['output']['res'], F'{model_name}-{data_name}{tree_id}', scores, labels)
   del model

# Join predictions
ensemble_labels, ensemble_scores = [], []
for i in range(len(labels[0])):
   s = defaultdict(float)
   for j in range(len(labels[0][i])):
      for k in range(trees):
         s[labels[k][i][j]] += scores[k][i][j]
      s = sorted(s.items(), key=lambda x: x[1], reverse=True)
      ensemble_labels.append([x[0] for x in s[:len(labels[0][i])]])
      ensemble_scores.append([x[1] for x in s[:len(labels[0][i])]])

# Get order
mlb.fit(y_train + y_test + ensemble_labels, min=0)
order = list()
labelPositionDic = {mlb.classes_[i]:i for i in range(len(mlb.classes_))}
for i in range(len(ensemble_labels)):
   labelList = list()
   for l in range(len(ensemble_labels[i])): 
      labelList.append(labelPositionDic[ensemble_labels[i][l]])
   labelSet = set(labelList)
   for l in range(len(mlb.classes_)):
      if l not in labelSet:
         labelList.append(l)
   order.append(labelList)

# Evaluate
order = np.asarray(order)
y_pred_test = [[l for l in g] for g in ensemble_scores]
metric = Metric(zip(range(len(y_test)), y_test), y_train, propensity_score=True)
scores = metric.computeScores(range(len(y_test)), y_pred_test, mlb.classes_, order, k=[1,5,10])
print(SEP_LINE.join([TAB.join([str(e) for e in item_]) for item_ in scores.items()]))
print_file(y_test, os.path.join(outputDirectoryPath, 'scores.txt'), lambda_=lambda row: row[0:30])