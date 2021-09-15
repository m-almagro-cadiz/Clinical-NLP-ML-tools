#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Module files
utils_module_path = './core/utils'
preprocessing_module_path = './core/dataPreparation'
features_module_path = './core/featureExtraction/tf'
models_module_path = './core/models/conventional'
evaluation_module_path = './core/evaluation'

# Load modules
sys.path.extend([utils_module_path, preprocessing_module_path, features_module_path, models_module_path, evaluation_module_path])
from utils_module import *
from getting_data_module import *
from preprocessing_module import *
from feature_module import *
from wrapperOvR_module import *
from evaluation_module import *

# Data files
goldstandardFileName = "./toy datasets/codiesp/labels.tsv"
corpusDirectoryName = "./toy datasets/codiesp/text_files"
outputDirectoryPath = "./results/models/ovr"

# Constants
ngram_N1 = 1
ngram_N2 = 3

# Create output folder
if not os.path.exists(outputDirectoryPath):
   os.makedirs(outputDirectoryPath, 0o777)

# Load data
goldstandard = readOneLineCsv(goldstandardFileName) # Read goldstandard
docs = getTextFromFiles(corpusDirectoryName) # Read documents
X, y, docids = getXY(docs, goldstandard) # Get input and output data

# Prepare and split data
nlp = NLP() # Preprocessing pipeline
X = nlp.preprocessDocuments(X, filter=lambda word: '_xxxx_' not in word, removeParenthesis=False, removeExpressions=False) # Tokenize and clean input data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Code labels
mlb = AdvMultiLabelBinarizer() # Label coding component
y_train_bin = mlb.fit_transform(y_train, min=30) # Discard all labels with less than 30 documents
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

# Compute features
features = Features(savingPath=outputDirectoryPath)
features.fitTFBNS(X_train, y_train_bin, mlb.classes_.tolist(), ngrams=(ngram_N1, ngram_N2))

# Train models
# classifier = SVC(probability=True, random_state=42)
# classifier = MLPClassifier((80,80,80), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001)
classifier = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
parameters = {'booster':['gbtree'], 'max_depth': [4, 8, 12], 'eta': [0.1, 0.5, 1], 'gamma': [0, 1, 3], 'lambda': [1, 2], 'alpha': [0, 1]}
clf = GridSearchCV(classifier, parameters, refit=True, scoring='accuracy') 
trainModelWithLabelFeatures(X_train, features.transformTFBNS_i, clf, y_train_bin, mlb.classes_.tolist(), outputDirectoryPath)

# Predict test labels
features.resetCache()
y_pred, y_prob, order = predictWithLabelFeatures(X_test, features.transformTFBNS_i, mlb.classes_.tolist(), outputDirectoryPath)

# Evaluate
metric = Metric(zip(range(len(y_test)), y_test), y_train, propensity_score=True)
scores = metric.computeScores(range(len(y_test)), y_pred, mlb.classes_, order, k=[1,5,10])
print(SEP_LINE.join([TAB.join([str(e) for e in item_]) for item_ in scores.items()]))
print_file(y_test, os.path.join(outputDirectoryPath, 'scores.txt'), lambda_=lambda row: row[0:30])