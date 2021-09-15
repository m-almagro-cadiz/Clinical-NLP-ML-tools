#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np
import os, io, sys, time, pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from utils_module import *

# Constants
CLF_FILE_NAME = 'classifier'
MAX_LENGTH = 180
N_JOBS = 4

# Train One-vs-Rest models using label-specific features
def trainModelWithLabelFeatures(X, getFeatures, classifier, y, labels, savingPath, usingCache=False, X_aux=np.asarray([]), y_aux=np.asarray([])):
   if len(X) == 0:
      raise ValueError('X empty!')
   if not y.any():
      raise ValueError('y empty!')
   if len(X) != y.shape[0]:
      raise ValueError('X and y dimensions does not match!')
   if not savingPath:
      raise ValueError('Classifier path empty!')
   if not os.path.exists(savingPath):
      os.makedirs(savingPath, 0o777)
   for i in range(len(labels)):
      start = time.time()
      if not os.path.exists(os.path.join(savingPath, labels[i][0:MAX_LENGTH] + '_' + CLF_FILE_NAME)):
         X_ = X.copy() + [X_aux[d] for d in range(len(X_aux)) if y_aux[d][i]==1]
         y_ = np.vstack([y.copy()] + [y_aux[d] for d in range(y_aux.shape[0]) if y_aux[d][i]==1])
         features = getFeatures(X_, labels[i], usingCache=False)
         y_i = np.array([item[i] for item in y_])
         classifier.fit(features, y_i)
         save_obj(classifier, os.path.join(savingPath, labels[i][0:MAX_LENGTH] + '_' + CLF_FILE_NAME))
      end = time.time()
      print('Classifier', i + 1, len(labels), labels[i][0:MAX_LENGTH], end - start)

# Train One-vs-Rest models using label-specific features based on samples with positive instances and only some negative ones
def trainModelWithLabelFeatures_NegSamp(X, getFeatures, classifier, y, labels, savingPath, usingCache=False, X_aux=np.asarray([]), y_aux=np.asarray([]), n=1):
   if len(X) == 0:
      raise ValueError('X empty!')
   if not y.any():
      raise ValueError('y empty!')
   if len(X) != y.shape[0]:
      raise ValueError('X and y dimensions does not match!')
   if not savingPath:
      raise ValueError('Classifier path empty!')
   if not os.path.exists(savingPath):
      os.makedirs(savingPath, 0o777)
   for i in range(len(labels)):
      start = time.time()
      if not os.path.exists(os.path.join(savingPath, labels[i][0:MAX_LENGTH] + '_' + CLF_FILE_NAME)):
         ns = negativeSample(n, i, labels, X_aux, y_aux)
         X_ = X.copy() + [X_aux[d] for d in range(len(X_aux)) if y_aux[d][i]==1] + ns
         y_ = np.vstack([y.copy()] + [y_aux[d] for d in range(y_aux.shape[0]) if y_aux[d][i]==1] + [np.zeros(y_aux[0].shape, dtype=int) for e in range(len(ns))])
         features = getFeatures(X_, labels[i], usingCache=False)
         y_i = np.array([item[i] for item in y_])
         classifier.fit(features, y_i)
         save_obj(classifier, os.path.join(savingPath, labels[i][0:MAX_LENGTH] + '_' + CLF_FILE_NAME))
      end = time.time()
      print('Classifier', i + 1, len(labels), labels[i][0:MAX_LENGTH], end - start)

# Train One-vs-Rest models using global features
def trainModelWithGlobalFeatures(X, classifier, y, savingPath):
   if X.nonzero()[0].shape[0] == 0:
      raise ValueError('X empty!')
   if not y.any():
      raise ValueError('y empty!')
   if X.shape[0] != y.shape[0]:
      raise ValueError('X and y dimensions does not match!')
   if not savingPath:
      raise ValueError('Classifier path empty!')
   if not os.path.exists(savingPath):
      os.makedirs(savingPath, 0o777)
   start = time.time()
   clf = OneVsRestClassifier(classifier, N_JOBS)
   clf.fit(X, y)
   save_obj(clf, os.path.join(savingPath, CLF_FILE_NAME))
   end = time.time()
   print("Time", end - start)
   return X_bsn, y_bin

# Predict labels using label-specific features and One-vs-Rest models
def predictWithLabelFeatures(X, getFeatures, labels, savingPath):
   if len(X) == 0:
      raise ValueError('X empty!')
   if not savingPath or not os.path.exists(savingPath):
      raise ValueError('Classifier path empty!')
   y_pred, y_prob = [list() for e in range(len(X))], list()
   for i in range(len(labels)):
      start = time.time()
      error = False
      if os.path.exists(os.path.join(savingPath, labels[i][0:MAX_LENGTH] + '_' + CLF_FILE_NAME)):
         features = getFeatures(X, labels[i], usingCache=True)
         classifier = load_obj(os.path.join(savingPath, labels[i][0:MAX_LENGTH] + '_' + CLF_FILE_NAME))
         y_pred_i = classifier.predict(features)
         for location in y_pred_i.nonzero()[0]:
            y_pred[location].append(labels[i])
         y_prob_i_ = classifier.predict_proba(features).tolist()
         y_prob_i = [item[1] for item in y_prob_i_]
         y_prob.append(y_prob_i)
      else:
         y_prob.append([0] * len(labels))
         error = True
      end = time.time()
      print(('***ERROR*** ' if error else '') + 'Prediction', i + 1, len(labels), labels[i][0:MAX_LENGTH], end - start)
   prob_labels = np.transpose(np.array(y_prob))
   order = np.argsort(-prob_labels)
   return y_pred, prob_labels, order

# Predict labels using global features and One-vs-Rest models
def predictWithGlobalFeatures(X, labels, savingPath):
   if X.nonzero()[0].shape[0] == 0:
      raise ValueError('X empty!')
   if not savingPath or not os.path.exists(savingPath):
      raise ValueError('Classifier path empty!')
   y_pred, y_prob = [list() for e in range(X.shape[0])], list()
   start = time.time()
   classifier = load_obj(os.path.join(savingPath, CLF_FILE_NAME))
   y_pred_, y_prob_ = classifier.predict(X), classifier.predict_proba(X)
   for i in range(len(labels)):
      for location in y_pred_[i].nonzero()[0]:
         y_pred[location].append(labels[i])
      y_prob_i_ = y_prob_[i].tolist()
      y_prob_i = [item[1] for item in y_prob_i_]
      y_prob.append(y_prob_i)
   prob_labels = np.transpose(np.array(y_prob))
   order = np.argsort(-prob_labels)
   end = time.time()
   print('Prediction', end - start)
   return y_pred, prob_labels, order
