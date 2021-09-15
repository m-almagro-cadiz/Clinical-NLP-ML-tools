#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string, os, json, re, io, random, sys, time, array, warnings, itertools
from nltk import ngrams as buildNgrams
import numpy as np
from scipy.sparse import SparseEfficiencyWarning
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from bns_module import *
from utils_module import *

# Constants
featureConfigFileName = 'featureConfig'
weighTfModelName = 'weighVectorizer'
tfModelFileName = 'tfVectorizer'
selectKBestModelName = 'selectKBest'
scalerModelFileName = 'scaler'

# Class to encode multiple labels in one-hot vectors
# It includes advanced functionalities, such as a filter to remove those with less than N instances and a function to group hierarchically
class AdvMultiLabelBinarizer(MultiLabelBinarizer):
   # Initialization
   def __init__(self, *, classes=None, sparse_output=False):
      self.classes = classes
      self.sparse_output = sparse_output
   
   # Learn a coding based on input data
   def fit(self, y, min=0, grouping=False, **argsGrouping):
      if self.classes is None:
         counter = Counter(itertools.chain.from_iterable(y)) # Get statistics
         self.freq = {k:v for k,v in counter.items()}
         classes = sorted([k for k,v in counter.items() if v >= min]) # Filter labels with few examples
         self.n_training_original_labels, self.n_training_original_instances, self.n_training_instances = len(counter), sum(counter.values()), sum([v for k,v in counter.items() if k in classes])
         if grouping: # Group labels linked to the same parent node
            groups, grouped_classes = groupCodes(classes, **argsGrouping)
      elif len(set(self.classes)) < len(self.classes):
         raise ValueError("The classes argument contains duplicate classes. Remove these duplicates before passing them to MultiLabelBinarizer.")
      else:
         classes = self.classes
      dtype = np.int if all(isinstance(c, int) for c in classes) else object
      self.classes_ = np.empty(len(classes), dtype=dtype)
      self.classes_[:] = classes
      if grouping:
         dtype = np.int if all(isinstance(c, int) for c in grouped_classes) else object
         self.grouped_classes_ = np.asarray(grouped_classes)
         self.groups = np.asarray(groups)
      return self
   
   # Learn the coding and return one-hot vectors
   def fit_transform(self, y, min=0, grouping=False, **argsGrouping):
      if self.classes is not None:
         return self.fit(y, min=min, grouping=grouping).transform(y, grouping=grouping)
      class_mapping = defaultdict(int)
      class_mapping.default_factory = class_mapping.__len__
      yt, unknown_ = self._transform(y, class_mapping)
      yt_count = np.sum(yt, axis=0)
      counter = Counter(itertools.chain.from_iterable(y))
      self.freq = {k:v for k,v in counter.items()}
      tmp = sorted(class_mapping, key=class_mapping.get)
      self.n_training_original_labels, self.n_training_original_instances = len(tmp), sum([len(g) for g in y])
      if min > 0: # Filter labels with few examples
         indices_ = np.where(yt_count>=min)[1]
         yt = yt[:, indices_]
         tmp = [tmp[i] for i in range(len(tmp)) if i in indices_]
      self.n_training_instances = np.sum(yt)
      dtype = np.int if all(isinstance(c, int) for c in tmp) else object
      class_mapping = np.empty(len(tmp), dtype=dtype)
      class_mapping[:] = tmp
      self.classes_, inverse = np.unique(class_mapping, return_inverse=True)
      if grouping: # Group labels linked to the same parent node
         groups, grouped_classes = groupCodes(tmp, **argsGrouping)
         self.grouped_classes_ = np.asarray(grouped_classes)
         self.groups = np.asarray(groups)
         indices_ = [[tmp.index(label) for label in labels.split(' ')] for labels in self.grouped_classes_]
         grouped_yt = yt[:, [ind_[0] for ind_ in indices_]]
         warnings.simplefilter('ignore',SparseEfficiencyWarning)
         for i in range(len(indices_)):
            for ind_ in indices_[i][1::]:
               grouped_yt[:, i] += yt[:, ind_]
         if not self.sparse_output:
            grouped_yt = grouped_yt.toarray()
         return grouped_yt
      yt.indices = np.array(inverse[yt.indices], dtype=yt.indices.dtype, copy=False)
      if not self.sparse_output:
         yt = yt.toarray()
      return yt
   
   # Return one-hot vectors based on the learnt coding
   def transform(self, y, grouping=False):
      class_to_index = self._build_cache(grouping)
      yt, unknown_ = self._transform(y, class_to_index)
      self.n_test_original_labels, self.n_test_new_labels, self.n_test_original_instances, self.n_test_instances = len({l for g in y for l in g}), len(unknown_), sum([len(g) for g in y]), np.sum(yt)
      if not self.sparse_output:
         yt = yt.toarray()
      return yt
   
   # Build a dictionary with the new indexes after grouping labels
   def _build_cache(self, grouping):
      if grouping:
         return dict([(k_,v) for k,v in zip(self.grouped_classes_, range(len(self.grouped_classes_))) for k_ in k.split(' ')])
      else:
         return dict(zip(self.classes_, range(len(self.classes_))))
   
   # Return one-hot vectors based on the provided mapping
   def _transform(self, y, class_mapping):
      indices = array.array('i')
      indptr = array.array('i', [0])
      unknown = set()
      for labels in y:
         index = set()
         for label in labels:
            try:
               index.add(class_mapping[label])
            except KeyError:
               unknown.add(label)
         indices.extend(index)
         indptr.append(len(indices))
      data = np.ones(len(indices), dtype=int)
      return sp.sparse.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(set(class_mapping.values())))), unknown
   
   # Recover original labels from coded vectors
   def inverse_transform(self, yt, grouping=False):
      if yt.shape[1] != len(self.classes_) and not grouping:
         raise ValueError('Expected indicator for {0} classes, but got {1}'.format(len(self.classes_), yt.shape[1]))
      if yt.shape[1] != len(self.grouped_classes_) and grouping:
         raise ValueError('Expected indicator for {0} classes, but got {1}'.format(len(self.grouped_classes_), yt.shape[1]))
      if sp.sparse.issparse(yt):
         yt = yt.tocsr()
         if len(yt.data) != 0 and len(np.setdiff1d(yt.data, [0, 1])) > 0:
            raise ValueError('Expected only 0s and 1s in label indicator.')
         if grouping:
            return [tuple(self.grouped_classes_.take(yt.indices[start:end])) for start, end in zip(yt.indptr[:-1], yt.indptr[1:])]
         if not grouping:
            return [tuple(self.classes_.take(yt.indices[start:end])) for start, end in zip(yt.indptr[:-1], yt.indptr[1:])]
      else:
         unexpected = np.setdiff1d(yt, [0, 1])
         if len(unexpected) > 0:
            raise ValueError('Expected only 0s and 1s in label indicator. Also got {0}'.format(unexpected))
         if grouping:
            return [tuple(self.grouped_classes_.compress(indicators)) for indicators in yt]
         else:
            return [tuple(self.classes_.compress(indicators)) for indicators in yt]
   
   # Obtain statistics generated during the learning of coding.
   def getTrainingStatistics(self):
      return (self.n_training_original_labels, len(self.classes_), self.n_training_original_labels - len(self.classes_)), (self.n_training_original_instances, self.n_training_instances, self.n_training_original_instances - self.n_training_instances)
   
   # Obtain statistics generated during the transforming.
   def getTestStatistics(self):
      return (self.n_test_original_labels, self.n_test_original_labels - self.n_test_new_labels, self.n_test_new_labels), (self.n_test_original_instances, self.n_test_instances, self.n_test_original_instances - self.n_test_instances)

# Get an array with the presence or absence of a label in each instance
def getBinaryOutput(y, label):
   return np.asarray([1 if label in group else 0 for group in y])

# Class to transform raw text documents into Term Frequency vectors, weighed by Inverse Document Frequency (IDF) or Bi-Normal Separation (BNS) scores
class Features:
   #Initialization
   def __init__(self, savingPath):
      if not savingPath:
         raise ValueError('Feature path empty!')
      if not os.path.exists(savingPath):
         os.makedirs(savingPath, 0o777)
      self.savingPath = savingPath
      self.features = np.asarray([])
   
   # Generate n-grams
   def getNgrams(self, sentence, n1, n2):
      ngramList = list()
      for i in range(n1, n2 + 1):
         ngramList.extend([list(e) for e in buildNgrams(sentence, i)])
      return ngramList
   
   # Learn TF-IDF representation
   def fitTransformTFIDF(self, X, y=np.asarray([]), filtering='all', ngrams=(1,1)):
      start = time.time()
      if not X or len(X) == 0:
         raise ValueError('X empty!')
      if not isinstance(X[0], list) or not isinstance(X[0][0], list) or not isinstance(X[0][0][0], str):
         raise ValueError('X invalid format!')
      if not isinstance(filtering, int) or filtering <= 0:
         filtering = 'all'
      save_obj((filtering, ngrams), os.path.join(self.savingPath, featureConfigFileName))
      tfidfVectorizer, selectKBest = TfidfVectorizer(), SelectKBest(chi2, k=filtering)
      X_idf = tfidfVectorizer.fit_transform([' '.join([' '.join(['_'.join(ngram) for ngram in self.getNgrams(sentence, ngrams[0], ngrams[1])]) for sentence in doc]) for doc in X])
      save_obj(tfidfVectorizer, os.path.join(self.savingPath, weighTfModelName))
      if y.any() and filtering != 'all':
         X_idf = selectKBest.fit_transform(X_idf, y)
         save_obj(selectKBest, os.path.join(self.savingPath, selectKBestModelName))
      end = time.time()
      print("Time", end - start)
      return X_idf
   
   # Transform all the instances into TF-IDF representation
   def transformTFIDF(self, X):
      start = time.time()
      if not X or len(X) == 0:
         raise ValueError('X empty!')
      if not isinstance(X[0], list) or not isinstance(X[0][0], list) or not isinstance(X[0][0][0], str):
         raise ValueError('X invalid format!')
      if not os.path.exists(os.path.join(self.savingPath, featureConfigFileName)):
         raise ValueError('Configuration file unlocated!')
      filtering, ngrams = load_obj(os.path.join(self.savingPath, featureConfigFileName))
      tfidfVectorizer = load_obj(os.path.join(self.savingPath, weighTfModelName))
      X_idf = tfidfVectorizer.transform([' '.join([' '.join(['_'.join(ngram) for ngram in self.getNgrams(sentence, ngrams[0], ngrams[1])]) for sentence in doc]) for doc in X])
      if filtering != 'all':
         selectKBest = load_obj(os.path.join(self.savingPath, selectKBestModelName))
         X_idf = selectKBest.transform(X_idf)
      end = time.time()
      print("Time", end - start)
      return X_idf
   
   # Learn TF-BNS representations
   def fitTFBNS(self, X, y, labels, filtering='all', ngrams=(1,1), X_aux=np.asarray([]), y_aux=np.asarray([])):
      if not X or len(X) == 0:
         raise ValueError('X empty!')
      if not isinstance(X[0], list) or not isinstance(X[0][0], list) or not isinstance(X[0][0][0], str):
         raise ValueError('X invalid format!')
      if not y.any() or y.shape[0] == 0:
         raise ValueError('y invalid format!')
      if not isinstance(filtering, int) or filtering <= 0:
         filtering = 'all'
      self.features = np.asarray([])
      save_obj((filtering, ngrams), os.path.join(self.savingPath, featureConfigFileName))
      X_ngrams = [' '.join([' '.join(['_'.join(ngram) for ngram in self.getNgrams(sentence, ngrams[0], ngrams[1])]) for sentence in doc]) for doc in X]
      X_aux_ngrams = [' '.join([' '.join(['_'.join(ngram) for ngram in self.getNgrams(sentence, ngrams[0], ngrams[1])]) for sentence in doc]) for doc in X_aux]
      if not os.path.exists(os.path.join(self.savingPath, tfModelFileName)):
         countVectorizer = CountVectorizer()
         X_joint_aux = countVectorizer.fit(X_ngrams + X_aux_ngrams)
         save_obj(countVectorizer, os.path.join(self.savingPath, tfModelFileName))
      else:
         countVectorizer = load_obj(os.path.join(self.savingPath, tfModelFileName))
      X_joint_aux = countVectorizer.transform(X_ngrams)
      X_aux_ = countVectorizer.transform(X_aux_ngrams)
      for i in range(len(labels)):
         start = time.time()
         X_joint = sp.sparse.vstack([X_joint_aux.copy()] + [X_aux_[d] for d in range(X_aux_.shape[0]) if y_aux[d][i]==1])
         y_joint = np.vstack([y.copy()] + [y_aux[d] for d in range(y_aux.shape[0]) if y_aux[d][i]==1])
         if filtering != 'all':
            if not os.path.exists(os.path.join(self.savingPath, labels[i][0:180] + '_' + selectKBestModelName)):
               selectKBest = SelectKBest(chi2, k=filtering)
               X_joint = selectKBest.fit_transform(X_joint, y_joint[:, i])
               save_obj(selectKBest, os.path.join(self.savingPath, labels[i][0:180] + '_' + selectKBestModelName))
            else:
               if not os.path.exists(os.path.join(self.savingPath, labels[i][0:180] + '_' + weighTfModelName + '_')):
                  selectKBest = load_obj(os.path.join(self.savingPath, labels[i][0:180] + '_' + selectKBestModelName))
                  X_joint = selectKBest.transform(X_joint)
         if not os.path.exists(os.path.join(self.savingPath, labels[i][0:180] + '_' + weighTfModelName + '_')):
            bnsTransformer = BnsTransformer()
            bnsTransformer.fit(X_joint, y_joint[:,i])
            save_obj(bnsTransformer, os.path.join(self.savingPath, labels[i][0:180] + '_' + weighTfModelName + '_'))
         end = time.time()
         print('Features', i + 1, len(labels), labels[i][0:180], end - start, str(X_joint_aux.shape[0]) + ':' + str(X_joint.shape[0]))
   
   # Learn TF-BNS representation by only using positive examples and the most similar negative examples for each label
   def fitTFBNS_NegSamp(self, X, y, labels, filtering='all', ngrams=(1,1), X_aux=np.asarray([]), y_aux=np.asarray([]), n=1):
      if not X or len(X) == 0:
         raise ValueError('X empty!')
      if not isinstance(X[0], list) or not isinstance(X[0][0], list) or not isinstance(X[0][0][0], str):
         raise ValueError('X invalid format!')
      if not y.any() or y.shape[0] == 0:
         raise ValueError('y invalid format!')
      if not isinstance(filtering, int) or filtering <= 0:
         filtering = 'all'
      self.features = np.asarray([])
      save_obj((filtering, ngrams), os.path.join(self.savingPath, featureConfigFileName))
      X_ngrams = [' '.join([' '.join(['_'.join(ngram) for ngram in self.getNgrams(sentence, ngrams[0], ngrams[1])]) for sentence in doc]) for doc in X]
      X_aux_ngrams = [' '.join([' '.join(['_'.join(ngram) for ngram in self.getNgrams(sentence, ngrams[0], ngrams[1])]) for sentence in doc]) for doc in X_aux]
      if not os.path.exists(os.path.join(self.savingPath, tfModelFileName)):
         countVectorizer = CountVectorizer()
         X_joint_aux = countVectorizer.fit(X_ngrams + X_aux_ngrams)
         save_obj(countVectorizer, os.path.join(self.savingPath, tfModelFileName))
      else:
         countVectorizer = load_obj(os.path.join(self.savingPath, tfModelFileName))
      X_joint_aux = countVectorizer.transform(X_ngrams)
      X_aux_ = countVectorizer.transform(X_aux_ngrams)
      for i in range(len(labels)):
         start = time.time()
         ns = negativeSample(n, i, labels, X_aux_, y_aux)
         X_joint = sp.sparse.vstack([X_joint_aux.copy()] + [X_aux_[d] for d in range(X_aux_.shape[0]) if y_aux[d][i]==1] + ns)
         y_joint = np.vstack([y.copy()] + [y_aux[d] for d in range(y_aux.shape[0]) if y_aux[d][i]==1] + [np.zeros(y_aux[0].shape, dtype=int) for e in range(len(ns))])
         if filtering != 'all':
            if not os.path.exists(os.path.join(self.savingPath, labels[i][0:180] + '_' + selectKBestModelName)):
               selectKBest = SelectKBest(chi2, k=filtering)
               X_joint = selectKBest.fit_transform(X_joint, y_joint[:, i])
               save_obj(selectKBest, os.path.join(self.savingPath, labels[i][0:180] + '_' + selectKBestModelName))
            else:
               if not os.path.exists(os.path.join(self.savingPath, labels[i][0:180] + '_' + weighTfModelName + '_')):
                  selectKBest = load_obj(os.path.join(self.savingPath, labels[i][0:180] + '_' + selectKBestModelName))
                  X_joint = selectKBest.transform(X_joint)
         if not os.path.exists(os.path.join(self.savingPath, labels[i][0:180] + '_' + weighTfModelName + '_')):
            bnsTransformer = BnsTransformer()
            bnsTransformer.fit(X_joint, y_joint[:,i])
            save_obj(bnsTransformer, os.path.join(self.savingPath, labels[i][0:180] + '_' + weighTfModelName + '_'))
         end = time.time()
         print('Features', i + 1, len(labels), labels[i][0:180], end - start, str(X_joint_aux.shape[0]) + ':' + str(X_joint.shape[0]))
   
   # Transform all the instances into TF-BNS representations
   def transformTFBNS(self, X, labels):
      X_bsn = list()
      for i in range(len(labels)):
         start = time.time()
         X_bsn.append(self.transformTFBNS_i(X, labels[i], usingCache=True))
         end = time.time()
         print('Features', i + 1, len(labels), labels[i][0:180], end - start)
      return X_bsn
   
   # Transform only one instance into TF-BNS representation
   def transformTFBNS_i(self, X, label, usingCache=False):
      if not os.path.exists(os.path.join(self.savingPath, featureConfigFileName)):
         raise ValueError('Configuration file unlocated!')
      filtering, ngrams = load_obj(os.path.join(self.savingPath, featureConfigFileName))
      if not usingCache or self.features.shape[0] == 0:
         if not X or len(X) == 0:
            raise ValueError('X empty!')
         if not isinstance(X[0], list) or not isinstance(X[0][0], list) or not isinstance(X[0][0][0], str):
            raise ValueError('X invalid format!')
         if not isinstance(filtering, int) or filtering <= 0:
            filtering = 'all'
         countVectorizer = load_obj(os.path.join(self.savingPath, tfModelFileName))
         X_joint_aux = countVectorizer.transform([' '.join([' '.join(['_'.join(ngram) for ngram in self.getNgrams(sentence, ngrams[0], ngrams[1])]) for sentence in doc]) for doc in X])
         if usingCache:
            self.features = X_joint_aux
      else:
         X_joint_aux = self.features
      X_joint = X_joint_aux.copy()
      if filtering != 'all':
         selectKBest = load_obj(os.path.join(self.savingPath, label[0:180] + '_' + selectKBestModelName))
         X_joint = selectKBest.transform(X_joint)
      bnsTransformer = load_obj(os.path.join(self.savingPath, label[0:180] + '_' + weighTfModelName + '_'))
      X_bsn_i = bnsTransformer.transform(X_joint)
      return X_bsn_i
   
   # Delete all learned representations
   def resetCache(self):
      self.features = np.asarray([])
