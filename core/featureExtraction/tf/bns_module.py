#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy as sp
from scipy.stats import norm
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

# Class to apply Bi-Normal Separation feature scoring to a sparse matrix of occurrence counts
class BnsTransformer(BaseEstimator, TransformerMixin):
   # Initialise the class
   def __init__(self, rate_range=(0.0005, 1 - 0.0005)):
      self.rate_range = ()
      self.poslabels = np.asarray([])
      self.neglabels = np.asarray([])
      self.pos = np.asarray([])
      self.neg = np.asarray([])
      self.bns_scores = np.asarray([])
   
   # Learn Bi-Normal Separation scores for input data
   def fit(self, X, y, rate_range=(0.0005, 1 - 0.0005)):
      self.rate_range = rate_range
      self.poslabels = y
      def f(a):
         if a is 0: return 1
         else: return 0
      f = np.vectorize(f)
      self.neglabels = f(np.copy(y))
      self.pos = np.sum(self.poslabels)
      self.neg = np.sum(self.neglabels)
      self.bns_scores = self.compute_bns(X.T)
      return self
   
   # Generate the output weighed with Bi-Normal Separation scores
   def transform(self, X):
      if not self.rate_range or not self.poslabels.any() or not self.neglabels.any() or not self.pos.any() or not self.neg.any() or not self.bns_scores.any():
         raise ValueError('Fit first!')
      return X.multiply(self.bns_scores)
   
   # Calculate Bi-Normal Separation scores and apply them to the input data
   def fit_transform(self, X):
      self.fit(X)
      return self.transform(X)
   
   # Apply the Bi-Normal Separation formula
   def compute_bns(self, X):
      tpr = X.dot(self.poslabels) / self.pos
      tnr = X.dot(self.neglabels) / self.neg
      tpr[tpr > self.rate_range[1]] = self.rate_range[1]
      tpr[tpr < self.rate_range[0]] = self.rate_range[0]
      tnr[tnr > self.rate_range[1]] = self.rate_range[1]
      tnr[tnr < self.rate_range[0]] = self.rate_range[0]
      bns_score = np.absolute(norm.ppf(tpr) - norm.ppf(tnr))
      return bns_score 

# Class to convert a collection of raw documents to a matrix of BNS features
class BnsVectorizer():
   # Initialise the class inherited from CountVectorizer
   def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True,
                preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                ngram_range=(1, 1), max_df=1.0, min_df=2, max_features=None, vocabulary=None, dtype=float):
      self.countvec = CountVectorizer(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents,
                                      lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
                                      stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df,
                                      min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=True, dtype=dtype)        
   
   # Learn conversion parameters from documents to array data
   def fit(self, raw_documents, y):
      X = self.countvec.fit_transform(raw_documents)     
      self.vocabulary_ = self.countvec.vocabulary_
      self._bns = BnsTransformer()
      self._bns.fit(X, y)
      return self
   
   # Learn the representation and return the vectors
   def fit_transform(self, raw_documents, y):
      X = self.countvec.fit_transform(raw_documents)
      self.vocabulary_ = self.countvec.vocabulary_
      self._bns = BnsTransformer()
      self._bns.fit(X, y)
      return self._bns.transform(X)
   
   # Transform raw text documents into bns vectors
   def transform(self, raw_documents):
      X = self.countvec.transform(raw_documents)
      return self._bns.transform(X)
