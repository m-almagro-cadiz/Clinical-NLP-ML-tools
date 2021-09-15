#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from sklearn.preprocessing import MultiLabelBinarizer
from utils_module import *

# Constants
MICRO_P = 'micro-P'
MICRO_R = 'micro-R'
MICRO_F = 'micro-F'
MICRO_SIM_P = 'micro-SIM_P'
MICRO_SIM_R = 'micro-SIM_R'
MICRO_SIM_F = 'micro-SIM_F'
MACRO_P = 'macro-P'
MACRO_R = 'macro-R'
MACRO_F = 'macro-F'
MACRO_SIM_P = 'macro-SIM_P'
MACRO_SIM_R = 'macro-SIM_R'
MACRO_SIM_F = 'macro-SIM_F'

# Class to evaluate classification tasks: hierarchical similarity and propensity values are added
class Metric:
   # Initialization: propensity is estimated based on training labels
   def __init__(self, goldstandard, y_train=None, propensity_score=False, A=0.55, B=1.5):
      self.propensity = dict()
      self.gold = dict()
      for docid,labels in goldstandard:
         if docid not in self.gold:
            self.gold[docid] = set()
         self.gold[docid].update(labels)
      self.gold = {key:list(values) for key,values in self.gold.items()}
      if propensity_score and y_train:
         self.propensity = self.compute_propensity(y_train, A, B)
   
   # Get the propensity value for a specific label
   def getPropensity(self, label):
      if label not in self.propensity:
         return 1
      else:
         return self.propensity[label]
   
   # Calculate the weight of each label based on its frequency in the dataset.
   def compute_propensity(self, y, A=0.55, B=1.5):
      mlb = MultiLabelBinarizer()
      y_bin = mlb.fit_transform(y)
      num_instances, _ = y_bin.shape
      freqs = np.ravel(np.sum(y_bin, axis=0))
      C = (np.log(num_instances)-1)*np.power(B+1, A)
      wts = 1.0 + C*np.power(freqs+B, -A)
      inv_propensity_scores_array = np.ravel(wts)
      propensity_scores = dict()
      for i in range(len(mlb.classes_)):
         propensity_scores[mlb.classes_[i]] = 1 - (1 / inv_propensity_scores_array[i])
      return propensity_scores
   
   # Get the confusion matrix
   def getConfussionInstances(self, label, docIds, predictions):
      partitions = [list(), list(), list(), list()]
      for i in range(len(predictions)):
         if label in predictions[i]:
            if label in self.gold[docIds[i]]:
               partitions[3].append(i)
            else:
               partitions[1].append(i)
         else:
            if label in self.gold[docIds[i]]:
               partitions[2].append(i)
            else:
               partitions[0].append(i)
      return partitions
   
   # Calculate base metrics per label
   def getScoresPerLabel(self, labelSet, docIds, predictions, confussion=True):
      hits = self.buildScoresPerLabelDic(labelSet, docIds, self.gold, predictions)
      scores = self.computeUnorderedScores(hits)
      if confussion:
         labels, matrix, simMatrix = list(), list(), list()
         for label,row in hits.items():
            labels.append(label)
            matrix.append([len(predictions) - row[0][0] - row[0][1], row[0][1], row[0][2], row[0][0]])
            simMatrix.append([len(predictions) - row[1][0] - row[1][1], row[1][1], row[1][2], row[1][0]])
         scores['confussion'] = (np.asarray(labels), np.asarray(matrix), np.asarray(simMatrix))
      return scores
   
   # Calculate the hits and fails for each label
   def buildScoresPerLabelDic(self, labelSet, docIds, gold, predictions):
      hits = dict()
      for label in labelSet:
         hits[label] = [[0, 0, 0], [0, 0, 0]]
      for i in range(len(predictions)):
         distributedPredictions = computeDistributedPredictions(predictions[i], gold[docIds[i]], -1)
         ditributedSimPredictions = computeWeightsForMaxAlignment(predictions[i], gold[docIds[i]], -1)
         negDistributedPredictions, negDitributedSimPredictions = 1 - distributedPredictions, 1 - ditributedSimPredictions
         for j in range(len(predictions[i])):
            if predictions[i][j] in hits:
               hits[predictions[i][j]][0][0] += distributedPredictions[j]
               hits[predictions[i][j]][1][0] += ditributedSimPredictions[j]
               hits[predictions[i][j]][0][1] += negDistributedPredictions[j]
               hits[predictions[i][j]][1][1] += negDitributedSimPredictions[j]
         negDistributedGold = 1 - computeDistributedPredictions(predictions[i], gold[docIds[i]], 1)
         negDitributedSimGold = 1 - computeWeightsForMaxAlignment(predictions[i], gold[docIds[i]], 1)
         for j in range(len(gold[docIds[i]])):
            if gold[docIds[i]][j] in hits:
               hits[gold[docIds[i]][j]][0][2] += negDistributedGold[j]
               hits[gold[docIds[i]][j]][1][2] += negDitributedSimGold[j]
      return hits
   
   # Estimate ranking metrics
   def computeOrderedScores(self, docIds, gold, predictions, getPropensity_=lambda label:1):
      scores = dict()
      ndcg, sim_ndcg = list(), list()
      for i in range(len(predictions)):
         ps_pred_array = np.asarray([getPropensity_(label) for label in predictions[i]])
         distributedPredictions = computeDistributedPredictions(predictions[i], gold[docIds[i]], -1) * ps_pred_array
         ditributedSimPredictions = computeWeightsForMaxAlignment(predictions[i], gold[docIds[i]], -1) * ps_pred_array
         ndcg.append(self.ndcg_at_k(distributedPredictions, len(distributedPredictions), method=0))
         sim_ndcg.append(self.ndcg_at_k(ditributedSimPredictions, len(distributedPredictions), method=0))
      scores['nDCG'] = sum(ndcg) / len(ndcg) * 100
      scores['SIM_nDCG'] = sum(sim_ndcg) / len(sim_ndcg) * 100
      return scores
   
   # Estimate set metrics
   def computeUnorderedScores(self, hits, getPropensity_=lambda label:1):
      scores = dict()
      tp, sim_tp, fp, sim_fp, fn, sim_fn, P, sim_P, R, sim_R, sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      for label,hit in hits.items():
         ps = getPropensity_(label)
         tp += hit[0][0] * ps
         sim_tp += hit[1][0] * ps
         fp += hit[0][1] * ps
         sim_fp += hit[1][1] * ps
         fn += hit[0][2] * ps
         sim_fn += hit[1][2] * ps
         d1 = (hit[0][0] + hit[0][1])
         sim_d1 = (hit[1][0] + hit[1][1])
         d2 = (hit[0][0] + hit[0][2])
         sim_d2 = (hit[1][0] + hit[1][2])
         if d1 > 0:
            P += ps * hit[0][0] / d1
         if sim_d1 > 0:
            sim_P += ps * hit[1][0] / sim_d1
         if d2 > 0:
            R += ps * hit[0][0] / d2
         if sim_d2 > 0:
            sim_R += ps * hit[1][0] / sim_d2
         sum += ps
      scores[MACRO_P] = 0
      scores[MACRO_R] = 0
      scores[MACRO_F] = 0
      if sum > 0:
         scores[MACRO_P] = 100 * P / sum
         scores[MACRO_R] = 100 * R / sum
      if (scores[MACRO_P] + scores[MACRO_R]) > 0:
         scores[MACRO_F] = 2 * scores[MACRO_P] * scores[MACRO_R] / (scores[MACRO_P] + scores[MACRO_R])
      scores[MACRO_SIM_P] = 0
      scores[MACRO_SIM_R] = 0
      scores[MACRO_SIM_F] = 0
      if sum > 0:
         scores[MACRO_SIM_P] = 100 * sim_P / sum
         scores[MACRO_SIM_R] = 100 * sim_R / sum
      if (scores[MACRO_SIM_P] + scores[MACRO_SIM_R]):
         scores[MACRO_SIM_F] = 2 * scores[MACRO_SIM_P] * scores[MACRO_SIM_R] / (scores[MACRO_SIM_P] + scores[MACRO_SIM_R])
      scores[MICRO_P] = 0
      scores[MICRO_R] = 0
      scores[MICRO_F] = 0
      scores[MICRO_SIM_P] = 0
      scores[MICRO_SIM_R] = 0
      scores[MICRO_SIM_F] = 0
      d1 = (tp + fp)
      sim_d1 = (sim_tp + sim_fp)
      d2 = (tp + fn)
      sim_d2 = (sim_tp + sim_fn)
      if d1 > 0:
         scores[MICRO_P] = 100 * tp / d1
      if d2 > 0:
         scores[MICRO_R] = 100 * tp / d2
      if (scores[MICRO_P] + scores[MICRO_R]) > 0:
         scores[MICRO_F] = 2 * scores[MICRO_P] * scores[MICRO_R] / (scores[MICRO_P] + scores[MICRO_R])
      if sim_d1 > 0:
         scores[MICRO_SIM_P] = 100 * sim_tp / sim_d1
      if sim_d2 > 0:
         scores[MICRO_SIM_R] = 100 * sim_tp / sim_d2
      if (scores[MICRO_SIM_P] + scores[MICRO_SIM_R]) > 0:
         scores[MICRO_SIM_F] = 2 * scores[MICRO_SIM_P] * scores[MICRO_SIM_R] / (scores[MICRO_SIM_P] + scores[MICRO_SIM_R])
      return scores
   
   # Calculate scores for each metric
   def computeScores(self, docIds, predictions, labels=np.zeros(0), order=np.zeros(0), k=[1, 5, 10], fixedK=True, gold={}):
      if not gold:
         gold = self.gold
      scores = dict()
      labelSet = set([label for label in labels])
      hits = self.buildScoresPerLabelDic(labelSet, docIds, gold, predictions)
      scores.update(self.computeUnorderedScores(hits))
      scores.update(self.computeOrderedScores(docIds, gold, predictions))
      if self.propensity:
         scores.update({name.split('-')[0] + '-' + 'PS_' + name.split('-')[1]:metric for name,metric in self.computeUnorderedScores(hits, getPropensity_=self.getPropensity).items()})
         scores.update({'Ps_' + name:metric for name,metric in self.computeOrderedScores(docIds, gold, predictions, getPropensity_=self.getPropensity).items()})
      if order.shape[0] > 0 and labels.shape[0] > 0:
         for k_i in k:
            predictions_k = [labels[order[i][0:k_i]] if fixedK or len(gold[i]) >= k_i else labels[order[i][0:len(gold[i])]] for i in range(order.shape[0])]
            hits_k = self.buildScoresPerLabelDic(labelSet, docIds, gold, predictions_k)
            scores.update({name + '@' + str(k_i):metric for name,metric in self.computeUnorderedScores(hits_k).items()})
            scores.update({name + '@' + str(k_i):metric for name,metric in self.computeOrderedScores(docIds, gold, predictions_k).items()})
            if self.propensity:
               scores.update({name.split('-')[0] + '-' + 'PS_' + name.split('-')[1] + '@' + str(k_i):metric for name,metric in self.computeUnorderedScores(hits_k, getPropensity_=self.getPropensity).items()})
               scores.update({'Ps_' + name + '@' + str(k_i):metric for name,metric in self.computeOrderedScores(docIds, gold, predictions_k, getPropensity_=self.getPropensity).items()})
      return scores
   
   # Calculate scores distributed in groups for each metric 
   def computeScorePerLabelGroups(self, groups, docIds, predictions, labels=np.zeros(0), order=np.zeros(0), k=[1, 5, 10]):
      scores = list()
      for g in range(len(groups)):
         gold_g = {docId:[code for code in codes if code in groups[g]] for docId,codes in self.gold.items() if docId in docIds}
         predictions_g = [[code for code in codes if code in groups[g]] for codes in predictions]
         scores_g = self.computeScores(docIds, predictions_g, labels=labels, order=order, k=k, gold=gold_g)
         scores.append((g, len(groups[g]), sum([len(codes) for codes in gold_g.values()]), scores_g))
      return scores
   
   # Calculate Precision with the first K relevance values
   def precision_at_k(self, r, k):
      assert k >= 1
      r = np.asarray(r)[:k] != 0
      if r.size != k:
         raise ValueError('Relevance score length < k')
      return np.mean(r)
   
   # Compute the average Precision score using relevance values
   def average_precision(self, r):
      r = np.asarray(r) != 0
      out = [self.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
      if not out:
         return 0.
      return np.mean(out)
   
   # Compute the mean of several average Precision scores
   def mean_average_precision(self, rs):
      return np.mean([self.average_precision(r) for r in rs])
   
   # Calculate the Discounted Cumulative Gain score at the first K relevance values
   def dcg_at_k(self, r, k, method=0):
      r = np.asfarray(r)[:k]
      if r.size:
         if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
         elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
         else:
            raise ValueError('method must be 0 or 1.')
      return 0.
   
   # Calculate the normalized Discounted Cumulative Gain score at the first K relevance values
   def ndcg_at_k(self, r, k, method=0):
      dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
      if not dcg_max:
         return 0.
      return self.dcg_at_k(r, k, method) / dcg_max
   
   # Compute pearson values
   def computePearson(correlations, predictions):
      return sp.stats.pearsonr(correlations, predictions)[0]
