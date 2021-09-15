#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle, io, os, re, random
import numpy as np
import scipy as sp
import networkx as nx
import networkx.algorithms.matching as matching
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

#Constants
SPACE = ' '
MASK = '#'
NUM_CHAR = '0-9'
NUM_PUNCT = '\,\.\-'
NUM_PUNCT_M = '\\\/\*\^\+\%'
WORD_UP_C_CHAR = 'A-ZÑÁÉÍÓÚ'
WORD_LOW_C_CHAR = 'a-zñáéíóú'
WORD_UP_R_CHAR = 'ÇÀÈÌÒÙÄËÏÖÜÂÊÎÔÛ\#'
WORD_LOW_R_CHAR = 'çàèìòùäëïöüâêîôû'
LETTER_CHAR = WORD_UP_C_CHAR + WORD_LOW_C_CHAR + WORD_UP_R_CHAR + WORD_LOW_R_CHAR
WORD_CHAR = NUM_CHAR + LETTER_CHAR
LETTER_PUNCT = '\@\~\_\&\-'
WORD_PUNCT = LETTER_PUNCT
NWORD_PUNCT = '\|\¡\!\¬\{\[\(\)\]\}\·\,\.\;\:\"\¿\?\=\'\`\ª\º\$\<\>\\\/\*\^\+\%'
SEP_LINE = '\n'
TAB = '\t'
SEP_CHAR = SEP_LINE + TAB + '\r'
SYMBOL_END_LINE = '\n\.\;\:\?\!'
NWORD_CHAR = SPACE + SEP_CHAR
LETTER_PART = LETTER_CHAR + LETTER_PUNCT
NUM_PART = NUM_CHAR + NUM_PUNCT
WORD_PART = WORD_CHAR + LETTER_PUNCT
NWORD_PART = NWORD_CHAR + NWORD_PUNCT
BEF_NNUM_REGEX = '(^|(?<=[^' + NUM_PART + ']))'
AFT_NNUM_REGEX = '((?=[^' + NUM_CHAR + '])|$)'
BEF_NWORD_REGEX = '(^|(?<=[^' + WORD_CHAR + ']))'
AFT_NWORD_REGEX = '((?=[^' + WORD_CHAR + '])|$)'
BEF_WORD_REGEX = '(?<=[' + WORD_PART + '])'
AFT_WORD_REGEX = '(?=[' + WORD_CHAR + '])'
AFT_UP_REGEX = '(?=[' + WORD_UP_C_CHAR + WORD_UP_R_CHAR + '])'
WORD_REGEX = BEF_NWORD_REGEX + '([' + NUM_PART + ']+[' + NUM_CHAR + ']|[' + WORD_CHAR + '][' + WORD_PART +']+[' + WORD_CHAR + ']|[' + WORD_CHAR + ']+)' + AFT_NWORD_REGEX
NUM_REGEX = BEF_NWORD_REGEX + '([' + NUM_PART + ']+[' + NUM_CHAR + ']|[' + NUM_CHAR + ']+)' + AFT_NWORD_REGEX
SUBEXP_NUM_REGEX = '([' + NUM_PART + LETTER_PUNCT + ']+[' + NUM_CHAR + ']|[' + NUM_CHAR + ']+)'
EXP_NUM_REGEX = BEF_NWORD_REGEX + SUBEXP_NUM_REGEX + AFT_NWORD_REGEX
SMALL_WORD_REGEX = BEF_NWORD_REGEX + '([' + NUM_PART + '][' + NUM_CHAR + ']|[' + WORD_PART +'][' + WORD_CHAR + ']|[' + WORD_CHAR + ']{1,2})' + AFT_NWORD_REGEX

# Save objects
def save_obj(obj, filePath):
   with open(filePath, 'wb') as f:
      pickle.dump(obj, f, protocol=2)

# Load stored objects
def load_obj(filePath):
   with open(filePath, 'rb') as f:
      return pickle.load(f)

# Calculate the similarity value between two codes
def getCodeSimilarity(code1, code2):
   similarityValue = 0
   ic1 = len(code1)
   ic2 = len(code2)
   ics = 0
   for i in range(1, min(len(code1), len(code2)) + 1):
      if code1[0:i] == code2[0:i]:
         ics = i
   if ics > 0:
      similarityValue = (2 * ics / (ic1 + ic2))
   return similarityValue

# Check if an array is empty
def check_if_empty(list_of_lists):
   for elem in list_of_lists:
      if elem:
         return False
   return True

# Check if an array is a sublist of another array
def check_if_sublist(list1, list2):
   for e in list1:
      if e not in list2:
         return False
   return True

# Calculate the costs of aligning elements of one set with the elements of another set
def computeWeightsForMaxAlignment(codes1, codes2, match=0):
   left = True
   if match == -1:
      len_ = len(codes1)
   elif match == 1:
      len_ = len(codes2)
      left = False
   else:
      len_ = max(len(codes1), len(codes2))
      if len(codes1) < len(codes2):
         left = False
   matrixSimilarityValues = list()
   G = nx.Graph()
   for c1 in range(len(codes1)):
      listSimilarityValues = list()
      for c2 in range(len(codes2)):
         G.add_edge(c1, len(codes1) + c2, weight = getCodeSimilarity(codes1[c1], codes2[c2]))
         listSimilarityValues.append(getCodeSimilarity(codes1[c1], codes2[c2]))
      matrixSimilarityValues.append(listSimilarityValues)
   matrixSimilarityValues = np.asarray(matrixSimilarityValues)
   M = matching.max_weight_matching(G, maxcardinality=True)
   weights = [0] * len_
   for pair in M:
      if pair[0] < len(codes1):
         c1 = pair[0]
         c2 = pair[1] - len(codes1)
      else:         
         c1 = pair[1]
         c2 = pair[0] - len(codes1)
      if left:
         weights[c1] = matrixSimilarityValues[c1, c2]
      else:
         weights[c2] = matrixSimilarityValues[c1, c2]
   return np.asarray(weights)

# Calculate the similarity between two sets
def getCodeSetSimilarity(codes1, codes2, match=0):
   if len_ > 0:
      weights = computeWeightsForMaxAlignment(codes1, codes2, match)
      return np.sum(weights) / len(weights)
   else:
      return 0

# Assess the overlap between predictions and the goldstandard
def computeDistributedPredictions(codes1, codes2, match=0):
   if match == -1:
      left = True
   elif match == 1:
      left = False
   else:
      if len(codes1) < len(codes2):
         left = False
      else:
         left = True
   if left:
      return np.asarray([1 if code in codes2 else 0 for code in codes1])
   else:
      return np.asarray([1 if code in codes1 else 0 for code in codes2])

# Infer missing parent nodes
def getExtendedPathCIE(label, pathDict):
   if label in pathDict:
      return pathDict[label]
   else:
      if '.' in label:
         path, label_ = list(), label
         while label_ not in pathDict and len(label_) > 3:
            label_ = label_[0:-1]
            if label_[-1] != '.':
               path.append(label_)
         if label_ in pathDict:
            path.extend(pathDict[label_])
         return path
      else:
         return list()

# Group rare codes with common parents to create superclasses
def groupCodes(classes, parents, constraints=(1,7), step=6):
   groups = dict()
   for code in classes:
      path = getExtendedPathCIE(code, parents)
      l = len(path) + 1
      if l > 1 and l >= constraints[0] and l <= constraints[1] and step > 0:
         if len(path) < step:
            group = path[-1]
         else:
            group = path[step - 1]
      else:
         group = code
      if group not in groups:
         groups[group] = list()
      groups[group].append(code)
   error = True
   while error:
      error = False
      wrongGroups = set()
      for group in groups:
         if group in parents:
            for parent in parents[group]:
               if parent in groups and (len(groups[parent]) != 1 or parent != groups[parent][0]):
                  wrongGroups.add(parent)
      if len(wrongGroups) > 0:
         error = True
      for wrongGroup in wrongGroups:
         codeToRemove = set()
         for code in groups[wrongGroup]:
            if code != wrongGroup:
               path = getExtendedPathCIE(code, parents)
               ind = path.index(wrongGroup) - 1
               group = path[ind]
               if ind < 0:
                  group = code
               if group not in groups:
                  groups[group] = list()
               groups[group].append(code)
               codeToRemove.add(code)
         if len(codeToRemove) == len(groups[wrongGroup]):
            del groups[wrongGroup]
         else:
            groups[wrongGroup] = [code for code in groups[wrongGroup] if code not in codeToRemove]
   grouped_classes = sorted([groupName for groupName,group in groups.items() if group])
   return grouped_classes, [' '.join(groups[groupName]) for groupName in grouped_classes]

# Generate new ICD parent nodes
def generateParentsByMatchingCharacters(y):
   path = dict()
   for codes in y:
      for code in codes:
         if code not in path:
            path[code] = list()
            for l in range(1, len(code)):
               if code[-l - 1] != '.':
                  path[code].append(code[0:-l])
            for l in range(len(path[code])):
               if path[code][l] not in path:
                  path[path[code][l]] = path[code][l+1::]
   return path

# Check the affiliation of the codes to each group
def checkGroupedCodes(y, groups):
   validation, correspondence = dict(), dict()
   for group in groups.values():
      for code in group:
         validation[code] = False
         correspondence[code] = [code_ for code_ in group if code_ != code]
   for y_i in y:
      for code in y_i:
         if validation[code] == False:
            only = True
            for label in correspondence[code]:
               if label in y_i:
                  only = False
            validation[code] = only
   independentCodes = [code for code,condition in validation.items() if not condition]
   newGroups = {k:[code for code in group if code not in independentCodes] for k,group in groups.items()}
   newGroups.update({code + '_':[code] for code in independentCodes})
   return newGroups

# Group codes by similar frequencies, creating N clusters with a similar percentage of impact
def groupCodesByFrequency(freqDictionary, N=8, error=0.00001):
   codes, frequency = zip(*[(k,v) for k, v in sorted(freqDictionary.items(), key=lambda item: item[1])])
   freq_bin = sum(frequency) / N
   margin = freq_bin * (1 + error)
   while margin < frequency[-1]:
      N -= 1
      freq_bin = sum(frequency) / N
      margin = freq_bin * (1 + error)
   groups, group, acc = list(), list(), 0
   for i in range(len(frequency)):
      if (acc + frequency[i]) > margin:
         groups.append(group)
         group = list()
         acc = 0
      else:
         group.append(codes[i])
         acc += frequency[i]
   if acc > 0:
      groups.append(group)
   Max_f = [max([freqDictionary[c] for c in g]) for g in groups]
   Min_f = [min([freqDictionary[c] for c in g]) for g in groups]
   N_instances = [sum([freqDictionary[c] for c in g]) for g in groups]
   N_labels = [len(g) for g in groups]
   print(N_instances)
   return groups, N_labels, N_instances, Min_f, Max_f

# Calculate cosine similarity
def computeCosineSimilarity(vector1, vector2):
   return 1 - sp.spatial.distance.cdist(vector1.reshape(1,-1), vector2.reshape(1,-1), 'cosine')[0][0]

# Save structured data in table format
def print_file(data, pathFile, lambda_=lambda row: row, sep=TAB, header=[]):
   with io.open(pathFile, 'w') as fout:
      if header:
         fout.write(sep.join(header) + SEP_LINE)
      for i in range(len(data)):
         row_ = lambda_(data[i])
         fout.write(sep.join([str(e) for e in row_]) + SEP_LINE)

# Generate a sample of neighbouring instances that do not match the class
def negativeSample(n, i, labels, X_aux_, y_aux):
   simValues, indices_aux, negativeSamp = dict(), dict(), list()
   for l in range(len(labels)):
      if i != l:
         v = getCodeSimilarity(labels[i], labels[l])
         if v not in simValues:
            simValues[v] = list()
         simValues[v].append(l)
   maxN = n * y_aux[:][i].sum()
   for simValues_,indices_ in sorted(simValues.items(), reverse=True):
      negInstances = {index_:y_aux[:][index_].sum() for index_ in indices_}
      value_ = maxN
      total_ = sum(negInstances.values())
      if total_ <= maxN:
         value_ = maxN - total_
         for k,v in negInstances.items():
            indices_aux[k] = v
      else:
         valueAvg_ = int(value_ / len(negInstances))
         negInstances_ = dict()
         for k,v in negInstances.items():
            if v > valueAvg_:
               if valueAvg_ > 0:
                  indices_aux[k] = valueAvg_
               negInstances_[k] = v - valueAvg_
            else:
               indices_aux[k] = v
         realValue_ = sum(negInstances.values()) - sum(negInstances_.values())
         if realValue_ != value_:
            ls = random.sample([lab for k,v in negInstances.items() for lab in [k] * v], value_ - realValue_)
            for lab in ls:
               if lab not in indices_aux:
                  indices_aux[lab] = 0
               indices_aux[lab] += 1
      maxN -= value_
      if maxN <= 0:
         break
   for d in range(y_aux.shape[0]):
      toDel = list()
      for index_ in indices_aux.keys():
         if y_aux[d][index_]==1:
            negativeSamp.append(X_aux_[d])
            if indices_aux[index_] == 1:
               toDel.append(index_)
            else:
               indices_aux[index_] -= 1
            break
      for d in toDel:
         del indices_aux[d]
      if len(indices_aux) == 0:
         break
   return negativeSamp
