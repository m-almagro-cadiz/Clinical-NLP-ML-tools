#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, tables, os, pickle
from unidecode import unidecode
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Class containing a heuristic method based on statistical information on word frequency for medical word decompositions
class wordDecomposer:
   def __init__(self):
      # Read stored data from files
      pathCieVocab = os.path.join(os.path.dirname(__file__), 'data', 'cieVocab.txt') # File with word frequencies within CIE-10-ES
      pathSnVocab = os.path.join(os.path.dirname(__file__), 'data', 'snVocab.txt') # File with word frequencies within SNOMED-CT
      pathTags = os.path.join(os.path.dirname(__file__), 'data', 'tags.txt') # File with the part-of-speech tags for each word
      pathLemmas = os.path.join(os.path.dirname(__file__), 'data', 'extendedLemmas.txt') # File with the lemmas for each word
      pathPrefixes = os.path.join(os.path.dirname(__file__), 'data', 'medicalPrefixes.txt') # File with the list of common medical prefixes
      pathOtherPrefixes = os.path.join(os.path.dirname(__file__), 'data', 'otherMedicalPrefixes.txt') # File with the list of rare medical prefixes
      pathSuffixes = os.path.join(os.path.dirname(__file__), 'data', 'medicalSuffixes.txt') # File with the list of common medical suffixes
      pathWordIndex = os.path.join(os.path.dirname(__file__), 'data', 'wordIndex.obj') # File with the list of word embedding files in which each word is found
      pathConceptNetEmbeddings = os.path.join(os.path.dirname(__file__), 'embeddings') # Folder with the word embedding files
      with open(pathCieVocab, encoding='utf8') as f: cieVocab = [word for word in re.split('[ \n\t]', f.read())] # Read word frequencies
      with open(pathSnVocab, encoding='utf8') as f: snVocab = [word for word in re.split('[ \n\t]', f.read())] # Read word frequencies
      with open(pathTags, 'r', encoding='utf8') as f: wordTag = {line.strip().split('\t')[0]:{element.split(' ')[0]:element.split(' ')[1] for element in line.strip().split('\t')[1::]} for line in re.split('\n', f.read())} # Read word POS tags
      with open(pathLemmas, 'r', encoding='utf8') as f: lemmaList = [line.strip().split('\t') for line in re.split('\n', f.read())] # Read lemmas
      with open(pathPrefixes, encoding='utf8') as f: medicalPrefixes = {prefix.split('\t')[0].lower().replace('-', '').strip():prefix.strip().split('\t')[1::] if len(prefix.split('\t')) > 1 else '' for prefix in re.split('\n', f.read()) if prefix} # Read common prefixes
      with open(pathOtherPrefixes, encoding='utf8') as f: otherMedicalPrefixes = {prefix.split('\t')[0]:prefix.strip().split('\t')[1::] if len(prefix.split('\t')) > 1 else '' for prefix in re.split('\n', f.read()) if prefix} # Read rare prefixes
      with open(pathSuffixes, encoding='utf8') as f: medicalSuffixes = {prefix.split('\t')[0].lower().replace('-', '').strip():prefix.strip().split('\t')[1::] if len(prefix.split('\t')) > 1 else '' for prefix in re.split('\n', f.read()) if prefix} # Read suffixes
      with open(pathWordIndex, 'rb') as f: wordIndex = pickle.load(f) # Read word-file associations
      # Count words from CIE-10-ES and SNOMED-CT
      freq_aux = {}
      for word in cieVocab + snVocab:
         word = word.lower()
         if word not in freq_aux:
            freq_aux[word] = 0
         freq_aux[word] += 1
      freq = Counter(freq_aux)
      # Build lemmas dictionary
      self.lemmas = dict()
      for line in lemmaList:
         key = line[0].split(' ')
         if len(key) == 1:
            key.append('')
         key = tuple(key)
         values, nToRem = line[1::], list()
         for v in range(len(values)):
            value = values[v].split(' ')
            if len(value) == 1:
               value.append('')
            value = tuple(value)
            values[v] = value
         if key not in self.lemmas:
            self.lemmas[key] = list()
         self.lemmas[key] = values
      # Extend tags from lemmas
      self.tagPriority = {'a':0, 'n':1, 'r':2, 'v':3, '':4}
      self.tagger = {word:{tag[0] for tag in wordTag[word]} for word in wordTag}
      for lemma in self.lemmas:
         if lemma[0] not in self.tagger:
            self.tagger[lemma[0]] = set()
         self.tagger[lemma[0]].add(lemma[1])
      # Collect derivational words without tags
      self.splittedLemmas, self.splittedTags = dict(), dict()
      for lemma in self.lemmas:
         if lemma[0]:
            if lemma[0] not in self.splittedLemmas:
               self.splittedLemmas[lemma[0]] = list()
               self.splittedTags[lemma[0]] = list()
            for i in range(len(self.lemmas[lemma])):
               if self.lemmas[lemma][i][0] not in self.splittedLemmas[lemma[0]]:
                  self.splittedLemmas[lemma[0]].append(self.lemmas[lemma][i][0])
                  self.splittedTags[lemma[0]].append(self.lemmas[lemma][i][1])
            word_ = unidecode(lemma[0]).lower()
            if word_ not in self.lemmas:
               if word_ not in self.splittedLemmas:
                  self.splittedLemmas[word_] = list()
                  self.splittedTags[word_] = list()
               if self.lemmas[lemma][i][0] not in self.splittedLemmas[word_]:
                  self.splittedLemmas[word_].append(self.lemmas[lemma][i][0])
                  self.splittedTags[word_].append(self.lemmas[lemma][i][1])
      # Gather word counts from CIE-10-ES and SNOMED-CT using lemmas
      freq_aux = dict()
      for w in freq:
         lemmas_ = [w]
         if w in self.splittedLemmas:
            lemmas_ = self.splittedLemmas[w]
         f = freq[w]
         for lemma_ in lemmas_:
            if lemma_ not in freq_aux:
               freq_aux[lemma_] = 0
            freq_aux[lemma_] += f
      self.freq = Counter(freq_aux)
      self.freqAcc = sum([self.freq[w] for w in self.freq])
      self.prefixesDict = dict()
      self.prefixesDict.update(medicalPrefixes)
      self.prefixesDict.update(otherMedicalPrefixes)
      self.prefixes = list(set(medicalPrefixes.keys()))
      self.suffixes = list(set(medicalSuffixes.keys()))
   
   # Get lemmas given a word
   def getLemmas(self, word):
      if word in self.splittedLemmas:
         lt_ = [(self.splittedLemmas[word][i], self.splittedTags[word][i]) for i in range(len(self.splittedLemmas[word]))]
         lemmas_, tags_ = zip(*sorted(lt_, key=lambda tup: len(tup[0])))
         return list(lemmas_), list(tags_)
      else:
         return [word], ['']
   
   # Estimate the lexical similarity between two words
   def getWordSimilarityValue(self, word1, word2):
      similarityValue = 0
      ic1 = len(word1)
      ic2 = len(word2)
      ics = 0
      for i in range(1, min(len(word1), len(word2)) + 1):
         if word1[0:i] == word2[0:i]:
            ics = i
      if ics > 0:
         similarityValue = (2 * ics / (ic1 + ic2))
      return similarityValue
   
   # Get the most probable lemma given a word
   def getMostProbLemma(self, word):
      candidates, tagCandidates = list(self.getLemmas(word))
      if candidates:
         candidates = sorted([(candidates[i], self.getWordSimilarityValue(word, candidates[i]), tagCandidates[i], self.tagPriority[tagCandidates[i]]) for i in range(len(candidates))], key=lambda x : (-x[1], x[3]))
         candidate = candidates[0]
         if len(candidates) > 1 and candidate[2] == 'n' and candidates[1][2] == 'a' and abs(len(candidate[0]) - len(candidates[1][0])) < 2 and candidates[1][0].startswith(candidate[0][0:-1]):
            candidate = candidates[1]
         return candidate[0]
      else:
         return word
   
   # Get the absolute frequency given a word
   def getAbsFreq(self, word):
      lemmas_ = list()
      if word in self.splittedLemmas:
         lemmas_, tags_ = self.getLemmas(word)
      elif word not in self.freq:
         word_ = unidecode(word).lower()
         lemmas_, tags_ = self.getLemmas(word_)
      f = 0
      if lemmas_:
         for lemma_ in lemmas_:
            if lemma_ in self.freq:
               f += self.freq[lemma_]
      elif word in self.freq:
         f = self.freq[word]
      return f
   
   # Get the relative frequency given a word
   def getRelFreq(self, word):
      return self.getAbsFreq(word) / self.freqAcc
   
   # Get the tags given a word
   def getTags(self, word):
      tagsFromLemmas = set()
      if word in self.splittedTags:
         tagsFromLemmas = set(self.splittedTags[word])
      return self.tagger[word] - tagsFromLemmas if word in self.tagger else {}
   
   # Check if two words contain common tags
   def isSharedElement(self, tags1, tags2):
      shared = False
      for tag in tags1:
         if tag in tags2:
            shared = True
            break
      return shared
   
   # Get the possible separations given a word
   def getCandidateSubwords(self, word, prefixes, suffixes=[], reverse=False):
      possibilities = list()
      if len(word) > 3:
         word_ = ''.join(reversed(word)) if reverse else word
         i, subwords, subwords__ = 0, [[(list(), word_.lower())]], list()
         while i < len(prefixes):
            if len(subwords) == 1 or len(prefixes[i]) > 2:
               for subword in subwords[-1]:
                  if subword[1].startswith(prefixes[i]):
                     #Getting subword
                     subword_ = re.sub(r'^\W+', '', subword[1][len(prefixes[i])::]) #Deleting '-'
                     if subword_ and prefixes[i][-1] in ['a', 'e', 'i', 'o', 'u']: #Deleting duplicated r
                        subword_ = re.sub(r'^r(r)', r'\1', subword_)
                        if subword_ and len(prefixes[i]) > 1 and prefixes[i][-1] != subword_[0]: #Adding missing letter
                           subwords__.append((subword[0] + [prefixes[i]], prefixes[i][-1] + subword_))
                     startsWith_bp = prefixes[i][-1] != 'm' or (len(subword_) > 0 and subword_[0] in ['b', 'p'])
                     notStartsWith_bp = prefixes[i][-1] != 'n' or (len(subword_) > 0 and subword_[0] not in ['b', 'p'])
                     if (startsWith_bp and notStartsWith_bp) or reverse:
                        subwords__.append((subword[0] + [prefixes[i]], subword_))
            i += 1
            if i == len(prefixes) and len(subwords__) > 0:
               subwords.append(subwords__)
               i, subwords__ = 0, list()
         subwords = [([''.join(reversed(prefix)) for prefix in reversed(e[0])], ''.join(reversed(e[1]))) if reverse else (e[0], e[1]) for group in reversed(subwords[1::]) for e in group]
         wtag, fWord = self.getTags(word), self.getRelFreq(word) #Getting whole word tag and frequency
         for subword in subwords:
            fSubword = self.getRelFreq(subword[1])
            subwordLemmas_, tags_ = self.getLemmas(subword[1])
            notShort_ = max([len(lemma_) for lemma_ in subwordLemmas_]) > 3 if subwordLemmas_ else len(subword[1]) > 3
            inSuffixes_ = subword[1] in suffixes
            if (fSubword > 0 and notShort_):
               swtag = self.getTags(subword[1]) #Getting subword tag
               matchingTag = self.isSharedElement(wtag, swtag) #Comparing tags
               matching = len(wtag) == 0 or len(swtag) == 0 or matchingTag
               isVerb = ('v' in wtag and len(wtag) == 1) or ('v' in swtag and len(swtag) == 1)
               em = -1
               if word in self.wordIndex and subword[1] in self.wordIndex:
                  nb1 = pd.read_hdf(self.pathConceptNetEmbeddings + str(self.wordIndex[word][1]) + '.h5')
                  nb2 = pd.read_hdf(self.pathConceptNetEmbeddings + str(self.wordIndex[subword[1]][1]) + '.h5')
                  em = cosine_similarity([nb1.loc[self.wordIndex[word][0]].values, nb2.loc[self.wordIndex[subword[1]][0]].values])[0][1]
               elif subword[1] in self.wordIndex:
                  em = -2
               if fSubword == 0:
                  rate_ = -1
               else:
                  rate_ = fWord  / fSubword
               possibilities.append((subword, fWord, fSubword, rate_, em, 1 if matching else 0, 1 if isVerb else 0, len(subword[1]), len(subword[0]), len(''.join(subword[0])), 1 if inSuffixes_ else 0))
         possibilities = sorted(possibilities, key=lambda tup: (-tup[7], tup[8], tup[3], -tup[4]))
      return possibilities
   
   # Check if the possible separation satisfies the imposed restrictions
   def isCandidate(self, candidate):
      if (candidate[3] < 5) or (candidate[2] > 0.00001 and candidate[3] < 10):
         return True
      else:
         return False
   
   # Select the best choices (decompositions) for the corresponding word
   def selectCandidate(self, candidates):
      candidates_ = [candidate for candidate in candidates if self.isCandidate(candidate) and candidate[6] == 0]
      if len(candidates_) > 0:
         max_f_subword = max([candidate[2] for candidate in candidates_])
         min_l_subword = min([candidate[7] for candidate in candidates_ if 5 * candidate[2] > max_f_subword])
         min_n_prefix = min([candidate[8] for candidate in candidates_ if candidate[7] == min_l_subword])
         min_l_prefix = min([candidate[9] for candidate in candidates_ if candidate[8] == min_n_prefix])
         min_conditions = [1 if 5 * candidate[2] > max_f_subword and candidate[7] == min_l_subword and candidate[8] == min_n_prefix and candidate[9] == min_l_prefix else 0 for candidate in candidates_]
         conditions = [1 if min_conditions[i] == 1 and candidates_[i][6] == 0 and candidates_[i][2] > 0.0000002 and (candidates_[i][4] >= 0.4 or candidates_[i][4] <= -1) else 0 for i in range(len(candidates_))]
         candidates_ = [candidates_[i] for i in range(len(candidates_)) if conditions[i] == 1]
      return candidates_
   
   # Generate all possible decompositions of a word according to all the suffixes and prefixes
   def divideCompoundWord(self, word):
      candidates = self.getCandidateSubwords(word, self.prefixes, self.suffixes)
      compositions = self.selectCandidate(candidates)
      if len(compositions) > 0:
         return ' '.join([self.prefixesDict[prefix][0].replace('_', ' ') if len(self.prefixesDict[prefix]) == 1 else prefix for prefix in compositions[0][0][0]] + [compositions[0][0][1]]).split(' ')
      else:
         return [word]
   
   # Generate all possible decompositions for each word in a sentence
   def divideCompositions(self, sentences):
      return [[subword for word in sentence for subword in self.divideCompoundWord(word)] for sentence in sentences]
