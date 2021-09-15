#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, string, re, pickle
from nltk.stem import SnowballStemmer
from unidecode import unidecode
from collections import Counter
from utils_module import *
from compound_module import *

NEGATIVE_PARTICLES = ['no', 'sin', 'ausencia', 'falta', 'ninguno', 'ningunos', 'ninguna', 'ningunas', 'tampoco', 'opuesto', 'contrario', 'ni']
UPPER_N = 'Ñ'
LOWER_N = 'ñ'
UPPER_C = 'Ç'
LOWER_C = 'ç'

# Class for storing word transformations
class traceableWord:
   def __init__(self, origin, word, start, end):
      self.origin = origin
      self.word = word
      self.start = start
      self.end = end

# Class containing a rule-based method for cleaning and transforming clinical texts
class NLP:
   def __init__(self, stemming=True, lemmatization=True):
      pathStopwords = os.path.join(os.path.dirname(__file__), 'data', 'stopwords.txt') # File with the stopword list
      pathStressedwords = os.path.join(os.path.dirname(__file__), 'data', 'stressedWords.txt') # File with the list of stressed words
      pathRelatedWords = os.path.join(os.path.dirname(__file__), 'data', 'relatedWordsFromIndiceAlfabetico.txt') # File with the related words from source 1
      pathRelatedAdjectives = os.path.join(os.path.dirname(__file__), 'data', 'relatedAdjectives.txt') # File with the related words from source 2
      pathCieAbsFreq  = os.path.join(os.path.dirname(__file__), 'data', 'cieStemFreqPerCategory.txt')  # File with word frequencies within CIE-10-ES
      pathLemmas  = os.path.join(os.path.dirname(__file__), 'data', 'extendedLemmas.txt') # File with the lemmas for each word
      # Define which method is preferred to reduce words
      self.stemWord_ = self.stemWord
      if stemming and lemmatization:
         self.stemWord_ = self.lemmatizeAndStemWord
      elif lemmatization:
         self.stemWord_ = self.lemmatizeWord
      # Load other components
      self.stemmer = SnowballStemmer('spanish')
      self.decomposer = wordDecomposer()
      #Define REGEX
      self.negativeParticles = NEGATIVE_PARTICLES
      self.simpleTokenReg = re.compile(r'[' + SEP_CHAR + SPACE + ']')
      self.tokenReg = re.compile(WORD_REGEX)
      self.lineSplitterReg = re.compile(r'[' + SEP_CHAR + ';]+|' + BEF_WORD_REGEX + ' *\.+ *' + AFT_UP_REGEX)
      self.shortReg = re.compile(r'^[' + WORD_LOW_C_CHAR + WORD_LOW_R_CHAR + ']{1,2}$')
      # Load all files
      with open(pathStopwords, encoding='utf8') as f: self.stopwords = re.split('[' + SEP_CHAR + ']', f.read())
      with open(pathStressedwords, encoding='utf8') as f: self.stressedwords = {group.split(TAB)[0]:group.split(TAB)[1] for group in re.split(SEP_LINE, f.read())}
      with open(pathRelatedWords, encoding='utf8') as f: relatedAdjectives = [list(sorted(group.split(TAB), key=self.len_stem, reverse=True)) for group in re.split(SEP_LINE, f.read())]
      with open(pathRelatedAdjectives, encoding='utf8') as f: relatedWords = [list(sorted(group.split(TAB), key=self.len_stem, reverse=True)) for group in re.split(SEP_LINE, f.read())]
      with open(pathCieAbsFreq, encoding='utf8') as f: self.cieAbsFreq = Counter({line.strip().split(TAB)[0]:float(line.strip().split(TAB)[1]) for line in re.split(SEP_LINE, f.read())})
      self.stemmedStopwords = [self.stemWord_(word) for word in self.stopwords]
      # Build related word dictionary
      relatedWords.extend(relatedAdjectives)
      relatedWords = [[terms.replace('_', ' ') for terms in group] for group in relatedWords if group != ['']]
      indicesEmptyRelatedWords = {}
      relatedWords = [[[word.word for word in wordsInSentence] for wordsInSentence in self.normalizeWordsInSentences(relatedWords[i], stripAcc=True, stripPunct=True, lowercase=True, stripNum=False, maskNum=False, removeParenthesis=True, removeExpressions=False, dealWithNegation=True, groupRelatedwords=False, removeStopwords=False, divideCompoundWords=False)] for i in range(len(relatedWords)) if i not in indicesEmptyRelatedWords]
      self.relatedWords = self.buildDict(relatedWords)
   
   # Function to normalise the frequency of ICD codes
   def getCieRelevance(self, word, parameter=532):
      f = self.cieAbsFreq[word]
      if f > parameter:
         return 0
      else:
         return (parameter - f) / parameter
   
   # Get the length of the stemmed word
   def len_stem(self, word):
      return len(self.stemWord_(word))
   
   # Remove all accent marks within words
   def stripAccents(self, sentence):
      nLocations = [i.start() for i in re.finditer(LOWER_N, sentence)]
      NLocations = [i.start() for i in re.finditer(UPPER_N, sentence)]
      cLocations = [i.start() for i in re.finditer(LOWER_C, sentence)]
      CLocations = [i.start() for i in re.finditer(UPPER_C, sentence)]
      sentence_ = unidecode(sentence)
      if nLocations or NLocations or cLocations or CLocations:
         words_ = list(sentence_)
         for nl in nLocations:
            words_[nl] = LOWER_N
         for nl in NLocations:
            words_[nl] = UPPER_N
         for nl in cLocations:
            words_[nl] = LOWER_C
         for nl in CLocations:
            words_[nl] = UPPER_C
         sentence_ = ''.join(words_)
      return sentence_
   
   # Tokenize and transform all words in a sentece: this is a process adapted to the clinical domain
   def preprocessWords(self, sentence, stripAcc=True, stripPunct=False, lowercase=True, stripNum=False, maskNum=True, removeParenthesis=False, removeExpressions=False):
      # Tokenize
      sentence = re.sub('[^' + WORD_PART + NWORD_PART + ']', lambda m: ' ' * len(m.group()), sentence)
      sentence = re.sub(r'([' + WORD_PART + '])[' + WORD_PUNCT + ']([^' + WORD_PART + ']|$)', r'\1 \2', sentence)
      if lowercase: # Remove capital letters
         sentence = sentence.lower()
      if stripAcc: # Remove accent marks
         sentence = self.stripAccents(sentence)
      if removeParenthesis: # Remove parentheses
         sentence = re.sub(r'\([^\)]+\)', lambda m: ' ' * len(m.group()), sentence)
         sentence = re.sub(r'\[[^\]]+\]', lambda m: ' ' * len(m.group()), sentence)
         sentence = re.sub(r'\{[^\}]+\}', lambda m: ' ' * len(m.group()), sentence)
      if removeExpressions: # Remove other expressions
         pass # TODO
      tokenList = list()
      beforePoint = 0
      for m in re.compile(WORD_REGEX).finditer(sentence): # Apply word-level transformations
         if not stripPunct:
            punctuationList = list(sentence[beforePoint:m.start(0)])
            for p in range(len(punctuationList)):
               if punctuationList[p] != SPACE: # Add punctuation marks
                  tokenList.append(traceableWord(punctuationList[p], punctuationList[p], beforePoint, p))
         # Identify and transform numerical terms
         beforePoint = m.end(0)
         numMatching = re.findall(SUBEXP_NUM_REGEX, m.group(0))
         numMatching = [match for match in numMatching if not re.match('^' + '[' + NUM_CHAR + ']' + '$', match)]
         if (stripNum or maskNum) and numMatching:
            if maskNum:
               word = m.group(0)
               for match in numMatching:
                  word = word.replace(match, MASK)
               tokenList.append(traceableWord(m.group(0), word, m.start(), m.end()))
            else:
               word = m.group(0)
               for match in numMatching:
                  word = word.replace(match, '')
               if word.strip():
                  tokenList.append(traceableWord(m.group(0), word, m.start(), m.end()))
         else:
            if not maskNum or m.group(0) != MASK:
               tokenList.append(traceableWord(m.group(0), m.group(0), m.start(), m.end()))
      if not stripPunct:
         punctuationList = list(sentence[beforePoint:len(sentence)])
         for p in range(len(punctuationList)):
            if punctuationList[p] != SPACE:
               tokenList.append(traceableWord(punctuationList[p], punctuationList[p], beforePoint, p))
      return tokenList
   
   #Detect denied entities and transform
   def manageNegation(self, wordsInSentences):
      for i in range(len(wordsInSentences)):
         for j in reversed(range(len(wordsInSentences[i]))):
            if wordsInSentences[i][j].word.lower() in self.negativeParticles:
               j_ = j + 1
               for j_ in range(j + 1, len(wordsInSentences[i])):
                  if self.stemWord_(wordsInSentences[i][j_].word) not in self.stopwords:
                     break
               if j_ >= len(wordsInSentences[i]):
                  wordsInSentences[i] = wordsInSentences[i][0:len(wordsInSentences[i])-1] + [traceableWord(wordsInSentences[i][j].origin, 'neg', wordsInSentences[i][j].start, wordsInSentences[i][j].end)]
               else:
                  newWord = traceableWord(wordsInSentences[i][j_].origin, wordsInSentences[i][j_].word, wordsInSentences[i][j].start, wordsInSentences[i][j_].end)
                  if not newWord.word.startswith('neg_'):
                     newWord.origin = wordsInSentences[i][j].origin + ' ' + wordsInSentences[i][j_].origin
                     newWord.word = 'neg_' + wordsInSentences[i][j_].word
                  wordsInSentences[i] = wordsInSentences[i][0:j] + wordsInSentences[i][j+1:j_] + [newWord] + wordsInSentences[i][j_+1::]
   
   # Normalise words in sentences
   def normalizeWordsInSentences(self, sentences, stripAcc=True, stripPunct=True, lowercase=True, stripNum=True, maskNum=True, removeParenthesis=True, removeExpressions=True, dealWithNegation=True, groupRelatedwords=True, removeStopwords=True, divideCompoundWords=False, removeShortwords=True, expressionToRemove={}):
      wordsInSentences = [self.preprocessWords(sentence, stripAcc=stripAcc, stripPunct=stripPunct, lowercase=lowercase, stripNum=stripNum, maskNum=maskNum, removeParenthesis=removeParenthesis, removeExpressions=removeExpressions) for sentence in sentences]
      if expressionToRemove: # Remove all words within the list of unnecessary expressions
         expressionToRemove = {tag:[[traceableWord(word.origin, self.stemWord_(word.word), word.start, word.end) for word in self.preprocessWords(sentence, stripAcc=stripAcc, stripPunct=stripPunct, lowercase=lowercase, stripNum=stripNum, maskNum=maskNum, removeParenthesis=removeParenthesis, removeExpressions=removeExpressions)] for sentence in sentences] for tag,sentences in expressionToRemove.items()}
         dictToReplace = [self.buildDict([sentences], tag) for tag,sentences in expressionToRemove.items()]
         for expressions in dictToReplace:
            self.replaceSequences(wordsInSentences, expressions, stemming=True)
      if divideCompoundWords: # Decompose words in terms of affixes
         for i in range(len(wordsInSentences)):
            for j in reversed(range(len(wordsInSentences[i]))):
               compounds = self.decomposer.divideCompoundWord(wordsInSentences[i][j].word)
               if len(compounds) > 1:
                  for compound in reversed(compounds[1::]):
                     wordsInSentences[i].insert(j + 1, traceableWord(wordsInSentences[i][j].origin, compound, wordsInSentences[i][j].start, wordsInSentences[i][j].end))
                  wordsInSentences[i][j].word = compounds[0]
         if stripAcc:
            for i in range(len(wordsInSentences)):
               for j in range(len(wordsInSentences[i])):
                  wordsInSentences[i][j].word = self.stripAccents(wordsInSentences[i][j].word)
      if dealWithNegation: # Detect negation
         self.manageNegation(wordsInSentences)
      if groupRelatedwords: # Unify related words
         self.replaceSequences(wordsInSentences, self.relatedWords)
      if removeStopwords: # Remove stopwords
         wordsInSentences = [[words[w] for w in range(len(words)) if (words[w].word not in self.stopwords or (words[w].word.isupper() and (w != 0 or len(words[w].word) > 1)))] for words in wordsInSentences]
      if removeShortwords: # Eliminate very short words
         wordsInSentences = [[words[w] for w in range(len(words)) if (not self.shortReg.match(words[w].word) or words[w].word == 'neg')] for words in wordsInSentences]
      return wordsInSentences
   
   # Replace sequences of terms with other sequences
   def replaceSequences(self, wordsInSentences, replacements, stemming=False):
      for i in range(len(wordsInSentences)):
         candidates, changes = list(), list()
         for w in range(len(wordsInSentences[i])):
            indicesCandidatesToRemove, word = list(), self.stemWord_(wordsInSentences[i][w].word.lower()) if stemming else wordsInSentences[i][w].word.lower()
            for c in range(len(candidates)):
               if word in candidates[c][0]:
                  candidates[c] = (candidates[c][0][word], candidates[c][1])
                  if '' in candidates[c][0]:
                     changes.append((candidates[c][0][''], candidates[c][1], w + 1))
               else:
                  if '' in candidates[c][0]:
                     changes.append((candidates[c][0][''], candidates[c][1], w))
                  indicesCandidatesToRemove.append(c)
            for c in reversed(indicesCandidatesToRemove):
               del candidates[c]
            if word in replacements:
               candidates.append((replacements[word], w))
         for c in range(len(candidates)):
            if '' in candidates[c][0]:
               changes.append((candidates[c][0][''], candidates[c][1], w+1))
         offset = len(wordsInSentences[i]) + 1
         for c in reversed(range(len(changes))):
            if changes[c][1] < offset:
               changes_ = changes[c][0].split('_')
               if changes_[0] == 'neg' and len(changes_) > 1:
                  changes_ = ['neg_' + changes_[1]] + changes_[2::]
               wordsInSentences[i][changes[c][1]:changes[c][2]] = [traceableWord(wordsInSentences[i][changes[c][1]].origin, change, wordsInSentences[i][changes[c][1]].start, wordsInSentences[i][changes[c][1]].end) for change in changes_]
               offset = changes[c][1]
         wordsInSentences[i] = [w_ for w in [word if isinstance(word, list) else [word] for word in wordsInSentences[i]] for w_ in w]
   
   # Function for lemmatizing first and then applying stemming
   def lemmatizeAndStem(self, word):
      if word in self.decomposer.splittedLemmas:
         return self.decomposer.getMostProbLemma(word)
      else:
         return self.stemmer.stem(word)
   
   # Lemmatize words according to customised list
   def lemmatize(self, word):
      if word in self.decomposer.splittedLemmas:
         return self.decomposer.getMostProbLemma(word)
      else:
         return word
   
   # Lemmatize all words in a sentence
   def lemmatizeWord(self, word):
      word_ = [unidecode(self.lemmatize(w_.lower())) for w_ in word.split('_')]
      return '_'.join(word_)
   
   # Lemmatize and apply stemming to all words in a sentence
   def lemmatizeAndStemWord(self, word):
      word_ = [unidecode(self.lemmatizeAndStem(w_.lower())) for w_ in word.split('_')]
      return '_'.join(word_)
   
   # Apply stemming
   def stemWord(self, word):
      word_ = unidecode(word.lowe()).split('_')
      if word_[-1] in self.stressedwords:
         word_[-1] = self.stressedwords[word_[-1]]
      return self.stemmer.stem('_'.join(word_))
   
   # Apply the preprocessing pipeline to all words in each sentence
   def preprocessSentences(self, sentences, stripAcc=True, stripPunct=True, lowercase=True, stripNum=False, maskNum=False, removeParenthesis=True, removeExpressions=True, dealWithNegation=True, groupRelatedwords=True, removeStopwords=True, divideCompoundWords=False, stemming=True, removeShortwords=True, expressionToRemove={}):
      wordsInSentences = self.normalizeWordsInSentences(sentences, stripAcc=stripAcc, stripPunct=stripPunct, lowercase=lowercase, stripNum=stripNum, maskNum=maskNum, removeParenthesis=removeParenthesis, removeExpressions=removeExpressions, dealWithNegation=dealWithNegation, groupRelatedwords=groupRelatedwords, removeStopwords=removeStopwords, divideCompoundWords=divideCompoundWords, removeShortwords=removeShortwords, expressionToRemove=expressionToRemove)
      if stemming:
         for words in wordsInSentences:
            for tWord in words:
               tWord.word = self.stemWord_(tWord.word)
      return wordsInSentences
   
   # Apply the preprocessing pipeline to all words in each sentence in multiple documents
   def preprocessDocuments(self, documents, traceable=False, filter=lambda word:True, stripAcc=True, stripPunct=True, lowercase=True, stripNum=False, maskNum=False, removeParenthesis=True, removeExpressions=True, dealWithNegation=True, groupRelatedwords=True, removeStopwords=True, divideCompoundWords=False, stemming=True, removeShortwords=True, expressionToRemove={}):
      preprocessDocs = list()
      for sentences in documents:
         preprocessedSentences = self.preprocessSentences(sentences, stripAcc=stripAcc, stripPunct=stripPunct, lowercase=lowercase, stripNum=stripNum, maskNum=maskNum, removeParenthesis=removeParenthesis, removeExpressions=removeExpressions, dealWithNegation=dealWithNegation, groupRelatedwords=groupRelatedwords, removeStopwords=removeStopwords, divideCompoundWords=divideCompoundWords, stemming=stemming, removeShortwords=removeShortwords, expressionToRemove=expressionToRemove)
         newSentences = list()
         for sentence in preprocessedSentences:
            words = list()
            for word in sentence:
               if filter(word.word):
                  if traceable:
                     words.append(word)
                  else:
                     words.append(word.word)
            newSentences.append(words)
         preprocessDocs.append(newSentences)
      return preprocessDocs
   
   # Remove exception expressions to clean up SNOMED-CT descriptions
   def manageExceptions(self, sentences):
      exceptions = list()
      for s in range(len(sentences)):
         m = re.search(r'(?i)\b(excepto|excepci[oóòôö]n|salvo)([^' + string.punctuation + ']+|$)', sentences[s])
         if m:
            exceptions.append(m.group(2))
            sentences[s] = sentences[s][0:m.start()] + sentences[s][m.end()::]
         else:
            exceptions.append('')
      return sentences, exceptions
   
   # Extract sentence chunks given a document
   def getContextualWindowForDoc(self, doc, length, unique=False):
      newSentences, index = list(), list()
      for i in range(len(doc)):
         sentence_ = doc[i].split(' ')
         N = len(sentence_) - length + 1
         if N > 0:
            for j in range(N):
               ngram = ' '.join(sentence_[k:k+length])
               if not unique or ngram not in newSentences:
                  newSentences.append(ngram)
                  index.append(i)
         else:
            ngram = ' '.join(sentence_)
            if not unique or ngram not in newSentences:
               newSentences.append(ngram)
               index.append(i)
      return newSentences,index
   
   # Extract sentence chunks given multiple documents
   def getContextualWindowForDocs(self, docs, length, unique=False):
      newSentences, indexDoc, indexSentence = list(), list(), list()
      for i in range(len(docs)):
         for j in range(len(docs[i])):
            sentence_ = docs[i][j].split(' ')
            N = len(sentence_) - length + 1
            if N > 0:
               for k in range(N):
                  ngram = ' '.join(sentence_[k:k+length])
                  if not unique or ngram not in newSentences:
                     newSentences.append(ngram)
                     indexDoc.append(i)
                     indexSentence.append(j)
            else:
               ngram = ' '.join(sentence_)
               if not unique or ngram not in newSentences:
                  newSentences.append(ngram)
                  indexDoc.append(i)
                  indexSentence.append(j)
      return newSentences, indexDoc, indexSentence
   
   # Build a dictionary with word sequences
   def buildDict(self, words, token_=''):
      strippedWords = words.copy()
      words_ = dict()
      for g in range(len(words)):
         if token_:
            token = token_
         else:
            token = '_'.join(words[g][0])
         for e in range(len(words[g])):
            i = 0
            dicc_ = words_
            while i != len(strippedWords[g][e]):
               if strippedWords[g][e][i] not in dicc_:
                  dicc_[strippedWords[g][e][i]] = dict()
               if len(strippedWords[g][e]) - i == 1:
                  dicc_[strippedWords[g][e][i]][''] = token
               dicc_ = dicc_[strippedWords[g][e][i]]
               i += 1
      return words_