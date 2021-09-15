#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string, os, json, csv, re, io, random, mimetypes, wikipedia
from utils_module import *
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.cmapdb import CMapDB
from pdfminer.layout import LAParams
from pdfminer.image import ImageWriter
from io import StringIO

# RegEx
LINE_REGEX = re.compile(r'(^|(?<=[' + SEP_CHAR + ';])) *([^' + SEP_CHAR + ';]+) *((?=[' + SEP_CHAR + ';])|$)')
DOT_REGEX = re.compile(r'( *\. *)([' + WORD_UP_C_CHAR + '] *[' + WORD_LOW_C_CHAR + '])')
UPLO_CHANGE_REGEX = re.compile(r'([' + WORD_UP_C_CHAR + ']+)([\. ]+)([' + WORD_UP_C_CHAR + '] *[' + WORD_LOW_C_CHAR + '])')
END_PAGE_REGEX = re.compile('[' + NWORD_CHAR + ']+[' + LETTER_PUNCT + NWORD_PUNCT + ' ]*[' + NUM_CHAR + ']+[' + LETTER_PUNCT + NWORD_PUNCT + ' ]*[' + NWORD_CHAR + ']+.*\x0c$')
ONE_JUMP_REGEX = re.compile('(?<!\n)\n|-\n')
FIRST_SENTENCE_REGEX = re.compile(r'^[^\n]+(?!u\n)')
LAST_SENTENCE_REGEX = re.compile(r'(?<!\n)[^\n]+$')

# Read text file
def readTxtFile(path):
   print('Reading ' + path)
   with open(path, 'r', encoding='utf8') as myfile:
      text = myfile.read()
   return text

# Read multiple files
def getTextFromFiles(directoryPath, n=-1, tokenize=True):
   docs, N = dict(), 0
   for (dirpath, dirnames, filenames) in os.walk(directoryPath):
      if dirpath == directoryPath:
         for filename in filenames:
            if N <= n or n < 0:
               print('File ' + str(N))
               id = filename.replace('.txt', '')
               docs[id] = list()
               print(os.path.join(dirpath, filename))
               text = readTxtFile(os.path.join(dirpath, filename))
               if tokenize:
                  text = tokenizeSentences(text)
               docs[id] = text
               N += 1
   return docs

# Transform the content from PDF files into text
def getPdfFromFiles(directoryPath, stopwords):
   N, COMPLETE_LEN = 8, 50
   FIRST_LOWER_REGEX = re.compile('^([^' + LETTER_CHAR + '])*([' + WORD_LOW_C_CHAR + WORD_LOW_R_CHAR + '])')
   UPPER_REGEX = re.compile('[' + WORD_UP_C_CHAR + WORD_UP_R_CHAR + ']')
   LOWER_REGEX = re.compile('[' + WORD_LOW_C_CHAR + WORD_LOW_R_CHAR + ']')
   LINE_JUMP_REGEX = re.compile('(?<!\n)\n')
   DASH_JUMP_REGEX = re.compile('-\n *')
   SPLIT_WNUM = re.compile('(?<=[' + LETTER_CHAR + ']{2})' + '([' + NUM_PART + ']+[' + NUM_CHAR + ']|[' + NUM_CHAR + ']+)' + AFT_NWORD_REGEX + '|'
                         + '\n' + '([' + NUM_PART + ']+[' + NUM_CHAR + ']|[' + NUM_CHAR + ']+)' + '[^' + WORD_CHAR + ']*(\n|$)')
   STOPWORD_END_REGEX = re.compile('[^' + WORD_PART + '](' + '|'.join(stopwords) + '|,) ?$')
   DASH_NONALPH_END_REGEX = re.compile('[^' + WORD_PART + NWORD_PART + ']$')
   docs = getTextFromFiles(directoryPath, n=-1, tokenize=False)
   for key,value in docs.items():
      text = re.sub(r'^\(#START_PAGE#\)|\(#END_PAGE#\)$', '', value)
      pages = text.split('(#END_PAGE#)(#START_PAGE#)')
      pages = replaceEmptyLines(pages)
      pages = replacePageNumbers(pages, N)
      pages = replaceHeaders(pages, N)
      reversed_pages = replaceHeaders([page[::-1] for page in pages], N)
      pages = [page[::-1] for page in reversed_pages]
      indices = list()
      for p in range(len(pages)):
         for match in SPLIT_WNUM.finditer(pages[p]):
            indices.append((p, match.start(), match.end()))
      for index in reversed(indices):
         pages[index[0]] = pages[index[0]][0:index[1]] + pages[index[0]][index[2]::]
      lines = [lines for page in pages for lines in page.split(SEP_LINE)]
      lineIndices, endsWithDash = list(), False
      for l in range(len(lines)):
         if endsWithDash:
            lineIndices.append(l)
         endsWithDash = lines[l].endswith('-') or len(DASH_NONALPH_END_REGEX.findall(lines[l])) > 0
      for ind in reversed(lineIndices):
         lines[ind - 1] = lines[ind - 1][0:-1] + lines[ind]
         del lines[ind]
      lineIndices, prevLen, prevNotEnd = list(), 0, list()
      for l in range(len(lines)):
         if FIRST_LOWER_REGEX.match(lines[l]) or len(prevNotEnd) > 0:
            lineIndices.append(l)
         elif UPPER_REGEX.search(lines[l]) and not LOWER_REGEX.search(lines[l]) and prevLen > COMPLETE_LEN:
            lineIndices.append(l)
         prevLen = len(lines[l])
         prevNotEnd = STOPWORD_END_REGEX.findall(lines[l])
      for ind in reversed(lineIndices):
         lines[ind - 1] += SPACE + lines[ind]
         del lines[ind]
      docs[key] = SEP_LINE.join(lines)
   return docs

# Replace empty lines
def replaceEmptyLines(pages):
   SWD_REGEX = re.compile(SMALL_WORD_REGEX)
   WD_REGEX = re.compile(WORD_REGEX)
   LINE_SEP_REGEX = re.compile(SEP_LINE)
   pages_ = list()
   for p in range(len(pages)):
      page_ = list()
      for line in LINE_SEP_REGEX.split(pages[p]):
         WN = WD_REGEX.findall(line)
         SWN = SWD_REGEX.findall(line)
         if not SWN or SWN < WN:
            page_.append(line)
      pages_.append(SEP_LINE.join(page_))
   return pages_

# Replace page numbers
def replacePageNumbers(pages, N):
   NPAGE_REGEX = re.compile('^[' + SEP_LINE + ']?([^' + SEP_LINE + ']+[' + SEP_LINE + ']+){0,' + str(N) + '}[' + LETTER_PUNCT + NWORD_PUNCT + ' ]*[' + NUM_CHAR + ']+[' + LETTER_PUNCT + NWORD_PUNCT + ' ]*([' + SEP_LINE + ']+|$)')
   numPageBefore, numPageAfter = 0, 0
   for page in pages:
      if NPAGE_REGEX.match(page):
         numPageBefore += 1
      if NPAGE_REGEX.match(page[::-1]):
         numPageAfter += 1
   for p in range(len(pages)):
      if numPageBefore > numPageAfter:
         pages[p] = NPAGE_REGEX.sub('', pages[p]).strip()
      else:
         pages[p] = NPAGE_REGEX.sub('', pages[p][::-1])[::-1].strip()
   return pages

# Replace headers
def replaceHeaders(pages, N):
   WD_REGEX = re.compile(WORD_REGEX)
   N_REGEX = re.compile(NUM_REGEX)
   LINE_SEP_REGEX = re.compile('(^|(?<=' + SEP_LINE + '))[^' + SEP_LINE + ']+(' + SEP_LINE + '+|$)')
   headerCandidates, beforeSentences = dict(), dict()
   for p in range(len(pages)):
      afterSentences = set()
      for match in LINE_SEP_REGEX.finditer(pages[p]):
         if len(afterSentences) >= N:
            break
         sentence = N_REGEX.sub('', match.group()).strip()
         if WD_REGEX.search(sentence):
            if sentence in beforeSentences:
               if sentence not in headerCandidates:
                  headerCandidates[sentence] = 1
               else:
                  headerCandidates[sentence] += 1
            afterSentences.add(sentence)
      beforeSentences_ = dict()
      for bs,val in beforeSentences.items():
         if val > 1:
            beforeSentences_[bs] = val - 1
      beforeSentences = beforeSentences_.copy()
      beforeSentences.update({afs:N for afs in afterSentences})
   replacements = list()
   for p in range(len(pages)):
      afterSentences, checkMatches = set(), list()
      for match in LINE_SEP_REGEX.finditer(pages[p]):
         if len(afterSentences) >= N:
            break
         sentence = N_REGEX.sub('', match.group()).strip()
         if sentence in headerCandidates and headerCandidates[sentence] >= 3 and all(checkMatches):
            checkMatches.append(True)
            replacements.append((p, match.start(), match.end()))
         else:
            checkMatches.append(False)
         afterSentences.add(sentence)
   for replacement in reversed(replacements):
      pages[replacement[0]] = pages[replacement[0]][0:replacement[1]] + pages[replacement[0]][replacement[2]::]
   return pages

# Split a document into sentences
def tokenizeSentences(text):
   sentences = list()
   text_ = DOT_REGEX.sub(lambda m: TAB * len(m.group(1)) + m.group(2), text)
   text_ = UPLO_CHANGE_REGEX.sub(lambda m: m.group(1) + TAB * len(m.group(2)) + m.group(3), text_)
   matches = [match for match in LINE_REGEX.finditer(text_)]
   for match in matches:
      if len(match.group(2)) > 0:
         sentences.append(match.group(2))
   return sentences

# Get features and labels
def getXY(docs, gold):
   X, y, ids = list(), list(), list()
   for id in docs:
      if id in gold:
         X.append(docs[id])
         y.append(gold[id])
         ids.append(id)
   return X, y, ids

# Read json files from pubmed
def readPubmedJson(filePath, tokenize=True):
   docs, N = dict(), 0
   with open(filePath, 'r', encoding='utf8') as f:
      data = json.load(f)
   for article in data['articles']:
      print('Doc ' + str(N))
      docs[article['pmid']] = list()
      text = article['abstractText']
      docs[article['pmid']] = text
      if tokenize:
         docs[article['pmid']] = tokenizeSentences(text)
      N += 1
   return docs

# Read csv files from pubmed
def readPubmedCsv(directoryPath, tokenize=True):
   csv.field_size_limit(2147483647)
   docs = dict()
   groupedElements = readCSVs(directoryPath, header=False)
   for elements in groupedElements:
      for element in elements:
         if element[1] == 'es':
            text = element[2] + '\n\n' + element[3]
            docs[element[0]] = text
            if tokenize:
               docs[element[0]] = tokenizeSentences(text)
   return docs

# Read scielo papers
def readScielo(filename, tokenize=True):
   docs = dict()
   lines = readTxtFile(filename).split(SEP_LINE)
   text = list()
   for line in lines:
      if line:
         if line[0] == ' ':
            text[-1] += '\n' + line
         else:
            text.append(line)
   for i in range(len(text)):
      docs[i] = text[i]
      if tokenize:
         docs[i] = tokenizeSentences(text[i])
   return docs

# Download wikipedia articles
def getWikipediaPages(category, tokenize=True):
   docs = dict()
   wikipedia.set_lang("es")
   subcategories = wikipedia.search(category, results=100000)
   for s in range(len(subcategories)):
      results = wikipedia.search('incategory:' + re.sub('^Categoría:', '', subcategories[s]), results=100000)
      for r in range(len(results)):
         print(category + '\t' + str(s) + '/' + str(len(subcategories)) + '\t' + str(r) + '/' + str(len(results)))
         if results[r] not in docs:
            try:
               page = wikipedia.page(results[r])
               docs[results[r]] = page.content
               if tokenize:
                  docs[results[r]] = tokenizeSentences(page.content)
            except wikipedia.exceptions.DisambiguationError:
               print('Disambiguation warning!')
            except:
               print('Other error')
   return docs

# Read CSV files
def readCSVs(directoryPath, header=True):
   csv.field_size_limit(2147483647)
   groupedElements, N = list(), 0
   for (dirpath, dirnames, filenames) in os.walk(directoryPath):
      if dirpath == directoryPath:
         for filename in filenames:
            print('File ' + str(N))
            rows = list(csv.reader(open(os.path.join(dirpath, filename), 'r', encoding='utf8'), delimiter='\t'))
            if header:
               rows = rows[1::]
            groupedElements.append(rows)
            N += 1
   return groupedElements

#Read CSV file
def readOneLineCsv(filePath, header=True):
   csv.field_size_limit(2147483647)
   content = list(csv.reader(open(filePath, 'r', encoding='utf8'), delimiter='\t'))
   idsDict = {id:list() for id in [e[0] for e in content]}
   for id,element in content:
      idsDict[id].append(element)
   return idsDict
