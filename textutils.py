import re
import numpy as np

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
numbers = '01234567890'
alphabet_lower = 'abcdefghijklmnopqrstuvwxyz'
alphabet_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def filterText (text, filter,hasFilter=False):
  # remove filter itmes from the string
  filter_txt = ""
  for char in text:
    if hasFilter==True and char in filter:
      filter_txt = filter_txt + char
    elif hasFilter==False and char not in filter:
        filter_txt = filter_txt + char
  return filter_txt

def remove_brackets(text):
  cleanedText= re.sub('\[\*\*.*\*\*\]|\\n|\s+', ' ', text).replace('  ', ' ')
  return cleanedText

def remove_specialChar(text):
  return  re.sub("[\\\"\[\]~!@#$%^&*()_+{}|`<>?/;':,]","",text)

# remove parantheses, punctuations and get only text values
def cleanText (text):
  #  cleaned = remove_brackets(text)
   cleaned = remove_specialChar(text)
   return cleaned.lower().strip()

def Word2Vector(tokens, vocabulary):
  return_vec = []
  for sent in tokens:
      row = list(np.zeros(len(vocabulary)))
      for i in range(len(vocabulary)):
          row[i] = 1 if vocabulary[i] in sent else 0
      return_vec.append(row)
  return return_vec

def Vector2Word(resultBool, vocabulary):
    if len(vocabulary)!=len(resultBool):
        return "Invalid comparision"
    results = []
    for i in range(len(vocabulary)):
        if resultBool[i]==1:
            results.append(vocabulary[i])
    return results

