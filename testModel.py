# author: Sowmya R
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import textutils as TU

''' ===== START TEST PARAM ==============='''
sentence = "myocardial infarction and profound vagal reaction"
model_path = 'best_model.pkl'
'''============= END TEST PARAM=============='''

def getWordTokens(text):
    # remove the special characters, number words, convert the string to lowercase and then tokenize
    # stop_words=_stop_words.ENGLISH_STOP_WORDS
    stop_words = list(set(stopwords.words('english')))
    cleanedText= TU.cleanText(text)
    tokens = word_tokenize(cleanedText, language="english", preserve_line=False)
    # remove the stop words from the token 
    if stopwords is not None:
        cleanedText = [w  for w in tokens if (w not in stop_words and not w.isdigit() and w != '-')]
    else:
        cleanedText = [w for w in tokens if (not w.isdigit() and w != '-')]
    return cleanedText

all_words = []
with open('inputfeatures.txt','r') as f:
    fileData = f.readlines()
    for line in fileData:
        all_words.append(line.replace('\n','').strip())

label_tokens = []
with open('labels.txt','r') as f:
    fileData = f.readlines()
    for line in fileData:
        label_tokens.append(line.replace('\n','').strip())

word_tokens = getWordTokens(sentence)
test_input = TU.Word2Vector([word_tokens], all_words)

clf_model = pickle.load(open(model_path, 'rb'))
pred = clf_model.predict(test_input)

# print the possible factors associated with the condition

print(TU.Vector2Word(pred[0], label_tokens))
