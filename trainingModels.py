import pandas as pd
import numpy as np
import nltk

import textutils as TU
import clinicalDataUtils as CU

dataset = pd.read_csv("all_notes.csv")
# =========================================
# Download required nltk dataset. These download command can be comment after the first time. 
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# =====================================

# ====== Tokenize ================
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

# Get all input vocabulary list

allowed_word_types= ['NN','NNS','JJ']
noun_preceders = []
dataTokens = []
for index, row in dataset.iterrows():
    diagnosis = getWordTokens(row['diagnosis'])
    notesTokens = getWordTokens(str(row['complaint'])+" "+str(row["history"]))
    vocabulary = notesTokens
    vocabulary.extend(diagnosis)
    dataTokens.append(vocabulary)
    sTags =nltk.pos_tag(vocabulary)
    word_tag_pairs = nltk.ngrams(sTags,2)
    for (a, b) in word_tag_pairs:
        if b[1] in allowed_word_types and a[0]!=".":
            noun_preceders.append(a[0])
diag_freq = nltk.FreqDist(noun_preceders)
all_words = [k for k in list((dict(diag_freq)).keys())]


# get Label vocabulary
annotationTokens =[]
allowed_label_types= ["Reason","Drug","ADE"]
fdist = nltk.probability.FreqDist()
for index, row in dataset.iterrows():
    _ann = eval(row["annotations"])
    annotationTokens.append([i['pattern'] for i in _ann])
    if _ann["label"] in ["Reason","Drug","ADE"]:
        fdist[_ann['pattern']] = fdist[_ann['pattern']]+1
label_tokens = list(dict(fdist.most_common(200)).keys())  # get only top 200 words for label


print ("Number of labels = {}".format(len(label_tokens)))
print("Number of input Words = {}".format(len(all_words)))

# save the vocabulary and labels in a file for future use
with open("inputfeatures.txt",'w') as f:
    for w in all_words:
        f.write(w)
        f.write("\n")

with open("labels.txt",'w') as f:
    for w in label_tokens:
        f.write(w)
        f.write("\n")

# Word 2 Vector conversion
input_word_vec = TU.Word2Vector(dataTokens, all_words)
output_word_vec = TU.Word2Vector(annotationTokens, label_tokens)


# =============== Model building  START =======================
from sklearn.model_selection import train_test_split
from sklearn import metrics as MET

def evaluateModel(pred, y):
    print("coverage error = ", MET.coverage_error(pred, y))
    print("hamming loss = ", MET.hamming_loss(pred,y))

Xtrain,Xtest, ytrain, ytest = train_test_split(input_word_vec,output_word_vec, test_size=0.2, random_state=0)

# ANN
import Network as NN
testX = np.array(Xtest)
testy=np.array(ytest)

nnModel = NN.trainNNetwork(Xtrain,ytrain)
history = nnModel['history']
model =nnModel['model']
ypred = model.predict(testX)
ypred = np.array(ypred,dtype=int)
print("Results for ANN model: ")
evaluateModel(ypred,ytest)

# scikit multioutput models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

clf1 = MultiOutputClassifier(LogisticRegression()).fit(Xtrain,ytrain)
pred = clf1.predict(Xtest)
print("Results for MultiOutputClassifier(LogisticRegression()) model: ")
evaluateModel(pred,ytest)

clf2 = OneVsRestClassifier(SVC()).fit(Xtrain,ytrain)
pred = clf2.predict(Xtest)
print("Results for OneVsRestClassifier(SVC()) model: ")
evaluateModel(pred,ytest)

clf3 = OneVsRestClassifier(MultinomialNB()).fit(Xtrain,ytrain)
pred = clf3.predict(Xtest)
print("Results for OneVsRestClassifier(MultinomialNB()) model: ")
evaluateModel(pred,ytest)

clf4 = MultiOutputClassifier(BernoulliNB()).fit(Xtrain,ytrain)
pred = clf4.predict(Xtest)
print("Results for MultiOutputClassifier(BernoulliNB()) model: ")
evaluateModel(pred,ytest)

# ============= Model building END =================

# ================= Save the optimal model ====================
# After running the above experiments, cf3 seems to provide best results in comparison.
import pickle
pickle.dump(clf3, open('best_model.pkl','wb'))

# =========== END========================