import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys
import string
import re
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print('Fryzlewicz, Sara, A20457833 solution:')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
ignore = False
argsPassed = sys.argv
rickVocab={}
mortyVocab={}
vocabDict={}
txt = ''
truePos = 0
falsePos = 0
trueNeg = 0
falseNeg = 0
RickProb = 0
MortyProb = 0


#text pre-processing
#returns list of words from list of sentences
def screaming(sent):
    x = sent
    if ('aaa' in x) or ('ooo' in x)or ('hhh' in x) or ('iii' in x)or ('uuu' in x):
        x=' scream '
    if ('-' in x) or ('www' in x and 'aw' not in x) or ('…' in x):
        x=' stutter '
    return x

def sprzataczka(sent):
    s = sent.split()
    pfid= [ps.stem(word) for word in s if word not in stop_words]
    sents=((screaming(ch)) for ch in pfid)
    ch=''
    sentence=''
    for word in sents:
        for punctuation in string.punctuation:
            word = word.replace(punctuation, '')
        sentence = sentence + ' ' + word
    if ignore==False:
        sentence.lower()
    return sentence
    

#separating rick vs morty vocab/text
def rickNMorty(speaker, sentence):
    for s in sentence.split():
        if speaker == 'Rick':
            if s not in rickVocab.keys():
                rickVocab[s]=1
            else:
                rickVocab[s]+=1
        else:
            if s not in mortyVocab.keys():
                mortyVocab[s]=1
            else:
                mortyVocab[s]+=1
        
        if s not in vocabDict.keys():
            vocabDict[s]=1
        else:
            vocabDict[s]+=1

def naive_bayes_classifier(txt, rickV, mortyV, rickProb, mortyProb):
    rickVocab=rickV
    mortyVocab=mortyV
    r=rickProb
    m=mortyProb
    #print(txt)
    for wrd in txt:
        if wrd not in rickVocab:
            rV=0
        else:
            rV=rickVocab[wrd]
        if wrd not in mortyVocab:
            mV=0
        else:
            mV=mortyVocab[wrd]
        if wrd not in vocabDict:
            vD=1
        else:
            vD=vocabDict[wrd]
        r = r * ((rV+1)/(vD+1)) #with smoothing
        m = m * ((mV+1)/(vD+1)) #with smoothing
    if r>m:
        label = 'Rick'
    else:
        label = 'Morty'
    return (r, m, label)

preProcessStep=''
#checking for user input
if (len(argsPassed)!=2):
    ignore = False
    preProcessStep= 'NONE'
else:
    ignore = True
    preProcessStep= 'LOWERCASING'
    

print('Ignored pre-processing step: '+ str(preProcessStep))


#Loading data
csv = pd.read_csv('RicknMorty.csv')
csv.drop(['Unnamed: 0','episode no.'], axis=1, inplace=True)
csv=csv[csv['speaker'].str.contains('Rick|Morty', na=False)]

#fixing someones ass-backwards data entry
csv = csv[csv["speaker"].str.contains("President") == False]
RnM = csv[csv.speaker == 'Rick'] #RnM == Rick and Morty data
x = csv[csv.speaker == 'Morty']
y = csv[csv.speaker == 'Rick:'].copy()
y.loc[y.speaker == 'Rick:', 'speaker'] = 'Rick' 
z = csv[csv.speaker == 'Morty:'].copy()
z.loc[z.speaker == 'Morty:', 'speaker'] = 'Morty' 
RnM = pd.concat([RnM, x, y, z], ignore_index=True)

#processing text, and dividing testing & training data 80/20
X_train, X_test, y_train, y_test = train_test_split(RnM['dialouge'], RnM['speaker'], test_size=0.2, random_state=0)

#cleaning data
X_train = pd.DataFrame(X_train)
X_train = X_train.reset_index(drop=True)
X_train=pd.concat([X_train.apply(lambda x: sprzataczka(x.dialouge), axis=1)], axis=1)
X_test = pd.DataFrame(X_test)
X_test = X_test.reset_index(drop=True)
X_test.apply(lambda x: sprzataczka(x.dialouge), axis=1)
y_test = pd.DataFrame(y_test)
y_test = y_test.reset_index(drop=True)
y_train = pd.DataFrame(y_train)
y_train = y_train.reset_index(drop=True)

RnM= pd.concat([y_train, X_train], axis=1)
RnM.columns = ['speaker', 'dialouge']
RnM_test = pd.concat([y_test, X_test], axis=1)
RnM_test.columns = ['speaker', 'dialouge']


RickProb = RnM['speaker'].value_counts()['Rick']/len(RnM)
MortyProb = RnM['speaker'].value_counts()['Morty']/len(RnM)

#training
print('Training classifier...')
RnM.apply(lambda x: rickNMorty(x.speaker, x.dialouge), axis=1)
trainResults= RnM_test.apply(lambda x: naive_bayes_classifier(x.dialouge, rickVocab, mortyVocab, RickProb, MortyProb), axis=1)


trainX=[]
trainY=[]
for rows in trainResults:
    if (rows[2]=='Rick'):
        trainY.append(1)
        trainX.append(rows[0])
    else:
        trainY.append(0)
        trainX.append(rows[1])
trainX = np.array(trainX)
trainX = np.reshape(trainX, (1, -1))
trainY = np.array(trainY)
trainY = np.reshape(trainY, (1,-1))



#testing
print('Testing classifier...')
testResults= RnM_test.apply(lambda x: naive_bayes_classifier(x.dialouge, rickVocab, mortyVocab, RickProb, MortyProb), axis=1)

testFP=[]
testTP=[]
count=0
        
for rows in testResults:
    if(y_test['speaker'][count]=='Rick'):
        if rows[2]=='Rick':
            truePos = truePos+1
        else:
            falsePos = falsePos + 1
    else:
        if rows[2]=='Morty':
            trueNeg = trueNeg+1
        else:
            falseNeg = falseNeg + 1
    count = count + 1


print('Test results/metrics:\n')
print('Number of true positives: ', truePos)
print('Number of true negatives: ', trueNeg)
print('Number of false positives: ', falsePos)
print('Number of false negatives: ', falseNeg)
precision = truePos/(truePos+falsePos)
recall = truePos/(truePos+falseNeg)
if truePos == 0:
    truePos = 1
if trueNeg == 0:
    trueNeg = 1
if falsePos == 0:
    falsePos = 1
if falseNeg == 0:
    falseNeg = 1
print('Sensitivity (recall): ', str(recall))
print('Specificity: ', str(trueNeg/(trueNeg+falsePos)))
print('Precision: ', str(precision) )
print('Negative predictive value: ', str(trueNeg/(trueNeg+falseNeg)))
print('Accuracy: ', str((truePos+trueNeg)/(truePos+trueNeg+falsePos+falseNeg)))
print('F-score: ', str((2*precision*recall)/(precision+recall)))




while txt != 'N':
    print('\nEnter a sentence:')
    txt = input()
    print('Sentence S: ', txt)
    txt = sprzataczka(txt)
    rick, morty, label = naive_bayes_classifier(txt, rickVocab, mortyVocab, RickProb, MortyProb)
    print('was classified as ' + label)
    print('P(Rick|S) = ' + str(rick))
    print('P(Morty|S) = ' + str(morty))
    print('Do you want to enter another sentence [Y/N]?')
    txt=input().upper()
