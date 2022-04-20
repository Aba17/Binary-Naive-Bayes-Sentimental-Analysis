from collections import Counter

from pyparsing import WordStart
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import requests
from io import StringIO
import math


class PreprocessData:
    def __init__(self,path,lang='english'): 
        self.dataset=pd.read_csv(path, sep=",")
        self.stopwords=stopwords.words(lang)
        self.preprocess()
    
    @staticmethod
    def remove_punctuation(text):
        '''a function for removing punctuation'''
        # replacing the punctuations with no space, 
        # which in effect deletes the punctuation marks 
        translator = str.maketrans('', '', string.punctuation)
        # return the text stripped of punctuation marks
        return text.translate(translator)

    def replace_sentiment(self):
    #Convert the sentiment to numerical value......1 for positive and 0 for negative
        self.dataset = self.dataset.replace({'sentiment': {'positive': 1, 'negative': 0}})

    #A function to remove the stopwords
    def remove_stopwords(self,text):
        text = [word.lower() for word in text.split() if word.lower() not in self.stopwords]
        # joining the list of words with space separator
        return " ".join(text)

    def preprocess(self):
        self.dataset['review'] = self.dataset['review'].apply(self.remove_punctuation)
        self.dataset['review'] = self.dataset['review'].apply(self.remove_stopwords)
        self.replace_sentiment()





class TF_IDF:
    def __init__(self,dataset):
        self.dataset=dataset
        self.ponderations={}
        self.tf_idf()
    
    def term_frequency(self,word,document):
        #Frequence of the word over number of word of document
        word_list=document.split(" ")
        dict=Counter(word_list) #Give a dictionnary which the keys are the words and the values are the number of occurences in word_list
        return dict[word]/len(word_list)
    
    def inverse_document_frequency(self,word,document):
        N=len(self.dataset)
        #Contains return a list of True or False
        doc_t=len(self.dataset[self.dataset['review'].str.contains(word)])
        return math.log(N/doc_t)

    def tf_idf(self):
        for i in range(len(self.dataset)):
            document=self.dataset.iloc[i,0]
            word_list=document.split(" ")
            for word in word_list:
                if word not in self.ponderations:
                    self.ponderations[word]=self.term_frequency(word,document)*self.inverse_document_frequency(word,document)






path="/Users/aba/Desktop/BNB/IMDB_Dataset.csv"
documents=PreprocessData(path)
words=TF_IDF(documents.dataset)
i=0
for val in words.ponderations:
    print(words.ponderations[val])
    i+=1
    if i==10:
        break