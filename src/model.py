import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import kagglehub

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

def download_nltk():
  # nltk.download('wordnet')
  # nltk.download('punkt_tab')
  # nltk.download('averaged_perceptron_tagger')
  # nltk.download('stopwords')
  pass


class GetDataset:
  def __init__(self):
    self.train_filename = "twitter_training.csv"
    self.test_filename = "twitter_validation.csv"
    self.base_path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")
    self.df_train = pd.read_csv(f"{self.base_path}/{self.train_filename}", header=None)
    self.df_test = pd.read_csv(f"{self.base_path}/{self.test_filename}", header=None)
    self.df_train = self.df_train.dropna()[[2, 3]]
    self.df_test = self.df_test.dropna()[[2, 3]]
    self.df_train = self.df_train[self.df_train[2] != 'Irrelevant']
    self.df_test = self.df_test[self.df_test[2] != 'Irrelevant']

    # self.get_df()
  
  def get_df(self):
    return [self.df_train, self.df_test]
  
class LemmaTokenizer:
  def __init__(self):
    download_nltk()
    # self.stop_words = set(stopwords.words('english'))
    self.wnl = WordNetLemmatizer()
  def __call__(self, doc):
    tokens = word_tokenize(doc)
    word_and_tag = nltk.pos_tag(tokens)
    return [self.wnl.lemmatize(word, pos=self.get_pos_tag(tag)) for word, tag in word_and_tag]
  
  def get_pos_tag(self, tag):
    if tag.startswith('J'):
      return wordnet.ADJ
    elif tag.startswith('V'):
      return wordnet.VERB
    elif tag.startswith('N'):
      return wordnet.NOUN
    elif tag.startswith('R'):
      return wordnet.ADV
    else:
      return wordnet.NOUN



class SentimentAnalysisModel:
  def __init__(self):
    self.model = LogisticRegression(max_iter=1000)
    print("Lemmatizing and vectorizing the dataset...")
    self.vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
    self.dataset = GetDataset()
    # print(len(self.dataset))
    self.dataset = self.dataset.get_df()
    # print(self.dataset[0][3])
    self.X_train = self.vectorizer.fit_transform(self.dataset[0][3])
    self.X_test = self.vectorizer.transform(self.dataset[1][3])
    print("Dataset lemmatized and vectorized.")
    # print(len(self.X_train), len(self.X_test))
    self.y_train = self.dataset[0][2]
    self.y_test = self.dataset[1][2]
    
    # Train the model
    self.model.fit(self.X_train, self.y_train)
  
  def calculate_accuracy(self):
    y_pred = self.model.predict(self.X_test)
    return accuracy_score(self.y_test, y_pred)

  def calculate_auc(self):
    y_pred = self.model.predict(self.X_test)
    return f1_score(self.y_test, y_pred, average='weighted')
  
  def predict(self, text):
    return self.model.predict(self.vectorizer.transform([text]))[0]