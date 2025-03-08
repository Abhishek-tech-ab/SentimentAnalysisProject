import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import string
import unicodedata
import nltk
import pickle
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('IMDB Dataset.csv')

dataset.head()

print(f'Rows:{dataset.shape[1]}\nColumns:{dataset.shape[0]}')
print(f'Columns Names: {list(dataset.columns)}')

nlp = English()
stopwords = list(STOP_WORDS)
punctuations = string.punctuation

def tokenizer(sentence):
        mytokens = nlp(sentence)
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
        mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
        return mytokens

class predictors(TransformerMixin):
        def transform(self, X, **transform_params):
            return [clean_text(text) for text in X]
        def fit(self, X, y, **fit_params):
            return self
        def get_params(self, deep=True):
            return {}

def clean_text(text):
    return text.strip().lower()

vectorizer = CountVectorizer(tokenizer = tokenizer, ngram_range=(1,1))
tfvectorizer = TfidfVectorizer(tokenizer = tokenizer)

X = dataset['review']
y = dataset['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

classifier = LogisticRegression()
LRmodel = Pipeline([("cleaner", predictors()),
                ('vectorizer', vectorizer),
                ('classifier', classifier)])

# Train the Model\n",
LRmodel.fit(X_train, y_train)
LRpred = LRmodel.predict(X_test)
print(f'Confusion Matrix:\\n{confusion_matrix(y_test, LRpred)}')
print(f'\\nClassification Report:\\n{classification_report(y_test, LRpred)}')
print(f'Accuracy: {accuracy_score(y_test, LRpred) * 100}%')
pickle.dump(LRmodel, open('/saved_model/LinearRegression_model.sav', 'wb'))
print('Logistic Regression trained Model Saved')

# Another random review
pre = LRmodel.predict(["Production has an incredibly important place to shoot a series or film. Sometimes even a very minimalist story can reach an incredibly successful point after the right production stages. The Witcher series is far from minimalist. The Witcher is one of the best Middle-earth works in the world. Production quality is essential if you want to handle such a topic successfully."])
print(f'Prediction: {pre[0]}')

RFclassifier = RandomForestClassifier(n_estimators=200)
RFmodel = Pipeline([("cleaner", predictors()),
                ('vectorizer', vectorizer),
                ('classifier', RFclassifier)])

# Train the Model
RFmodel.fit(X_train, y_train)
RFpred = RFmodel.predict(X_test)
print(f'Confusion Matrix:\\n{confusion_matrix(y_test, RFpred)}')
print(f'\\nClassification Report:\\n{classification_report(y_test, RFpred)}')
print(f'Accuracy: {accuracy_score(y_test, RFpred) * 100}%')
pickle.dump(RFmodel, open('/saved_model/RandomForest_model.sav', 'wb'))
print('RandomForest trained Model Saved')

# Another random review
pre = RFmodel.predict(["I think this is my first review. This series is so bad I had to write one. I don't understand the good score. I have tried on 2 separate occasions to watch this show. Haven't even gotten past the 2nd episode because it is SO BORING."])
print(f'Prediction: {pre[0]}')

SVCclassifier = LinearSVC()
SVCmodel = Pipeline([("cleaner", predictors()),
                ('vectorizer', vectorizer),
                ('classifier', SVCclassifier)])

# Train the Model
SVCmodel.fit(X_train, y_train)
SVCpred = SVCmodel.predict(X_test)
print(f'Confusion Matrix:\\n{confusion_matrix(y_test, SVCpred)}')
print(f'\\nClassification Report:\\n{classification_report(y_test, SVCpred)}')
print(f'Accuracy: {accuracy_score(y_test, SVCpred) * 100}%')
pickle.dump(SVCmodel, open('/saved_model/LinearSVC_model.sav', 'wb'))
print('LinearSVC trained Model Saved')

pre = SVCmodel.predict(["Henry cavill nailed the role perfectly. The fight scenes, the music, the cinematography, the whole atmosphere is beyond amazing. Netflix did it again"])
print(f'Prediction: {pre[0]}')
