import string 
from nltk.stem import SnowballStemmer
import nltk
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

stemmer = SnowballStemmer("french")

class Chatbot:
    def __init__(self, path_to_json_file ) -> None:
        intents_file = open(path_to_json_file, encoding='utf-8').read()
        self.intents = json.loads(intents_file)
        self.intentions = self.intents['intentions']

        self.tags = []
        self.questions = []
        self.reponses = {}

        for intention in self.intentions:
            tag = intention.get("tag")
            for question in intention.get("questions"):
                self.questions.append(question)
                self.tags.append(tag)
            self.reponses[tag] = intention.get("reponses")
        
        self.df = pd.DataFrame({
            "questions": self.questions,
                "label": self.tags
        })

        self.df["questions"] = self.df["questions"].apply(lambda x : self.text_preprocess(x))

    def replace_punct_with_space(self, text):
        punctuations = string.punctuation
        for punct in punctuations:
            text = text.replace(punct, ' ')
        return text

    def sans_accent(self, text):
        text = text.replace("é","e")
        text = text.replace("è","e")
        text = text.replace("ê","e")
        text = text.replace("à","a")
        text = text.replace("ç","c")
        return text

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text)

    def stemming(self, tokenized_text):
        text=[stemmer.stem(word.lower())  for word in tokenized_text]
        return text

    def text_preprocess(self, text):
        text = self.replace_punct_with_space(text)
        text = self.sans_accent(text)
        tokenized_text = self.tokenize(text)
        text_preprocessed = " ".join(self.stemming(tokenized_text))
        return text_preprocessed


    def vectorize(self, ):
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.df["questions"])
        return self.vectorizer
        


    def labelize(self):
        self.encoder = LabelEncoder()
        self.encoder.fit(self.df["label"])
        return self.encoder
    

    def input_preprocess(self, text):
        text = self.text_preprocess(text)
        text_vect = self.vectorize().transform([text])
        return text_vect.toarray()

    def model(self):
        self.encoded_labels = self.labelize().transform(self.df["label"])
        self.data_vect = self.vectorize().transform(self.df["questions"])
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(self.data_vect.toarray()[0]),), activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(set(self.encoded_labels)), activation='softmax'))
        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return self.model
    
    def train(self):
        model = self.model()
        model.fit(self.data_vect.toarray(), self.encoded_labels, epochs=1000, batch_size=80)
        model.save("ditbot.h5")
