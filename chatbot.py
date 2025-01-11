import random
import json
import pickle
from typing import List
import keras
import tensorflow as tf
import nltk
from keras.src.utils.module_utils import tensorflow
from tensorflow.python.keras.models import load_model
from termcolor import RESET

nltk.download('punkt_tab')
nltk.download('wordnet')
import numpy as np
from nltk.corpus.reader import documents

from nltk.stem import WordNetLemmatizer

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Activation, Dropout, TFSMLayer
from keras._tf_keras.keras.optimizers import SGD



lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

layer = keras.layers.TFSMLayer("saved_model", call_endpoint="serving_default")
model = Sequential([layer])
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))['output_0'][0]
    ERROR_THREASHOLD = 0.25 # Allows certain incertainty
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THREASHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            return result
    return "I'm sorry, I couldn't find an appropriate response."


print('**********************Chat bot start*********************8')

"""Only for console testing of the chatbot"""
# while True:
#     message = input("")
#     ints = predict_class(message)
#     print("ints", ints)
#     res = get_response(ints, intents)
#     print(res)