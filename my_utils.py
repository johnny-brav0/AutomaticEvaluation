import re
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential


def essay_to_wordlist(essay_v, remove_stopwords):
    #Remove the tagged labels and word tokenize the sentence.
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    """Make ar from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model.wv[word]) # might causing the warning
    featureVec = np.divide(featureVec,num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs


def get_model():
    """Define the model."""
    model = Sequential()
    model.add(Bidirectional(GRU(300, dropout=0.4, recurrent_dropout=0, return_sequences=True), input_shape=(1, 300)))
    model.add(Bidirectional(GRU(64, recurrent_dropout=0)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model
