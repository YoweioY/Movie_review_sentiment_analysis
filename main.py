# -*- coding: utf-8 -*-
import os
import re
import random
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Activation, GRU, Bidirectional
from gensim.models import Word2Vec
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datamining.ultis as ultis

tf.random.set_seed(1)
random.seed(2)
np.random.seed(3)

df = pd.read_csv('labeledTrainData_1w.tsv') 

max_length = 150
embedding_dim = 250
X_train_word, y_train, X_test_word, y_test = ultis.get_train_test(df, 'sentence2')
w2v_model = Word2Vec(X_train_word, min_count=1, size=embedding_dim, iter=10, sg=1)
X_train, X_test, tokenizer = ultis.text_tokenizer(X_train_word, X_test_word, max_length)
embedding_matrix, vocab_size = ultis.get_embedding_matrix(tokenizer, w2v_model, embedding_dim)

# model_type = ('LSTM', 'GRU', 'Bidirectional LSTM', 'Bidirectional GRU')
model_type = 'GRU'

if model_type == 'LSTM':
    #LSTM

    lstm_model = Sequential()
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)
    lstm_model.add(embedding_layer)
    lstm_model.add(Dropout(0.5))
    lstm_model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.summary()

    lstm_model.compile(
        loss='binary_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    lstm_record = lstm_model.fit(X_train, y_train, batch_size=1024, epochs=30, validation_split=0.2, verbose=1, shuffle=True)
    scores = lstm_model.predict(X_test, verbose=2)
    y_pred=np.where(scores>0.5,1,0)
    print(classification_report(y_test, y_pred,digits=4))

    ultis.plot_acc(lstm_record.history['accuracy'], lstm_record.history['val_accuracy'], 'LSTM Accuracy')
    ultis.plot_loss(lstm_record.history['loss'], lstm_record.history['val_loss'], 'LSTM Loss')

    pred = lstm_model.predict(X_test)
    fpr, tpr, best_thresholds = ultis.get_best_thresholds(lstm_model, pred, y_test)
    origin_acc = ultis.compute_acc(pred, y_test, 0.5)
    improve_acc = ultis.compute_acc(pred, y_test, best_thresholds)
    ultis.plot_roc_curve(fpr, tpr)
    print(origin_acc,improve_acc,best_thresholds)

elif model_type == 'GRU':
    #GRU

    GRU_model = Sequential()
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)
    GRU_model.add(embedding_layer)
    GRU_model.add(Dropout(0.5))
    GRU_model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
    GRU_model.add(Dense(1, activation='sigmoid'))
    GRU_model.summary()

    GRU_model.compile(
        loss='binary_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    GRU_record = GRU_model.fit(X_train, y_train, batch_size=1024, epochs=30, validation_split=0.2, verbose=1)
    scores = GRU_model.predict(X_test, verbose=2)
    y_pred=np.where(scores>0.5,1,0)
    cm=confusion_matrix(y_pred,y_test)
    print(classification_report(y_test, y_pred,digits=4))

    ultis.plot_acc(GRU_record.history['accuracy'], GRU_record.history['val_accuracy'], 'GRU Accuracy')
    ultis.plot_loss(GRU_record.history['loss'], GRU_record.history['val_loss'], 'GRU Loss')

elif model_type == 'Bidirectional LSTM':
    #Bidirectional LSTM

    Blstm_model = Sequential()
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)
    Blstm_model.add(embedding_layer)
    Blstm_model.add(Dropout(0.5))
    Blstm_model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    Blstm_model.add(Dense(1, activation='sigmoid'))
    Blstm_model.summary()

    Blstm_model.compile(
        loss='binary_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    Blstm_record = Blstm_model.fit(X_train, y_train, batch_size=1024, epochs=30, validation_split=0.2, verbose=1)

    scores = Blstm_model.predict(X_test, verbose=2)
    y_pred=np.where(scores>0.5,1,0)
    cm=confusion_matrix(y_pred,y_test)
    print(classification_report(y_test, y_pred,digits=4))

    ultis.plot_acc(Blstm_record.history['accuracy'], Blstm_record.history['val_accuracy'], 'Bidirectional LSTM Accuracy')
    ultis.plot_loss(Blstm_record.history['loss'], Blstm_record.history['val_loss'], 'Bidirectional LSTM Loss')


elif model_type == 'Bidirectional GRU':
    #Bidirectional GRU

    BGRU_model = Sequential()
    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)
    BGRU_model.add(embedding_layer)
    BGRU_model.add(Dropout(0.5))
    BGRU_model.add(Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.2)))
    BGRU_model.add(Dense(1, activation='sigmoid'))
    BGRU_model.summary()

    BGRU_model.compile(
        loss='binary_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )

    BGRU_record = BGRU_model.fit(X_train, y_train, batch_size=1024, epochs=30, validation_split=0.2, verbose=1)
    scores = BGRU_model.predict(X_test, verbose=2)
    y_pred=np.where(scores>0.5,1,0)
    cm=confusion_matrix(y_pred,y_test)
    print(classification_report(y_test, y_pred,digits=4))

    ultis.plot_acc(BGRU_record.history['accuracy'], BGRU_record.history['val_accuracy'], 'Bidirectional GRU Accuracy')
    ultis.plot_loss(BGRU_record.history['loss'], BGRU_record.history['val_loss'], 'Bidirectional GRU Loss')

