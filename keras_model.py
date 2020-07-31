# !pip install vaderSentiment

from google.colab import files
import tensorflow as tf
import pandas as pd
import io
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keras.models import Sequential, Model
from keras.layers import Embedding, SimpleRNN, Dense, Dropout, Activation, Input, LSTM, GRU

def convert_polarity(polarity):
  if polarity >= 0 : 
    return 1
  else :
    return 0

analyzer = open('tweets_en.txt', 'r')

sid_obj = SentimentIntensityAnalyzer()

sentences = []
labels = []

for tweet in analyzer:
  polarity = convert_polarity(sid_obj.polarity_scores(tweet)['compound'])
  sentences.append(tweet)
  labels.append(polarity)


# 0: Negative
# 2: Neutral
# 4: Positive

training_size = int(len(sentences) * 0.8)

training_sentences = sentences[0: training_size]
testing_senteces = sentences[: training_size]
training_labels = labels[0: training_size]
testing_labels = labels[: training_size]

# Put labels into list to use later:

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 1000
embedding_dim = 16
max_length = 280
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index


training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_senteces)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = tf.keras.models.Sequential()

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

# gru_input = Input(shape=(max_length,))
# embedding = Embedding(vocab_size, 128, input_length=max_length)(gru_input)
# gru = GRU(128)(embedding)
# dropout = Dropout(0.4)(gru)
# #dense_middle = Dense(128)(dropout)
# #dropout2 = Dropout(0.4)(dense_middle)
# dense = Dense(1)(dropout)
# activation = Activation('sigmoid')(dense)
# model = Model(gru_input, activation)
model.summary()

# model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,return_sequences=True)))
# model.add(tf.keras.layers.Dense(6, activation='relu'))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(0.01),
#               metrics=['accuracy'])

# callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',baseline=None, restore_best_weights=False)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
model.save('Trained_model')
num_epochs=25

# print(model.summary())

modelo = model.fit(training_padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final), validation_steps=30)

plt.plot(modelo.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Increase because the early stopping

plt.plot(modelo.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Use the model to predict a review
# fake_reviews = ['I love this phone', 'I hate spaghetti', 'Everything was cold', 'Everything was hot exactly as I wanted', 'Everything was green', 'the host seated us immediately', 'they gave us free chocolate cake', 'not sure about the wilted flowers on the table','only works when I stand on tippy toes', 'does not work when I stand on my head']

fake_reviews = ['you\'re class', 'you are really happy', 'i am happy that trump is not winning', 'we are not sad when he moved away', 'they weren\'t playing', 'i am sailing to england', 'they should never join us for dinner']

print(fake_reviews)

# Create the sequences
padding_type='post'
sample_sequences = tokenizer.texts_to_sequences(fake_reviews)
fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)

classes = model.predict(fakes_padded)
k = 0;
for i in classes:
  for a in i:
    print(fake_reviews[k], a)
    if a > 0.5 :
        print("Positive")
    elif a < 0.5:
        print("Negative")
    else:
        print("Neutral")
    k+=1
