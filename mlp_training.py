import numpy as np
from numpy.lib.npyio import load
import tensorflow as tf
import nltk
import pandas as pd
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf
import pickle

from text_utils import tokenize, load_data
from gru import GRU, OneStep
from mlp import MLP

from syllabify_.src.syllabifier.syllabifyARPA import syllabifyARPA
arpabet = nltk.corpus.cmudict.dict()

data = load_data()
data_haikus_ar = ['\n'.join([row[0], row[1], row[2]]) for row in data if row[3] == 'gutenberg' or row[3] == 'tempslibres']
data_haikus_ar_s = '\n\n'.join(data_haikus_ar)
vocab = sorted(set(' '.join(data_haikus_ar_s)))
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

example_batch = pickle.load(open('example_batch', 'rb'))
verse1_model = GRU(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=256,
    rnn_units=1024)
verse1_model(example_batch)
verse1_model.load_weights('verse0.h5')
verse1_model_oneStep = OneStep(verse1_model, chars_from_ids, ids_from_chars, temperature=1)
verse2_model = GRU(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=256,
    rnn_units=1024)
verse2_model(example_batch)
verse2_model.load_weights('verse1.h5')
verse2_model_oneStep = OneStep(verse2_model, chars_from_ids, ids_from_chars, temperature=1)
verse3_model = GRU(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=256,
    rnn_units=1024)
verse3_model(example_batch)
verse3_model.load_weights('verse2.h5')
verse3_model_oneStep = OneStep(verse3_model, chars_from_ids, ids_from_chars, temperature=1)

#verse 1 -> 2
x_sequences = [row[0] for row in data]
x = []
y = []
for seq in x_sequences:
    states_1 = None
    states_2 = None
    for char in seq:
        char = tf.constant([char])
        _, states_1 = verse1_model_oneStep.generate_one_step(char, states_1)
        _, states_2 = verse2_model_oneStep.generate_one_step(char, states_2)
    x.append(states_1)
    y.append(states_2)
pickle.dump((x, y), open('states1_2', 'wb'))
x, y = pickle.load(open('states1_2', 'rb'))

train_dataset = tf.data.Dataset.from_tensor_slices((x[:len(x)//6 * 5], y[:len(x)//6 * 5])).shuffle(2000).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((x[len(x)//6 * 5:], y[len(x)//6 * 5:])).batch(64)

mlp_model1 = MLP(rnn_units=1024)
mlp_model1.compile(optimizer='adam', loss='mae')
mlp_model1(x[0])
history = mlp_model1.fit(train_dataset, validation_data=val_dataset, epochs=100)
mlp_model1.save_weights('mlp1.h5')
mlp_model1.load_weights('mlp1.h5')

# verse 2 -> 3
x_sequences = [row[1] for row in data]
x = []
y = []
for seq in x_sequences:
    states_2 = None
    states_3 = None
    for char in seq:
        char = tf.constant([char])
        _, states_2 = verse2_model_oneStep.generate_one_step(char, states_2)
        _, states_3 = verse3_model_oneStep.generate_one_step(char, states_3)
    x.append(states_2)
    y.append(states_3)
pickle.dump((x, y), open('states2_3', 'wb'))
x, y = pickle.load(open('states2_3', 'rb'))

mlp_model2 = MLP(rnn_units=1024)
mlp_model2.compile(optimizer='adam', loss='mae')
mlp_model2(x[0])
history = mlp_model2.fit(train_dataset, validation_data=val_dataset, epochs=100)
mlp_model2.save_weights('mlp2.h5')
mlp_model2.load_weights('mlp2.h5')

haiku_starts = [row[0].split(' ')[0] for row in data]

with open('results.txt', 'w+') as fp:
    for start in haiku_starts:
        states = None
        next_char = tf.constant([start])
        result = [next_char]

        while next_char != '\n':
            next_char, states = verse1_model_oneStep.generate_one_step(next_char, states=states)
            result.append(next_char)

        states = mlp_model1(states)

        while next_char == '\n':
            next_char, states = verse2_model_oneStep.generate_one_step(next_char, states=states)
        result.append(next_char)
        while True:
            next_char, states = verse2_model_oneStep.generate_one_step(next_char, states=states)
            result.append(next_char)
            if next_char == '\n':
                break
        
        states = mlp_model2(states)

        while next_char == '\n':
            next_char, states = verse3_model_oneStep.generate_one_step(next_char, states=states)
        result.append(next_char)
        while True:
            next_char, states = verse3_model_oneStep.generate_one_step(next_char, states=states)
            result.append(next_char)
            if next_char == '\n':
                break

        result = tf.strings.join(result)
        fp.write(result[0].numpy().decode('utf-8') + '\n' + '_'*80 + '\n\n')
