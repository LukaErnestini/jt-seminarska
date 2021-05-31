import numpy as np
from numpy.lib.npyio import load
import tensorflow as tf
import nltk
import pandas as pd
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf

from text_utils import tokenize, load_data
from gru import GRU, OneStep

from syllabify_.src.syllabifier.syllabifyARPA import syllabifyARPA
arpabet = nltk.corpus.cmudict.dict()

data = load_data()
data_haikus_ar = ['\n'.join([row[0], row[1], row[2]]) for row in data]
data_haikus_ar_s = '\n\n'.join(data_haikus_ar)

vocab = sorted(set(' '.join(data_haikus_ar_s)))
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

haiku_ids = ids_from_chars(tf.strings.unicode_split(data_haikus_ar_s, 'UTF-8'))
ids_haikus_dataset = tf.data.Dataset.from_tensor_slices(haiku_ids)

sequence_length = 100
sequences = ids_haikus_dataset.batch(sequence_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

haikus_dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 1000

haikus_dataset = (
    haikus_dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

model = GRU(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=256,
    rnn_units=1024)

for input_example_batch, target_example_batch in haikus_dataset.take(1):
    model(input_example_batch)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)
#history = model.fit(haikus_dataset, epochs=50)
#model.save_weights('simple_GRU/modelGRU.h5')
model.load_weights('simple_GRU/modelGRU.h5')

one_step_model = OneStep(model, chars_from_ids, ids_from_chars, temperature=1)

haiku_starts = [row[0].split(' ')[0] for row in data]
with open('simple_GRU/results.txt', 'wb') as fp:
    for start in haiku_starts:
        states = None
        next_char = tf.constant([start])
        result = [next_char]

        while next_char != '\n':
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)
        while next_char == '\n':
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)
        while next_char != '\n':
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)
        while next_char == '\n':
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)
        while next_char != '\n':
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        fp.write((result[0].numpy().decode('utf-8') + '\n' + '_'*80 + '\n\n').encode('utf-8'))

