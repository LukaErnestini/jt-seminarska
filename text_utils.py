import string
import re
import pandas as pd
import numpy as np 

def tokenize(text):
    # remove punctuation
    myStrPunct = string.punctuation.replace('\'', '')
    text = text.translate(str.maketrans('', '', myStrPunct))
    # split to tokens
    tokens = text.split()
    # remove invalid
    valid = re.compile('^[a-zA-ZÀ-ž\']*$')
    tokens = list(filter(lambda token: valid.search(token), tokens))

    return tokens

def load_data():
    df = pd.read_csv('data/all_haiku.csv')
    data = df[['0', '1', '2', 'source']].to_numpy()
    dlt = []
    for i in range(len(data)):
        if type(data[i][0]) != type('string') or type(data[i][1]) != type('string') or type(data[i][2]) != type('string'):
            dlt.append(i)
            continue
        if data[i][3] != 'gutenberg' and data[i][3] != 'tempslibres':
            dlt.append(i)
            continue
    data = np.delete(data, dlt, axis=0)
    return data