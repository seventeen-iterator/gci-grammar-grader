import numpy as np
import pandas as pd
import gensim
import pickle

from keras.preprocessing.sequence import pad_sequences
from util import MAX_SEQUENCE_LENGTH, EssayIter, SequenceIter

ds = pd.read_csv('data/preprocessed_train.csv')
essay_iter = EssayIter(ds.essay)
model = gensim.models.Word2Vec(
    essay_iter, window=9, min_count=3, alpha=0.1, seed=666)

sequence_iter = SequenceIter(ds, MAX_SEQUENCE_LENGTH, model)

set_x = []
set_y = []
for e, m in sequence_iter:
    set_x.append(e)
    set_y.append(m)

set_x = pad_sequences(set_x, maxlen=MAX_SEQUENCE_LENGTH,
                      padding='pre', value=0)
set_y = np.array(set_y)

with open('processed_data.pickle', 'wb') as f:
    pickle.dump((set_x, set_y, model), f)
