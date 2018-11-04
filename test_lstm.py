import pickle
from math import sqrt

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as sk_mse
from util import preprocess_texts, Predictor, MAX_SEQUENCE_LENGTH

with open('processed_data.pickle', 'rb') as f:
    set_x, set_y, model = pickle.load(f)

weights = model.wv.vectors
vocab_size, embedding_size = weights.shape

train_valid_X, test_X, train_valid_y, test_y = train_test_split(
    set_x, set_y, test_size=0.1)
train_X, valid_X, train_y, valid_y = train_test_split(
    train_valid_X, train_valid_y, test_size=0.1)

nets = Sequential([
    Embedding(input_dim=vocab_size,
              output_dim=embedding_size,
              weights=[weights],
              input_length=MAX_SEQUENCE_LENGTH,
              mask_zero=True,
              trainable=False),
    Bidirectional(LSTM(20, activation='relu', dropout=0.5)),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001))
])
nets.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
nets.fit(train_X, train_y, epochs=2, batch_size=64,
         validation_data=(valid_X, valid_y), verbose=1)

predict_y = nets.predict(test_X, batch_size=64)
print("Test evaluation: ", sqrt(sk_mse(test_y, predict_y)))
print("Prediction on correct sentence: ", nets.predict(
    preprocess_texts(["Computers are the best."], model)) * 5)
print("Prediction on incorrect sentence: ", nets.predict(
    preprocess_texts(["sdfsdfsd sdfsdfg gfdfghghfgh"], model)) * 5)
print("Prediction on somewhat correct sentence: ", nets.predict(
    preprocess_texts(["Computars ain't good."], model)) * 5)


with open("lstm_test.pickle", "wb") as output_file:
    pred_function = Predictor(nets, model, preprocess_texts).predict
    pickle.dump((pred_function, nets), output_file)
