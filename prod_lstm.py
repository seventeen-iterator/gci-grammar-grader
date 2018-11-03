import pickle
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.losses import mae
from keras import regularizers

from util import preprocess_texts, Predictor, MAX_SEQUENCE_LENGTH

with open('processed_data.pickle', 'rb') as f:
    set_x, set_y, model = pickle.load(f)

weights = model.wv.vectors
vocab_size, embedding_size = weights.shape

nets = Sequential([
    Embedding(input_dim=vocab_size,
              output_dim=embedding_size,
              weights=[weights],
              input_length=MAX_SEQUENCE_LENGTH,
              mask_zero=True,
              trainable=False),
    Bidirectional(LSTM(20, activation='relu', dropout=0.5)),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))
])
nets.compile(optimizer='adam', loss=mae, metrics=['mse', 'mae'])
nets.fit(set_x, set_y, epochs=10, batch_size=32, verbose=0)

print("Prediction on correct sentence: ", nets.predict(
    preprocess_texts(["Computers are the best."], model)) * 5)
print("Prediction on incorrect sentence: ", nets.predict(
    preprocess_texts(["sdfsdfsd sdfsdfg gfdfghghfgh"], model)) * 5)
print("Prediction on somewhat correct sentence: ", nets.predict(
    preprocess_texts(["Computars ain't good."], model)) * 5)

with open("lstm_prod.pickle", "wb") as output_file:
    pred_function = Predictor(nets, model, preprocess_texts).predict
    pickle.dump((pred_function, nets), output_file)
