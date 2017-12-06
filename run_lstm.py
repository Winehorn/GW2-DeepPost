import json
import numpy as np
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, GRU

from data_prep import gen_cosine_amp_for_supervised as gen_testdata
from data_prep import print_data, series_to_supervised
from ufcnn import ufcnn_model_concat
from ufcnn_own import ufcnn_model

sequence_length = 672  # same as in Roni Mittelman's paper - this is 2 times 32 - a line in Ronis input contains 33 numbers, but 1 is time and is omitted
output_sequence_length = 192
features = 1  # guess changed Ernst 20160301
nb_filter = 50  # same as in Roni Mittelman's paper
filter_length = 5  # same as in Roni Mittelman's paper
output_dim = 1
batch_size = 64

# cos, train_y = gen_testdata(sequence_length*100)

with open('./ecto.json') as data_file:
    data = json.load(data_file)

tb_callback = TensorBoard(log_dir='./logs/lstm/672_192', histogram_freq=10,
                          batch_size=batch_size, write_graph=True, write_grads=True,
                          write_images=True, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None)

sell = np.array([float(d['sell']) for d in data])
sell = sell[::-1]
sell = np.insert(np.diff(sell), 0, 0)
sell = sell.reshape(-1, 1)
sell = series_to_supervised(sell, n_in=sequence_length, n_out=output_sequence_length)

train_x = sell.values[:, :sequence_length]
train_x = train_x.reshape(-1, sequence_length, 1)

train_y = sell.values[:, sequence_length:]
train_y = train_y.reshape(-1, output_sequence_length, 1)

# # train_y input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(GRU(150,
               return_sequences=False,
               input_shape=(sequence_length, features),
               dropout=0.4,
               #batch_input_shape=(batch_size, sequence_length, features),
               stateful=False))
model.add(RepeatVector(output_sequence_length))
model.add(GRU(150, return_sequences=True, dropout=0.4, stateful=False))
model.add(TimeDistributed(Dense(output_dim, activation='linear')))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y, validation_split=0.1,
          batch_size=batch_size,
          epochs=100,
          shuffle=False,
          callbacks=[tb_callback]
          )

# predicted = model.predict(x=cos, batch_size=batch_size)
