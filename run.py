import json
import numpy as np
from keras.callbacks import TensorBoard
from data_prep import gen_cosine_amp_for_supervised as gen_testdata
from data_prep import print_data, series_to_supervised, win_classification_prep
from ufcnn import ufcnn_model_concat
from ufcnn_own import ufcnn_model_bn, ufcnn_model_fulldropout, ufcnn_model_mix

name = 'class/ufcnn/fulldropout/1'
sequence_length = 128 #672
output_sequence_length = 32#192
features = 5
nb_filter = 50
filter_length = 5
dropout = 0.0
window_length = 3
nb_buy_values = 3
batch_size = 64

#cos, train_y = gen_testdata(sequence_length*100)

with open('./ecto.json') as data_file:
    data = json.load(data_file)

tb_callback = TensorBoard(log_dir='./logs/' + name, histogram_freq=10,
                          batch_size=batch_size, write_graph=True, write_grads=True,
                          write_images=True, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None)

sell = np.array([float(d['sell']) for d in data])
sell = sell[::-1]
#sell = np.insert(np.diff(sell), 0, 0)
sell = sell.reshape(-1, 1)
sell = series_to_supervised(sell, n_in=sequence_length, n_out=output_sequence_length)

buy = np.array([float(d['buy']) for d in data])
buy = buy[::-1]
#buy = np.insert(np.diff(buy), 0, 0)
buy = buy.reshape(-1, 1)
buy = series_to_supervised(buy, n_in=sequence_length, n_out=output_sequence_length)

timestamp = np.array([float(d['timestamp']) for d in data])
timestamp = timestamp[::-1]
#timestamp = np.insert(np.diff(timestamp), 0, 0)
timestamp = timestamp.reshape(-1, 1)
timestamp = series_to_supervised(timestamp, n_in=sequence_length, n_out=output_sequence_length)

supply = np.array([float(d['supply']) for d in data])
supply = supply[::-1]
#supply = np.insert(np.diff(supply), 0, 0)
supply = supply.reshape(-1, 1)
supply = series_to_supervised(supply, n_in=sequence_length, n_out=output_sequence_length)

demand = np.array([float(d['demand']) for d in data])
demand = demand[::-1]
#demand = np.insert(np.diff(demand), 0, 0)
demand = demand.reshape(-1, 1)
demand = series_to_supervised(demand, n_in=sequence_length, n_out=output_sequence_length)

train_y = win_classification_prep(sell, buy, nb_buy_values, sequence_length, window_length)

train_x = sell.values[:, :sequence_length]
train_x = np.stack((train_x,
                    buy.values[:, :sequence_length],
                    timestamp.values[:, :sequence_length],
                    supply.values[:, :sequence_length],
                    demand.values[:, :sequence_length]), -1)
#train_x = train_x.reshape(-1, sequence_length, features)

#train_y = sell.values[:, sequence_length:]
#train_y = train_y.reshape(-1, output_sequence_length)


model = ufcnn_model_fulldropout(sequence_length=sequence_length, features=features, filter_length=filter_length,
                        nb_filter=nb_filter, activation='relu', loss="binary_crossentropy", dropout=dropout,
                        output_sequence_length=output_sequence_length, resolution_levels=3)

#model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=10, validation_split=0.1,
#           callbacks=[tb_callback]
#)

#model_new.save('./models/' + name + '.h5')

#predicted = model.predict(x=cos, batch_size=batch_size)

#print_data(train_y, predicted)
