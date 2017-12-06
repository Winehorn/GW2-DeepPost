import json
import numpy as np
from keras.callbacks import TensorBoard
from data_prep import gen_cosine_amp_for_supervised as gen_testdata
from data_prep import print_data, series_to_supervised
from ufcnn import ufcnn_model_concat

sequence_length = 64        # same as in Roni Mittelman's paper - this is 2 times 32 - a line in Ronis input contains 33 numbers, but 1 is time and is omitted
features = 1                # guess changed Ernst 20160301
nb_filter = 150             # same as in Roni Mittelman's paper
filter_length = 5           # same as in Roni Mittelman's paper
output_dim = 1
batch_size = 128

#cos, expected = gen_testdata(sequence_length*100)

with open('./ecto.json') as data_file:
    data = json.load(data_file)

sell = np.array([float(d['sell']) for d in data])
sell = np.insert(np.diff(sell), 0, 0)
print(sell[0:5])
sell = sell.reshape(-1, 1)
sell = series_to_supervised(sell)

train = sell.values[:, 0]
print(train[0:5])
train = train.reshape(-1, 1, 1)
train = train[::-1]


expected = sell.values[:, 1]
print(expected[0:5])
expected = expected.reshape(-1, 1, 1)
expected = expected[::-1]


