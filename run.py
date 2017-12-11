import json
import numpy as np
from keras.callbacks import TensorBoard
from data_prep import gen_cosine_amp_for_supervised as gen_testdata
from data_prep import print_data, series_to_supervised
from ufcnn import ufcnn_model_concat
from ufcnn_own import ufcnn_model_bn, ufcnn_model_fulldropout

name = 'ufcnn/bn'
sequence_length = 672        # same as in Roni Mittelman's paper - this is 2 times 32 - a line in Ronis input contains 33 numbers, but 1 is time and is omitted
output_sequence_length = 192
features = 1                # guess changed Ernst 20160301
nb_filter = 30            # same as in Roni Mittelman's paper
filter_length = 10           # same as in Roni Mittelman's paper
dropout = 0.4
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
sell = np.insert(np.diff(sell), 0, 0)
sell = sell.reshape(-1, 1)
sell = series_to_supervised(sell, n_in=sequence_length, n_out=output_sequence_length)

train_x = sell.values[:, :sequence_length]
train_x = train_x.reshape(-1, sequence_length, 1)

train_y = sell.values[:, sequence_length:]
train_y = train_y.reshape(-1, output_sequence_length)


model = ufcnn_model_bn(sequence_length=sequence_length, filter_length=filter_length,
                        nb_filter=nb_filter, activation='relu',
                        output_sequence_length=output_sequence_length, resolution_levels=3)

model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=10, validation_split=0.1,
           #callbacks=[tb_callback]
)

#model_new.save('./models/' + name + '.h5')

#predicted = model.predict(x=cos, batch_size=batch_size)

#print_data(train_y, predicted)
