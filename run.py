import json
import numpy as np
from keras.callbacks import TensorBoard
from data_prep import gen_cosine_amp_for_supervised as gen_testdata
from data_prep import print_data, series_to_supervised
from ufcnn import ufcnn_model_concat
from ufcnn_own import ufcnn_model

sequence_length = 128        # same as in Roni Mittelman's paper - this is 2 times 32 - a line in Ronis input contains 33 numbers, but 1 is time and is omitted
output_sequence_length = 2
features = 1                # guess changed Ernst 20160301
nb_filter = 50            # same as in Roni Mittelman's paper
filter_length = 5           # same as in Roni Mittelman's paper
output_dim = 1
batch_size = 64

#cos, expected = gen_testdata(sequence_length*100)

with open('./ecto.json') as data_file:
    data = json.load(data_file)

tb_callback = TensorBoard(log_dir='./logs/ufcnn/testrunBatchSize256', histogram_freq=10,
                          batch_size=batch_size, write_graph=True, write_grads=True,
                          write_images=True, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None)

sell = np.array([float(d['sell']) for d in data])
sell = sell[::-1]
sell = np.insert(np.diff(sell), 0, 0)
sell = sell.reshape(-1, 1)
sell = series_to_supervised(sell, n_in=sequence_length, n_out=output_sequence_length)

train = sell.values[:, :sequence_length]
train = train.reshape(-1, sequence_length, 1)

expected = sell.values[:, sequence_length:]
expected = expected.reshape(-1, 1, output_sequence_length)


model_new = ufcnn_model(sequence_length=sequence_length, output_dim=2, activation='relu', resolution_levels=2)

model_new.fit(x=train, y=expected, batch_size=batch_size, epochs=2, validation_split=0.1,
#           callbacks=[tb_callback]
           )

#predicted = model.predict(x=cos, batch_size=batch_size)

#print_data(expected, predicted)
