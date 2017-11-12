import json
import numpy as np
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

series = np.array(float(d['sell']) for d in data)

model = ufcnn_model_concat(sequence_length=sequence_length, activation='relu')

#model.fit(x=cos, y=expected, batch_size=batch_size, epochs=5, validation_split=0.1)

#predicted = model.predict(x=cos, batch_size=batch_size)

#print_data(expected, predicted)
