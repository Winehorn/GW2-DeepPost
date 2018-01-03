import json
import numpy as np
from data_prep import series_to_supervised, win_classification_prep
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe


name = 'class/ufcnn/fulldropout/1'





def data():
    return


def createModel():
    sequence_length = 128
    output_sequence_length = 32
    features = 5
    filter_length = 5
    window_length = 3
    nb_buy_values = 3
    batch_size = 64

    sequence_length_choice = {{choice([32, 64, 128, 256, 512])}}
    output_sequence_length_choice = {{choice([32, 64, 128, 256, 512])}}
    kernel_choice_1 = {{choice([3, 5, 10])}}
    kernel_choice_2 = {{choice([3, 5, 10])}}
    kernel_choice_3 = {{choice([3, 5, 10])}}
    dense_choice = {{choice([32, 64, 128, 512])}}
    activation_choice = {{choice(['sigmoid', 'softmax'])}}
    optimizer_choice = {{choice(['rmsprop', 'adam', 'sgd'])}}
    dropout_uniform = {{uniform(0, 1)}}

    with open('./ecto.json') as data_file:
        data_loaded = json.load(data_file)

    sell = np.array([float(d['sell']) for d in data_loaded])
    sell = sell[::-1]
    sell = sell.reshape(-1, 1)
    sell = series_to_supervised(sell, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    buy = np.array([float(d['buy']) for d in data_loaded])
    buy = buy[::-1]
    buy = buy.reshape(-1, 1)
    buy = series_to_supervised(buy, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    timestamp = np.array([float(d['timestamp']) for d in data_loaded])
    timestamp = timestamp[::-1]
    timestamp = timestamp.reshape(-1, 1)
    timestamp = series_to_supervised(timestamp, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    supply = np.array([float(d['supply']) for d in data_loaded])
    supply = supply[::-1]
    supply = supply.reshape(-1, 1)
    supply = series_to_supervised(supply, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    demand = np.array([float(d['demand']) for d in data_loaded])
    demand = demand[::-1]
    demand = demand.reshape(-1, 1)
    demand = series_to_supervised(demand, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    train_y = win_classification_prep(sell, buy, nb_buy_values, sequence_length_choice, window_length)

    train_x = sell.values[:, :sequence_length_choice]
    train_x = np.stack((train_x,
                        buy.values[:, :sequence_length_choice],
                        timestamp.values[:, :sequence_length_choice],
                        supply.values[:, :sequence_length_choice],
                        demand.values[:, :sequence_length_choice]), -1)

    with open('./ecto_test.json') as data_file:
        data_loaded = json.load(data_file)

    sell = np.array([float(d['sell']) for d in data_loaded])
    sell = sell[::-1]
    sell = sell.reshape(-1, 1)
    sell = series_to_supervised(sell, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    buy = np.array([float(d['buy']) for d in data_loaded])
    buy = buy[::-1]
    buy = buy.reshape(-1, 1)
    buy = series_to_supervised(buy, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    timestamp = np.array([float(d['timestamp']) for d in data_loaded])
    timestamp = timestamp[::-1]
    timestamp = timestamp.reshape(-1, 1)
    timestamp = series_to_supervised(timestamp, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    supply = np.array([float(d['supply']) for d in data_loaded])
    supply = supply[::-1]
    supply = supply.reshape(-1, 1)
    supply = series_to_supervised(supply, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    demand = np.array([float(d['demand']) for d in data_loaded])
    demand = demand[::-1]
    demand = demand.reshape(-1, 1)
    demand = series_to_supervised(demand, n_in=sequence_length_choice, n_out=output_sequence_length_choice)

    test_y = win_classification_prep(sell, buy, nb_buy_values, sequence_length_choice, window_length)

    test_x = sell.values[:, :sequence_length_choice]
    test_x = np.stack((test_x,
                        buy.values[:, :sequence_length_choice],
                        timestamp.values[:, :sequence_length_choice],
                        supply.values[:, :sequence_length_choice],
                        demand.values[:, :sequence_length_choice]), -1)

    model = Sequential()
    model.add(Conv1D(32, kernel_choice_1, input_shape=(train_x[0].shape)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(32, kernel_choice_2))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, kernel_choice_3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(dense_choice))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_uniform))
    model.add(Dense(1))
    model.add(Activation(activation_choice))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer_choice,
                  metrics=['accuracy'])

    model.fit(train_x, train_y,
              batch_size=64,
              epochs=20,
              verbose=2,
              validation_data=(test_x, test_y))
    score, acc = model.evaluate(test_x, test_y, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=createModel,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())
print("Best performing model chosen hyper-parameters:")
print(best_run)
