import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler

with open('./ecto.json') as data_file:
    data = json.load(data_file)

def scale(data):
    scaler_array = []
    if len(data.shape) > 1:
        for i in range(0, data.shape[1]):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            arr = data[:, i]
            arr = arr.reshape(1, -1)
            scaler = scaler.fit(arr)
            arr = scaler.transform(arr)
            data[:, i] = arr.reshape(-1,)
            scaler_array.append(scaler)
        return scaler_array
    scaler = MinMaxScaler(feature_range=(-1, 1))
    arr = data
    arr = arr.reshape(1, -1)
    scaler = scaler.fit(arr)
    arr = scaler.transform(arr)
    data = arr.reshape(-1, )
    scaler_array.append(scaler)
    return scaler_array

def rescale(scaler,arr):
    arr = arr.reshape(-1, 1)
    arr = scaler.inverse_transform(arr)
    arr = arr.reshape(-1,)
    return arr

x_train = np.array([[int(d['timestamp']),
                     int(d['buy']),
                     int(d['sell']),
                     int(d['supply']),
                     int(d['demand'])] for d in data])

x_train = x_train[0:10000]
y_train = x_train[0:x_train.size:50]
x_train = x_train[::-1]
y_train = y_train[::-1]
y_train = y_train[:, 2]
x_train = np.delete(x_train, list(range(49, x_train.shape[0], 50)), axis=0)


# x_test_scalers = scale(x_test)
# y_test_scalers = scale(y_test)

x_train_scalers = scale(x_train)
y_train_scalers = scale(y_train)

x_train = x_train.reshape(200, -1, x_train.shape[1])
y_train = y_train.reshape(200, 1)

# x_test = x_test.reshape(50, -1, x_test.shape[1])
# y_test = y_test.reshape(50, 1)


batch_size = 10

tb_callback = TensorBoard(log_dir='./logs/run5', histogram_freq=10, batch_size=10, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

#expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(50,
               return_sequences=True,
               batch_input_shape=(batch_size, 49, 5),
               stateful=True))
model.add(LSTM(50, return_sequences=True, stateful=True))
model.add(LSTM(50, stateful=True))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.1,
          batch_size=batch_size,
          epochs=2000,
          shuffle=False,
          callbacks=[tb_callback])

# pred = model.predict(x_test, verbose=0, batch_size=10)
# pred = pred.reshape(1, -1)
# pred = y_test_scalers[0].inverse_transform(pred)
# y_pred = y_test.reshape(1, -1)
# correct_pred = y_test_scalers[0].inverse_transform(y_pred)
# print(pred)
# print(correct_pred)
