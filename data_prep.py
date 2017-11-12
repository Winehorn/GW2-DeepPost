import json
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
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
            data[:, i] = arr.reshape(-1, )
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


def rescale(scaler, arr):
    arr = arr.reshape(-1, 1)
    arr = scaler.inverse_transform(arr)
    arr = arr.reshape(-1, )
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


def gen_cosine_amp_for_supervised(amp=100, period=25, x0=0, xn=50000, step=1, k=0.0001, lahead=1):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing
    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
        lahead: number of timesteps between input and output
        Ernst 20160301 from https://github.com/fchollet/keras/blob/master/examples/stateful_lstm.py
        as a first test for the ufcnn
    """

    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(idx / (2 * np.pi * period))
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)

    expected_output = np.zeros((len(cos), 1, 1))
    for i in range(len(cos) - lahead):
        expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])
    return cos, expected_output


def print_data(expected, predicted):
    print('Ploting Results')
    plt.figure(figsize=(18, 3))
    # plt.subplot(2, 1, 1)
    plt.plot(expected.reshape(-1)[-10000:])
    # plt.title('Expected')
    # plt.subplot(2, 1, 2)
    plt.plot(predicted.reshape(-1)[-10000:])
    # plt.title('Predicted')
    # plt.savefig('sinus.png')
    plt.show()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
