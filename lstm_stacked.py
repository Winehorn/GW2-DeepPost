from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU
from keras.callbacks import TensorBoard
from data_prep import gen_cosine_amp_for_supervised


batch_size = 128
sequence_length = 64

cos, expected = gen_cosine_amp_for_supervised(xn=sequence_length*100)
cos = cos.reshape((-1, 64, 1))
print(expected.shape)
expected = expected.reshape((-1, 1))

# tb_callback = TensorBoard(log_dir='./logs/cos/run1_lstm', histogram_freq=10, batch_size=10, write_graph=True, write_grads=True,
#                           write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#
# # expected input data shape: (batch_size, timesteps, data_dim)
# model = Sequential()
# model.add(LSTM(50,
#                return_sequences=True,
#                input_shape=(sequence_length, 1),
#                #batch_input_shape=(batch_size, sequence_length, 1),
#                stateful=False))
# model.add(LSTM(50, return_sequences=True, stateful=False))
# model.add(LSTM(50, stateful=False))
# model.add(Dense(50, activation='sigmoid'))
# model.add(Dense(1, activation='linear'))
#
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(cos, expected, validation_split=0.1,
#           batch_size=batch_size,
#           epochs=200,
#           shuffle=False,
#           callbacks=[tb_callback]
#           )
#
# predicted = model.predict(x=cos, batch_size=batch_size)

