from keras.layers import Input, ZeroPadding1D, Conv1D, Concatenate, Dense, Flatten
from keras.models import Model


def ufcnn_model(sequence_length=5000,
                features=1,
                nb_filter=150,
                filter_length=5,
                output_sequence_length=1,
                optimizer='adagrad',
                loss='mse',
                activation="softplus",
                init="lecun_uniform",
                resolution_levels=3):
    main_input = Input(name='input', shape=(sequence_length, features))

    #########################################################

    #input_padding = ZeroPadding1D(2)(main_input)  # to avoid lookahead bias

    #########################################################

    H1 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='causal',
                kernel_initializer=init, activation=activation, name='H1')(main_input)

    h_list = [H1]

    # create H-layers
    for i in range(2, resolution_levels + 1):
        h_list.append(Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                             kernel_initializer=init, activation=activation, dilation_rate=2 ** (i - 1),
                             name='H' + str(i))(h_list[i - 2]))

    #########################################################

    g_list = [Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                         kernel_initializer=init, activation=activation, dilation_rate=2 ** (resolution_levels - 1),
                         name='G' + str(resolution_levels))(h_list[resolution_levels-1])]

    c_list = list()

    for i in range(resolution_levels - 1, 0, -1):
        c_list.append(Concatenate(name='C' + str(i))([h_list[i-1], g_list[resolution_levels - (i + 1)]]))
        g_list.append(Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                             kernel_initializer=init, activation=activation, dilation_rate=2 ** i,
                             name='G' + str(i))(c_list[resolution_levels - (i + 1)]))

    #########################################################

        G0 = Conv1D(filters=nb_filter, kernel_size=sequence_length, padding='same',
                    kernel_initializer=init, activation='linear', name='G0')(g_list[-1])
        F0 = Flatten()(G0)
        D0 = Dense(output_sequence_length)(F0)
        main_output = D0

    #########################################################

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', ])

    print(model.summary())
    return model
