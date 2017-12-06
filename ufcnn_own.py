from keras.layers import Input, ZeroPadding1D, Conv1D, Concatenate, Dense
from keras.models import Model


def ufcnn_model(sequence_length=5000,
                features=1,
                nb_filter=150,
                filter_length=5,
                output_dim=1,
                optimizer='adagrad',
                loss='mse',
                regression=True,
                class_mode=None,
                activation="softplus",
                init="lecun_uniform",
                resolution_levels=3):
    main_input = Input(name='input', shape=(None, features))

    #########################################################

    input_padding = ZeroPadding1D(2)(main_input)  # to avoid lookahead bias

    #########################################################

    H1 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='valid',
                kernel_initializer=init, activation=activation, name='H1')(input_padding)

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

    if regression:

        G0 = Conv1D(filters=output_dim, kernel_size=sequence_length, padding='same',
                    kernel_initializer=init, activation='linear', name='G0')(g_list[-1])
        main_output = G0
    else:

        G0 = Conv1D(filters=output_dim, kernel_size=sequence_length, padding='same',
                    kernel_initializer=init, activation='softmax', name='G0')(c_list[-1])
        main_output = G0

    #########################################################

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', ])

    print(model.summary())
    return model
