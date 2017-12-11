from keras.layers import Input, ZeroPadding1D, Conv1D, Concatenate, Dense, Flatten, Dropout, Add, BatchNormalization
from keras.models import Model


def ufcnn_model_fulldropout(sequence_length=5000,
                features=1,
                nb_filter=150,
                filter_length=5,
                output_sequence_length=1,
                dropout=0.0,
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

    HD1 = Dropout(dropout, name="HD1")(H1)

    h_list = [H1]
    hd_list = [HD1]

    # create H-layers
    for i in range(2, resolution_levels + 1):
        h_list.append(Conv1D(filters=nb_filter, kernel_size=filter_length, padding='causal',
                             kernel_initializer=init, activation=activation, dilation_rate=2 ** (i - 1),
                             name='H' + str(i))(hd_list[i - 2]))
        hd_list.append(Dropout(dropout, name='HD' + str(i))(h_list[i - 1]))

    #########################################################

    g_list = [Conv1D(filters=nb_filter, kernel_size=filter_length, padding='causal',
                         kernel_initializer=init, activation=activation, dilation_rate=2 ** (resolution_levels - 1),
                         name='G' + str(resolution_levels))(hd_list[resolution_levels-1])]
    
    gd_list = [Dropout(dropout, name='GD' + str(resolution_levels))(g_list[0])]

    c_list = list()

    for i in range(resolution_levels - 1, 0, -1):
        c_list.append(Add(name='C' + str(i))([hd_list[i-1], gd_list[resolution_levels - (i + 1)]]))
        g_list.append(Conv1D(filters=nb_filter, kernel_size=filter_length, padding='causal',
                             kernel_initializer=init, activation=activation, dilation_rate=2 ** i,
                             name='G' + str(i))(c_list[resolution_levels - (i + 1)]))
        gd_list.append(Dropout(dropout, name='GD' + str(i))(g_list[resolution_levels - (i)]))

    #########################################################

    G0 = Conv1D(filters=nb_filter, kernel_size=sequence_length, padding='causal',
                    kernel_initializer=init, activation=activation, name='G0')(gd_list[-1])
    GD0 = Dropout(dropout, name='GD0')(G0)
    F0 = Flatten(name='F0')(GD0)
    D0 = Dense(output_sequence_length, name="D0")(F0)
    main_output = D0

    #########################################################

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', ])

    print(model.summary())
    return model


def ufcnn_model_bn(sequence_length=5000,
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

    # input_padding = ZeroPadding1D(2)(main_input)  # to avoid lookahead bias

    #########################################################

    H1 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='causal',
                kernel_initializer=init, activation=activation, name='H1')(main_input)

    HBN1 = BatchNormalization(name="HBN1")(H1)

    h_list = [H1]
    hbn_list = [HBN1]

    # create H-layers
    for i in range(2, resolution_levels + 1):
        h_list.append(Conv1D(filters=nb_filter, kernel_size=filter_length, padding='causal',
                             kernel_initializer=init, activation=activation, dilation_rate=2 ** (i - 1),
                             name='H' + str(i))(hbn_list[i - 2]))
        hbn_list.append(BatchNormalization(name='HBN' + str(i))(h_list[i - 1]))

    #########################################################

    g_list = [Conv1D(filters=nb_filter, kernel_size=filter_length, padding='causal',
                     kernel_initializer=init, activation=activation, dilation_rate=2 ** (resolution_levels - 1),
                     name='G' + str(resolution_levels))(hbn_list[resolution_levels - 1])]

    gbn_list = [BatchNormalization(name='GBN' + str(resolution_levels))(g_list[0])]

    c_list = list()

    for i in range(resolution_levels - 1, 0, -1):
        c_list.append(Add(name='C' + str(i))([hbn_list[i - 1], gbn_list[resolution_levels - (i + 1)]]))
        g_list.append(Conv1D(filters=nb_filter, kernel_size=filter_length, padding='causal',
                             kernel_initializer=init, activation=activation, dilation_rate=2 ** i,
                             name='G' + str(i))(c_list[resolution_levels - (i + 1)]))
        gbn_list.append(BatchNormalization(name='GBN' + str(i))(g_list[resolution_levels - (i)]))

    #########################################################

    G0 = Conv1D(filters=nb_filter, kernel_size=sequence_length, padding='causal',
                kernel_initializer=init, activation=activation, name='G0')(gbn_list[-1])
    GBN0 = BatchNormalization(name='GBN0')(G0)
    F0 = Flatten(name='F0')(GBN0)
    D0 = Dense(output_sequence_length, name="D0")(F0)
    main_output = D0

    #########################################################

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', ])

    print(model.summary())
    return model
