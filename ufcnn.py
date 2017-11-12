from keras.layers import Input, ZeroPadding1D, Conv1D, Activation, merge, BatchNormalization
from keras.models import Model


def ufcnn_model_concat(sequence_length=5000,
                       features=1,
                       nb_filter=150,
                       filter_length=5,
                       output_dim=1,
                       optimizer='adagrad',
                       loss='mse',
                       regression=True,
                       class_mode=None,
                       activation="softplus",
                       init="lecun_uniform"):
    # model = Graph()

    # model.add_input(name='input', input_shape=(None, features))

    main_input = Input(name='input', shape=(None, features))

    #########################################################

    # model.add_node(ZeroPadding1D(2), name='input_padding', input='input') # to avoid lookahead bias

    input_padding = ZeroPadding1D(2)(main_input)  # to avoid lookahead bias

    #########################################################

    # model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', init=init, input_shape=(sequence_length, features)), name='conv1', input='input_padding')
    # model.add_node(Activation(activation), name='relu1', input='conv1')

    conv1 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='valid',
                   kernel_initializer=init, activation=activation, name='conv1')(input_padding)

    #########################################################
    # model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init), name='conv2', input='relu1')
    # model.add_node(Activation(activation), name='relu2', input='conv2')

    conv2 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                   kernel_initializer=init, activation=activation, dilation_rate=2, name='conv2')(conv1)

    #########################################################
    # model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init), name='conv3', input='relu2')
    # model.add_node(Activation(activation), name='relu3', input='conv3')


    conv3 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                   kernel_initializer=init, activation=activation, dilation_rate=4, name='conv3')(conv2)

    #########################################################
    # model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init), name='conv4', input='relu3')
    # model.add_node(Activation(activation), name='relu4', input='conv4')

    conv4 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                   kernel_initializer=init, activation=activation, dilation_rate=8, name='conv4')(conv3)

    #########################################################
    # model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init), name='conv5', input='relu4')
    # model.add_node(Activation(activation), name='relu5', input='conv5')

    conv5 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                   kernel_initializer=init, activation=activation, dilation_rate=8, name='conv5')(conv4)

    #########################################################
    # model.add_node(Convolution1D(nb_filter=nb_filter,filter_length=filter_length, border_mode='same', init=init),
    #                 name='conv6',
    #                 inputs=['relu3', 'relu5'],
    #                 merge_mode='concat', concat_axis=-1)
    # model.add_node(Activation(activation), name='relu6', input='conv6')


    merge6 = merge([conv3, conv5], mode='concat')
    conv6 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                   kernel_initializer=init, activation=activation, dilation_rate=4, name='conv6')(merge6)

    #########################################################
    # model.add_node(Convolution1D(nb_filter=nb_filter,filter_length=filter_length, border_mode='same', init=init),
    #                 name='conv7',
    #                 inputs=['relu2', 'relu6'],
    #                 merge_mode='concat', concat_axis=-1)
    # model.add_node(Activation(activation), name='relu7', input='conv7')

    merge7 = merge([conv2, conv6], mode='concat')
    conv7 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                   kernel_initializer=init, activation=activation, dilation_rate=2, name='conv7')(merge7)

    #########################################################
    # model.add_node(Convolution1D(nb_filter=nb_filter,filter_length=filter_length, border_mode='same', init=init),
    #                 name='conv8',
    #                 inputs=['relu1', 'relu7'],
    #                 merge_mode='concat', concat_axis=-1)
    # model.add_node(Activation(activation), name='relu8', input='conv8')

    merge8 = merge([conv1, conv7], mode='concat')
    conv8 = Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same',
                   kernel_initializer=init, activation=activation, name='conv8')(conv7)

    #########################################################
    if regression:
        #########################################################
        # model.add_node(Convolution1D(nb_filter=output_dim, filter_length=sequence_length, border_mode='same', init=init), name='conv9', input='relu8')
        # model.add_output(name='output', input='conv9')


        conv9 = Conv1D(filters=output_dim, kernel_size=sequence_length, padding='same',
                       kernel_initializer=init, activation='linear', name='conv9')(conv8)
        output = conv9
        # main_output = conv9.output

    else:
        # model.add_node(Convolution1D(nb_filter=output_dim, filter_length=sequence_length, border_mode='same', init=init), name='conv9', input='relu8')
        # model.add_node(Activation('softmax'), name='activation', input='conv9')
        # model.add_output(name='output', input='activation')

        conv9 = Conv1D(filters=output_dim, kernel_size=sequence_length, padding='same',
                       kernel_initializer=init, activation='softmax', name='conv9')(conv8)
        # main_output = activation.output
        output = conv9

    # model.compile(optimizer=optimizer, loss={'output': loss})

    model = Model(input=main_input, output=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', ])

    print(model.summary())
    return model


# def ufcnn_model_concat_bn(sequence_length=5,
#                           features=1,
#                           nb_filter=150,
#                           filter_length=5,
#                           output_dim=1,
#                           optimizer='adagrad',
#                           loss='mse',
#                           regression=True,
#                           class_mode=None,
#                           activation="softplus",
#                           init="lecun_uniform",
#                           batch_norm=False):
#     def conv_block(input, nb_filter, filter_length, init, postfix, border_mode='same', subsample_length=2):
#         conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode=border_mode,
#                              subsample_length=subsample_length, init=init, name='conv' + postfix)(input)
#         relu = Activation(activation, name='relu' + postfix)(conv)
#         if batch_norm:
#             y = BatchNormalization(name='bn' + postfix)(relu)
#         else:
#             y = relu
#         return y
#
#     main_input = Input(name='input', shape=(None, features))
#
#     #########################################################
#
#     input_padding = ZeroPadding1D(2)(main_input)  # to avoid lookahead bias
#
#     #########################################################
#
#     # conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', init=init)(input_padding)
#     # relu1 = Activation(activation)(conv1)
#     H1 = conv_block(input_padding, nb_filter, filter_length, init, postfix='1', border_mode='valid', subsample_length=1)
#
#     #########################################################
#
#     # conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(relu1)
#     # relu2 = Activation(activation)(conv2)
#     H2 = conv_block(H1, nb_filter, filter_length, init, postfix='2', subsample_length=1)
#
#     #########################################################
#
#     # conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(relu2)
#     # relu3 = Activation(activation)(conv3)
#     H3 = conv_block(H2, nb_filter, filter_length, init, postfix='3', subsample_length=1)
#
#     #########################################################
#
#     # conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(relu3)
#     # relu4 = Activation(activation)(conv4)
#     H4 = conv_block(H3, nb_filter, filter_length, init, postfix='4', subsample_length=1)
#
#     #########################################################
#
#     # conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(relu4)
#     # relu5 = Activation(activation)(conv5)
#     G4 = conv_block(H4, nb_filter, filter_length, init, postfix='5', subsample_length=1)
#
#     #########################################################
#
#     merge6 = merge([H3, G4], mode='concat')
#     # conv6 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(merge6)
#     # relu6 = Activation(activation)(conv6)
#     G3 = conv_block(merge6, nb_filter, filter_length, init, postfix='6', subsample_length=1)
#
#     #########################################################
#
#     merge7 = merge([H2, G3], mode='concat')
#     # conv7 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(merge7)
#     # relu7 = Activation(activation)(conv7)
#     G2 = conv_block(merge7, nb_filter, filter_length, init, postfix='7', subsample_length=1)
#
#     #########################################################
#
#     merge8 = merge([H1, G2], mode='concat')
#     # conv8 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='same', init=init)(merge8)
#     # relu8 = Activation(activation)(conv8)
#     G1 = conv_block(merge8, nb_filter, filter_length, init, postfix='8', subsample_length=1)
#
#     #########################################################
#     if regression:
#         conv9 = Convolution1D(nb_filter=output_dim, filter_length=sequence_length, border_mode='same', init=init)(G1)
#         output = conv9
#     else:
#         conv9 = Convolution1D(nb_filter=output_dim, filter_length=sequence_length, border_mode='same', init=init)(G1)
#         activation = (Activation('softmax'))(conv9)
#         output = activation
#
#     model = Model(input=main_input, output=output)
#     model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', ])
#
#     print(model.summary())
#     return model
