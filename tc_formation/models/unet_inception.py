import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def UnetInception(
        input_shape=None,
        input_tensor=None,
        filters_block=[64, 128, 256, 512, 1024],
        output_classes=2,
        classifier_activation='softmax',
        decoder_shortcut_mode='add', # 2 possible modes: 'concat' and 'add'
        model_name=None,
        include_top=True):
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if input_tensor is None:
        input_img = layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            input_img = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            input_img = input_tensor

    x = input_img

    x = layers.Conv2D(64, 7, strides=(1, 1), padding='same', name='pre_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name='pre_1_bn')(x)
    x = layers.Activation(
        'relu',
        name='pre_1_relu')(x)

    # Encoder part.
    encoder_blocks = []

    # Encoder blk 1: output channels: 256
    x = encoder_block(
            x,
            conv1_filters=64,
            conv3_reduction_filters=96,
            conv3_filters=128,
            conv5_reduction_filters=16,
            conv5_filters=32,
            max_pool_proj_filters=32,
            pooling=False,
            name='encoder_blk_1')
    encoder_blocks.append(x)

    # Encoder blk 2: output channels: 480
    x = encoder_block(
            x,
            conv1_filters=128,
            conv3_reduction_filters=128,
            conv3_filters=192,
            conv5_reduction_filters=32,
            conv5_filters=96,
            max_pool_proj_filters=64,
            name='encoder_blk_2')
    encoder_blocks.append(x)

    # Encoder blk 3: output channels: 512
    x = encoder_block(
            x,
            conv1_filters=192,
            conv3_reduction_filters=96,
            conv3_filters=208,
            conv5_reduction_filters=16,
            conv5_filters=48,
            max_pool_proj_filters=64,
            name='encoder_blk_3')
    encoder_blocks.append(x)

    # Decoder part.
    # Decoder blk 1: output_channels: 960
    x = decoder_block(
            x,
            encoder_blocks[1],
            conv1_filters=128,
            conv3_reduction_filters=128,
            conv3_filters=192,
            conv5_reduction_filters=32,
            conv5_filters=96,
            max_pool_proj_filters=64,
            name='decoder_blk_1')

    # Decoder blk 2: output_channels: 512
    x = decoder_block(
            x,
            encoder_blocks[0],
            conv1_filters=64,
            conv3_reduction_filters=96,
            conv3_filters=128,
            conv5_reduction_filters=16,
            conv5_filters=32,
            max_pool_proj_filters=32,
            name='decoder_blk_2')

    # The output part.
    if include_top:
        x = layers.Conv2D(
                output_classes,
                3,
                kernel_regularizer=keras.regularizers.l2(0.01),
                padding='SAME',
                name='out_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='out_bn')(x)
        if classifier_activation:
            x = layers.Activation(classifier_activation, name='activation_out')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = input_img

    model = keras.Model(inputs, outputs=x, name=model_name)
    return model

def dialated_inception_block(
        x,
        conv1_filters,
        conv3_reduction_filters,
        conv3_filters,
        conv5_reduction_filters,
        conv5_filters,
        max_pool_proj_filters,
        name=None):
    path1 = layers.Conv2D(conv1_filters, 1, activation=None, name=f'{name}_conv_1')(x)

    path2 = layers.Conv2D(conv3_reduction_filters, 1, activation=None, name=f'{name}_conv3_reduction')(x)
    path2 = layers.Conv2D(conv3_filters, 3, padding='SAME', activation=None, name=f'{name}_conv3')(path2)

    path3 = layers.Conv2D(conv5_reduction_filters, 1, activation=None, name=f'{name}_conv3_dialated_reduction')(x)
    path3 = layers.Conv2D(conv5_filters, 3, dilation_rate=(2, 2), padding='SAME', activation=None, name=f'{name}_conv3_dialated')(path3)

    path4 = layers.MaxPool2D((3, 3), strides=1, padding='SAME', name=f'{name}_max_pool')(x)
    path4 = layers.Conv2D(max_pool_proj_filters, 1, activation=None, name=f'{name}_max_pool_proj')(path4)

    return layers.Concatenate()([path1, path2, path3, path4])


def encoder_block(
        x,
        conv1_filters,
        conv3_reduction_filters,
        conv3_filters,
        conv5_reduction_filters,
        conv5_filters,
        max_pool_proj_filters,
        pooling=True,
        has_shortcut=False,
        name=None):
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if pooling:
        x = layers.MaxPool2D((2, 2), strides=2, name=f'{name}_pool')(x)

    shortcut = x if has_shortcut else None

    x = dialated_inception_block(
            x,
            conv1_filters=conv1_filters,
            conv3_reduction_filters=conv3_reduction_filters,
            conv3_filters=conv3_filters,
            conv5_reduction_filters=conv5_reduction_filters,
            conv5_filters=conv5_filters,
            max_pool_proj_filters=max_pool_proj_filters,
            name=f'{name}_inception_blk_1')
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=f'{name}_1_bn')(x)
    x = layers.Activation(
        'relu',
        name=f'{name}_1_relu')(x)

    # x = layers.Conv2D(
    #     filters,
    #     kernel_size,
    #     # kernel_regularizer=keras.regularizers.l2(0.01),
    #     padding='SAME',
    #     name=f'{name}_2_conv')(x)
    # x = layers.BatchNormalization(
    #     axis=bn_axis,
    #     epsilon=1.001e-5,
    #     name=f'{name}_2_bn')(x)
    # x = layers.Activation(
    #     'relu',
    #     name=f'{name}_2_relu')(x)

    if shortcut is not None:
        x = layers.Add(name=f'{name}_add_shortcut')([x, shortcut])

    return x

def decoder_block(
        x,
        encoder_output,
        conv1_filters,
        conv3_reduction_filters,
        conv3_filters,
        conv5_reduction_filters,
        conv5_filters,
        max_pool_proj_filters,
        decoder_shortcut_mode='concat',
        has_shortcut=False,
        upsampling=True,
        name=None):
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if upsampling:
        x = layers.UpSampling2D((2, 2), name=f'{name}_0_pool')(x)

    # Concatenate the decoder output and encoder output
    if encoder_output is not None:
        if encoder_output.shape[1] != x.shape[1]:
            x = layers.ZeroPadding2D(((0, 1), (0, 0)), name=f'{name}_0_height_pad')(x)
        if encoder_output.shape[2] != x.shape[2]:
            x = layers.ZeroPadding2D(((0, 0), (0, 1)), name=f'{name}_0_width_pad')(x)
        if decoder_shortcut_mode == 'add':
            encoder_output = layers.Conv2D(
                    x.shape[-1],
                    3,
                    # kernel_regularizer=keras.regularizers.l2(0.01),
                    padding='SAME',
                    name=f'{name}_0_decoder_conv')(encoder_output)
            x = layers.Add(name=f'{name}_0_decoder_shortcut_add')([x, encoder_output])
        else:
            x = layers.Concatenate(axis=bn_axis, name=f'{name}_0_decoder_shortcut_concat')([x, encoder_output])

    shortcut = x if has_shortcut else None

    x = dialated_inception_block(
            x,
            conv1_filters=conv1_filters,
            conv3_reduction_filters=conv3_reduction_filters,
            conv3_filters=conv3_filters,
            conv5_reduction_filters=conv5_reduction_filters,
            conv5_filters=conv5_filters,
            max_pool_proj_filters=max_pool_proj_filters,
            name=f'{name}_inception_blk_1')
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=f'{name}_1_bn')(x)
    x = layers.Activation(
        'relu',
        name=f'{name}_1_relu')(x)

    # x = layers.Conv2D(
    #     filters,
    #     kernel_size,
    #     # kernel_regularizer=keras.regularizers.l2(0.01),
    #     padding='SAME',
    #     name=f'{name}_2_conv')(x)
    # x = layers.BatchNormalization(
    #     axis=bn_axis,
    #     epsilon=1.001e-5,
    #     name=f'{name}_2_bn')(x)
    # x = layers.Activation(
    #     'relu',
    #     name=f'{name}_2_relu')(x)

    if shortcut is not None:
        x = layers.Add(name=f'{name}_add_shortcut')([x, shortcut])

    return x
