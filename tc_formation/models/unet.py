import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def Unet(
        input_shape=None,
        input_tensor=None,
        filters_block=[64, 128, 256, 512, 1024],
        output_classes=2,
        classifier_activation='softmax',
        decoder_shortcut_mode='add', # 2 possible modes: 'concat' and 'add'
        model_name=None):
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if input_tensor is None:
        input_img = layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            input_img = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            input_img = input_tensor

    # Encoder part.
    encoder_blocks = []
    x = input_img
    for i, filters in enumerate(filters_block):
        x = encoder_block(
                x,
                filters,
                pooling=(i != 0), # Shouldn't do pooling in the first encoder layer.
                has_shortcut=False, # Shortcut right now is broken!
                name=f'encoder_blk_{i}')
        encoder_blocks.append(x)

    # Decoder part.
    for i, (encoder_output, filters) in enumerate(zip(encoder_blocks[:-1][::-1], filters_block[:-1][::-1])):
        x = decoder_block(
                x,
                encoder_output,
                filters,
                decoder_shortcut_mode=decoder_shortcut_mode,
                has_shortcut=False, # Shortcut right now is broken!
                name=f'decoder_blk_{i}')

    # The output part.
    x = layers.Conv2D(output_classes, 3, padding='SAME', name='out_conv')(x)
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


def encoder_block(x, filters, kernel_size=3, pooling=True, has_shortcut=True, name=None):
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if pooling:
        x = layers.MaxPool2D((2, 2), strides=2, name=f'{name}_pool')(x)

    shortcut = x if has_shortcut else None

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='SAME',
        name=f'{name}_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=f'{name}_1_bn')(x)
    x = layers.Activation(
        'relu',
        name=f'{name}_1_relu')(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='SAME',
        name=f'{name}_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=f'{name}_2_bn')(x)
    x = layers.Activation(
        'relu',
        name=f'{name}_2_relu')(x)

    if shortcut is not None:
        x = layers.Add(name=f'{name}_add_shortcut')([x, shortcut])

    return x

def decoder_block(x, encoder_output, filters, kernel_size=3, decoder_shortcut_mode='add', has_shortcut=True, upsampling=True, name=None):
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
            encoder_output = layers.Conv2D(x.shape[-1], 3, padding='SAME', name=f'{name}_0_decoder_conv')(encoder_output)
            x = layers.Add(name=f'{name}_0_decoder_shortcut_add')([x, encoder_output])
        else:
            x = layers.Concatenate(axis=bn_axis, name=f'{name}_0_decoder_shortcut_concat')([x, encoder_output])

    shortcut = x if has_shortcut else None

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='SAME',
        name=f'{name}_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=f'{name}_1_bn')(x)
    x = layers.Activation(
        'relu',
        name=f'{name}_1_relu')(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='SAME',
        name=f'{name}_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=f'{name}_2_bn')(x)
    x = layers.Activation(
        'relu',
        name=f'{name}_2_relu')(x)

    if shortcut is not None:
        x = layers.Add(name=f'{name}_add_shortcut')([x, shortcut])

    return x

