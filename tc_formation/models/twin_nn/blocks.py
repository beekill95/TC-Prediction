from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers



def BaseBlock(input_shape, starting_channels:int = 64, name:str =None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    inputs = layers.Input(input_shape)
    x = inputs

    output_depth = starting_channels
    x = layers.Conv2D(
        output_depth,
        kernel_size=(3, 3),
        input_shape=input_shape,
        padding='SAME',
        name=name + '_1_conv',
    )(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_1_pooling')(x)

    output_depth *= 2
    x = layers.Conv2D(
        output_depth,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_2_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_2_pooling')(x)

    output_depth *= 2
    x = layers.Conv2D(
        output_depth,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_3_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_3_bn')(x)
    x = layers.Activation('relu', name=name + '_3_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_3_pooling')(x)

    output_depth *= 2
    x = layers.Conv2D(
        output_depth,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_4_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_4_bn')(x)
    x = layers.Activation('relu', name=name + '_4_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_4_pooling')(x)

    output = layers.GlobalAveragePooling2D()(x)
    model = keras.Model(inputs, output)

    return model


def FullyConnectedBlock(base_model: keras.Model, hidden_layers: list, name:str = None):
    x = base_model.output
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation='relu', name=f'{name}_fully_connected_{i}')(x)

    output_layer = layers.Dense(1, name=f'{name}_output')
    x = output_layer(x)
    model = keras.Model(base_model.inputs, x, name=name)

    return model, output_layer
