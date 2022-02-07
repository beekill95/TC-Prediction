import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def AutoEncoders(input_shape, name=None):
    inputs = layers.Input(shape=input_shape)

    x = inputs
    x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            padding='SAME',
            name=name + '_1_conv')(x)
    x = layers.MaxPooling2D((2, 2), padding='SAME')(x)

    x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation='relu',
            padding='SAME',
            name=name + '_2_conv')(x)
    x = layers.MaxPooling2D((2, 2), padding='SAME')(x)

    x = layers.Conv2DTranspose(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            activation='relu',
            padding='SAME',
            name=name + '_3_transposed_conv')(x)
    x = layers.Conv2DTranspose(
            filters=64,
            kernel_size=(3, 3),
            strides=2,
            activation='relu',
            padding='SAME',
            name=name + '_4_transposed_conv')(x)

    x = layers.Conv2D(
            filters=input_shape[-1],
            kernel_size=1,
            name=name + '_output')(x)

    model = keras.Model(inputs, x)
    return model

