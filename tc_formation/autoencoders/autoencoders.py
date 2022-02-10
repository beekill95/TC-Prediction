import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as backend
import tensorflow.keras.layers as layers

def AutoEncoders(input_shape, input_tensor=None, name=None):

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    inputs = (layers.Input(shape=input_shape)
              if input_tensor is None
              else input_tensor)
    x = inputs

    # Encoders part.
    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_1_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_1_pooling')(x)

    x = layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_2_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_2_pooling')(x)

    x = layers.Conv2D(
        256,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_3_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_3_bn')(x)
    x = layers.Activation('relu', name=name + '_3_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_3_pooling')(x)

    x = layers.Conv2D(
        512,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_4_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_4_bn')(x)
    x = layers.Activation('relu', name=name + '_4_relu')(x)

    x = layers.Conv2DTranspose(
            filters=256,
            kernel_size=(3, 3),
            strides=2,
            activation='relu',
            padding='SAME',
            name=name + '_decoder_1_transposed_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_decoder_1_bn')(x)
    x = layers.Activation('relu', name=name + '_decoder_1_relu')(x)

    x = layers.Conv2DTranspose(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            activation='relu',
            padding='SAME',
            name=name + '_decoder_2_transposed_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_decoder_2_bn')(x)
    x = layers.Activation('relu', name=name + '_decoder_2_relu')(x)

    x = layers.Conv2DTranspose(
            filters=64,
            kernel_size=(3, 3),
            strides=2,
            activation='relu',
            padding='SAME',
            name=name + '_decoder_3_transposed_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_decoder_3_bn')(x)
    x = layers.Activation('relu', name=name + '_decoder_3_relu')(x)

    x = layers.Conv2D(
            filters=input_shape[-1],
            kernel_size=1,
            name=name + '_decoder_output')(x)
    x = tf.image.resize(x, input_shape[:2])

    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    model = keras.Model(inputs, x)
    return model

