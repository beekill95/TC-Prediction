import tensorflow.keras as keras
import tensorflow.keras.backend as backend
import tensorflow.keras.layers as layers


def MultiscaleBaseline(input_shape, classes=1, output_activation=None, name=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    inputs = layers.Input(input_shape)
    x = inputs

    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        input_shape=input_shape,
        padding='SAME',
        name=name + '_1_conv',
    )(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_1_pooling')(x)

    # Output 1
    output_1 = x
    output_1 = layers.Conv2D(
        classes,
        kernel_size=(3, 3),
        padding='SAME',
        activation=output_activation,
        # name=name + '_1_output'
        name='output_1',
    )(output_1)

    # Block 2.
    x = layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_2_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_2_pooling')(x)

    # Output 2
    output_2 = x
    output_2 = layers.Conv2D(
        classes,
        kernel_size=(3, 3),
        padding='SAME',
        activation=output_activation,
        # name=name + '_2_output',
        name='output_2',
    )(output_2)

    # Block 3
    x = layers.Conv2D(
        256,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_3_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_3_bn')(x)
    x = layers.Activation('relu', name=name + '_3_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_3_pooling')(x)

    # Output 3
    output_3 = x
    output_3 = layers.Conv2D(
        classes,
        kernel_size=1,
        padding='SAME',
        activation=output_activation,
        # name=name + '_3_output',
        name='output_3',
    )(output_3)

    # Last block.
    x = layers.Conv2D(
        512,
        kernel_size=(3, 3),
        padding='SAME',
        name=name + '_4_conv',
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + '_4_bn')(x)
    x = layers.Activation('relu', name=name + '_4_relu')(x)
    x = layers.MaxPooling2D(2, strides=2, name=name + '_4_pooling')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        units=classes,
        activation=output_activation,
        # name=name + '_out',
        name='output',
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=dict(
            output=x,
            output_1=output_1,
            output_2=output_2,
            output_3=output_3,
        ))
