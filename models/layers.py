import tensorflow as tf
import tensorflow.keras as keras


def features_gated_block(input_tensor):
    _, _, _, channels = input_tensor.shape
    gated_unit = keras.layers.Conv2D(
        channels,
        (1, 1),
        activation='sigmoid')

    feature_masks = gated_unit(input_tensor)
    return input_tensor * feature_masks
