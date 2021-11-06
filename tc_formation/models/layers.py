import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def features_gated_block(input_tensor):
    _, _, _, channels = input_tensor.shape
    gated_unit = keras.layers.Conv2D(
        channels,
        (1, 1),
        activation='sigmoid')

    feature_masks = gated_unit(input_tensor)
    return input_tensor * feature_masks


def attention_layer(input_tensor, spatial_attention=True, channel_attention=True, name=None):
    att = input_tensor

    if spatial_attention:
        spatial_sum = tf.reduce_sum(
            input_tensor, axis=-1, keepdims=True, name=name + '_spatial_sum')
        batch_sum = tf.reduce_sum(
            spatial_sum, axis=[1, 2, 3], keepdims=True, name=name + '_spatial_batch_sum')
        spatial_mask = tf.divide(
            spatial_sum, batch_sum, name=name + '_spatial_mask')
        att = tf.multiply(att, spatial_mask, name=name + '_spatial_attention')

    if channel_attention:
        channel_sum = tf.reduce_sum(
            input_tensor, axis=[1, 2], keepdims=True, name=name + '_channel_sum')
        batch_sum = tf.reduce_sum(
            channel_sum, axis=[1, 2, 3], keepdims=True, name=name + '_channel_batch_sum')
        channel_mask = tf.divide(
            channel_sum, batch_sum, name=name + '_channel_mask')
        att = tf.multiply(att, channel_mask, name=name + '_channel_attention')

    return att


def tc_position_regression_layers(input_layer, hidden_layers=[2048, 2048], output_activation=None, name=None):
    x = input_layer

    for i, layer_size in enumerate(hidden_layers):
        x = keras.layers.Dense(
            units=layer_size,
            name=f'{name}_tc_position_regression_dense_{i}',
            activation='relu')(x)

    x = keras.layers.Dense(
        units=2,
        name=f'{name}_tc_position_regression_output',
        activation=output_activation)(x)

    return x


def tc_formation_prediction_layers(input_layer, hidden_layers=[2048, 2048], output_activation=None, name=None):
    x = input_layer

    for i, layer_size in enumerate(hidden_layers):
        x = keras.layers.Dense(
            units=layer_size,
            name=f'{name}_tc_formation_pred_dense_{i}',
            activation='relu')(x)

    x = keras.layers.Dense(
        units=1,
        name=f'{name}_tc_formation_pred_output',
        activation=output_activation)(x)

    return x


if __name__ == '__main__':
    inp = keras.Input((4, 10, 3))

    layer = inp
    layer = keras.layers.Conv2D(64, (3, 3), name='conv_3')(layer)
    layer = attention_layer(layer, name='att_test')

    model = keras.Model(inputs=inp, outputs=layer)
    model.summary()
