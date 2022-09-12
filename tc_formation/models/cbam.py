"""
CBAM module as described in
[CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521v2)
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def CBAM(x, *, gate_channels, reduction=16, name=None):
    channel_att = _channel_attention(
        x, gate_channels=gate_channels, reduction=reduction, name=name)
    x = layers.Multiply(name=name + '/apply_channel_att')([x, channel_att])

    spatial_att = _spatial_attention(x, name=name)
    x = layers.Multiply(name=name + '/apply_spatial_att')([x, spatial_att])
    return x


def _channel_attention(x, *, gate_channels, reduction, name=None):
    max_pool = layers.GlobalMaxPool2D(name=name + '/channel_att/max_pool')(x)
    avg_pool = layers.GlobalAveragePooling2D(name=name + '/channel_att/avg_pool')(x)

    mlp = keras.Sequential([
        layers.Dense(gate_channels // reduction, activation='relu', name=name + '/channel_att/mlp/dense_1'),
        layers.Dense(gate_channels, name=name + '/channel_att/mlp/dense_2'),
    ], name=name + '/channel_att/mlp')

    pool = layers.Add(name=name + '/channel_att/add')([mlp(max_pool), mlp(avg_pool)])
    att = layers.Activation('sigmoid', name=name + '/channel_att/sigmoid')(pool)
    return att


def _spatial_attention(x, name=None):
    kernel_size = 7

    max_pool = tf.reduce_max(x, axis=-1, keepdims=True, name=name + '/spatial_att/max_pool')
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True, name=name + '/spatial_att/avg_pool')
    pool = layers.Concatenate(name=name + '/spatial_att/concat')([max_pool, avg_pool])
    attention = layers.Conv2D(1,
        kernel_size,
        padding='SAME',
        activation='sigmoid',
        name=name + '/spatial_att/conv')(pool)

    return attention
