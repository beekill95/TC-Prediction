from __future__ import annotations


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class ResidualBlock(layers.Layer):
    def __init__(self,
            filters: int,
            kernel_size: int = 3,
            stride1: int = 1,
            kernel_regularizer: keras.regularizers.Regularizer | None = None,
            name: str | None = None,
            *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)

        self._filters = filters
        self._stride1 = stride1
        self._kernel_size = kernel_size
        self._kernel_regularizer = kernel_regularizer

        self._conv_1 = keras.Sequential([
            layers.Conv2D(
                filters,
                kernel_size,
                strides=stride1,
                padding='SAME',
                kernel_regularizer=kernel_regularizer),
            layers.LayerNormalization(axis=-1),
            layers.Activation('relu'),
        ], name=f'{name}_conv1')
        self._conv_2 = keras.Sequential([
            layers.Conv2D(
                filters,
                kernel_size,
                strides=1,
                padding='SAME',
                kernel_regularizer=kernel_regularizer),
            layers.LayerNormalization(axis=-1),
        ], name=f'{name}_conv2')

    def build(self, input_shape):
        prev_filters = input_shape[-1]

        # If the number of filters in the previous filters
        # is different from the current filters,
        # we will need a 1x1 2d Conv layer to match the dimension.
        if prev_filters != self._filters:
            self._conv_shortcut = keras.Sequential([
                layers.Conv2D(
                    self._filters,
                    1,
                    strides=self._stride1,
                    kernel_regularizer=self._kernel_regularizer),
                layers.LayerNormalization(axis=-1),
            ], name=f'{self.name}_conv_shortcut')
        else:
            self._conv_shortcut = None

    def call(self, inputs):
        shortcut = (self._conv_shortcut(inputs)
                    if self._conv_shortcut is not None
                    else inputs)

        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = x + shortcut
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        return dict(
            filters=self._filters,
            kernel_size=self._kernel_size,
            stride1=self._stride1,
            kernel_regularizer=self._kernel_regularizer,
            name=self.name,
        )


class BottleneckResidualBlock(layers.Layer):
    def __init__(self,
            filters: int,
            kernel_size: int = 3,
            stride1: int = 1,
            kernel_regularizer: keras.regularizers.Regularizer | None = None,
            name: str | None = None,
            *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)
        assert filters % 4 == 0

        self._filters = filters
        self._stride1 = stride1
        self._kernel_size = kernel_size
        self._kernel_regularizer = kernel_regularizer

        self._conv_1 = keras.Sequential([
            layers.Conv2D(
                filters // 4,
                1,
                strides=stride1,
                padding='SAME',
                kernel_regularizer=kernel_regularizer,
                name=f'{name}/conv1/conv'),
            layers.LayerNormalization(
                axis=-1,
                name=f'{name}/conv1/ln'),
            layers.Activation('relu'),
        ], name=f'{name}_conv1')

        self._conv_2 = keras.Sequential([
            layers.Conv2D(
                filters // 4,
                kernel_size,
                strides=1,
                padding='SAME',
                kernel_regularizer=kernel_regularizer,
                name=f'{name}/conv2/conv'),
            layers.LayerNormalization(
                axis=-1,
                name=f'{name}/conv2/ln'),
        ], name=f'{name}_conv2')

        self._conv_3 = keras.Sequential([
            layers.Conv2D(
                filters,
                1,
                padding='SAME',
                kernel_regularizer=kernel_regularizer,
                name=f'{name}/conv3/conv'),
            layers.LayerNormalization(
                axis=-1,
                name=f'{name}/conv3/ln'),
            layers.Activation('relu'),
        ], name=f'{name}_conv3')

    def build(self, input_shape):
        prev_filters = input_shape[-1]

        # If the number of filters in the previous filters
        # is different from the current filters,
        # we will need a 1x1 2d Conv layer to match the dimension.
        if prev_filters != self._filters:
            name = self.name
            self._conv_shortcut = keras.Sequential([
                layers.Conv2D(
                    self._filters,
                    1,
                    strides=self._stride1,
                    kernel_regularizer=self._kernel_regularizer,
                    name=f'{name}/conv_shortcut/conv'),
                layers.LayerNormalization(
                    axis=-1,
                    name=f'{name}/conv_shortcut/ln'),
            ], name=f'{name}_conv_shortcut')
        else:
            self._conv_shortcut = None

    def call(self, inputs):
        shortcut = (self._conv_shortcut(inputs)
                    if self._conv_shortcut is not None
                    else inputs)

        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = self._conv_3(x)
        x = x + shortcut
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        return dict(
            filters=self._filters,
            kernel_size=self._kernel_size,
            stride1=self._stride1,
            kernel_regularizer=self._kernel_regularizer,
            name=self.name,
        )
