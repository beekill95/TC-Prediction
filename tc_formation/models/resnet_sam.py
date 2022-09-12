from . import sam
from . import cbam

import tensorflow.keras as keras
import tensorflow.keras.backend as backend
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers

def _ResNetSAM(stack_fn,
               preact,
               use_bias,
               model_name='resnetSAM',
               include_top=True,
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax',
               **kwargs):
    # TODO: add SAM module
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Args:
      stack_fn: a function that returns output tensor for the
        stacked residual blocks.
      preact: whether to use pre-activation or not
        (True for ResNetV2, False for ResNet and ResNeXt).
      use_bias: whether to use biases for convolutional layers or not
        (True for ResNet and ResNetV2, False for ResNeXt).
      model_name: string, model name.
      include_top: whether to include the fully-connected
        layer at the top of the network.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      **kwargs: For backwards compatibility only.
    Returns:
      A `keras.Model` instance.
    """
    if kwargs:
        raise ValueError('Unknown argument(s): %s' % (kwargs,))

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias,
                      name='conv1_conv')(x)

    if not preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes,
                         kernel_regularizer=regularizers.L2(1e-4),
                         activation=classifier_activation,
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Create model.
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = keras.Model(inputs, x, name=model_name)

    return model


def _block0(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block for resnet 18 and resnet 34

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(
        filters, kernel_size, strides=stride, padding='SAME', name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def _stack0(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = _block0(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = _block0(x, filters, conv_shortcut=False,
                    name=name + '_block' + str(i))

    return x


def _sam0(x, filters, name=None):
    """SAM module for ResNet18"""
    x = sam.SAM(
        x,
        residual_block=lambda x, name: _block0(x, filters, conv_shortcut=True, name=name),
        residual_blk_out_filters=filters,
        name=name)
    return x


def ResNet18SAM(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    """
    Instantiates the ResNet18 architecture.
    """

    def stack_fn(x):
        x = _stack0(x, 64, 2, stride1=1, name='conv2')
        x = _sam0(x, 64, name='sam2')
        x = _stack0(x, 128, 2, name='conv3')
        x = _sam0(x, 128, name='sam3')
        x = _stack0(x, 256, 2, name='conv4')
        x = _sam0(x, 256, name='sam4')
        x = _stack0(x, 512, 2, name='conv5')
        return x

    return _ResNetSAM(stack_fn, False, True, 'resnet18SAM', include_top,
                   input_tensor, input_shape, pooling, classes, **kwargs)


def ResNet14SAM(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    """
    Instantiates the ResNet14 architecture.
    """

    def stack_fn(x):
        x = _stack0(x, 64, 2, stride1=1, name='conv2')
        x = _sam0(x, 64, name='sam2')
        x = _stack0(x, 128, 2, name='conv3')
        x = _sam0(x, 128, name='sam3')
        x = _stack0(x, 256, 2, name='conv4')
        # x = _sam0(x, 256, name='sam4')
        # x = _stack0(x, 512, 2, name='conv5')
        return x

    return _ResNetSAM(stack_fn, False, True, 'resnet14SAM', include_top,
                   input_tensor, input_shape, pooling, classes, **kwargs)


# ========== CBAM =========


def ResNet18CBAM(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    """
    Instantiates the ResNet18 architecture.
    """

    def stack_fn(x):
        x = _stack0(x, 64, 2, stride1=1, name='conv2')
        x = cbam.CBAM(x, gate_channels=64, name='cbam2')
        x = _stack0(x, 128, 2, name='conv3')
        x = cbam.CBAM(x, gate_channels=128, name='cbam3')
        x = _stack0(x, 256, 2, name='conv4')
        x = cbam.CBAM(x, gate_channels=256, name='cbam4')
        x = _stack0(x, 512, 2, name='conv5')
        return x

    return _ResNetSAM(stack_fn, False, True, 'resnet18SAM', include_top,
                   input_tensor, input_shape, pooling, classes, **kwargs)

def ResNet14CBAM(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    """
    Instantiates the ResNet14 architecture.
    """

    def stack_fn(x):
        x = _stack0(x, 64, 2, stride1=1, name='conv2')
        x = cbam.CBAM(x, gate_channels=64, name='cbam2')
        x = _stack0(x, 128, 2, name='conv3')
        x = cbam.CBAM(x, gate_channels=128, name='cbam3')
        x = _stack0(x, 256, 2, name='conv4')
        # x = cbam.CBAM(x, gate_channels=256, name='cbam4')
        # x = _stack0(x, 512, 2, name='conv5')
        return x

    return _ResNetSAM(stack_fn, False, True, 'resnet14SAM', include_top,
                   input_tensor, input_shape, pooling, classes, **kwargs)
