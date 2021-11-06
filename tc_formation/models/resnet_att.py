"""
Create Resnet with Attention capability.
"""
from . import resnet
from . import layers


def ResNet50Att(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                spatial_attention=True,
                channel_attention=True,
                **kwargs):
    """
    Instantiates the ResNet50 architecture with attention layers.
    Copied and modified from Keras's Resnet source code.
    """
    ...

    def stack_fn(x):
        x = resnet._stack1(x, 64, 3, stride1=1, name='conv2')
        x = layers.attention_layer(
            x,
            spatial_attention=spatial_attention,
            channel_attention=channel_attention,
            name='conv2_attention')
        x = resnet._stack1(x, 128, 4, name='conv3')
        x = layers.attention_layer(
            x,
            spatial_attention=spatial_attention,
            channel_attention=channel_attention,
            name='conv3_attention')
        x = resnet._stack1(x, 256, 6, name='conv4')
        x = layers.attention_layer(
            x,
            spatial_attention=spatial_attention,
            channel_attention=channel_attention,
            name='conv4_attention')
        x = resnet._stack1(x, 512, 3, name='conv5')
        x = layers.attention_layer(
            x,
            spatial_attention=spatial_attention,
            channel_attention=channel_attention,
            name='conv5_attention')
        return x

    return resnet._ResNet(stack_fn, False, True, 'resnet50att', include_top,
                          input_tensor, input_shape, pooling, classes, **kwargs)
