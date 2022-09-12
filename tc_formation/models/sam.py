"""
Spatial Attention Module as described in
[Residual Attention Network for Image Classification](https://ieeexplore.ieee.org/document/8100166)
"""
import tensorflow.keras.backend as backend
import tensorflow.keras.layers as layers


def SAM(x, *, residual_block, residual_blk_out_filters, p=1, r=2, t=1, name=None):
    """
    Construct a Spatial Attention Module.

    Parameters
    ----------
    x: input layer
    residual_block: a function that receives two parameters:
        + x: which is the input layer
        + name: the name of the block
    residual_blk_out_filters: number of filters in the output of residual block.
    p: number of residual blocks before and after splitting into trunk and soft mask branch.
    r: number of residual blocks in the soft mask branch.
    t: number of residual blocks in the trunk branch.
    name: name of the current SAM unit.
    """
    # Before splitting into trunk and soft mask branch.
    for i in range(p):
        x = residual_block(x, name=name + f'/pre_residual_blk_{i}')

    # Now, we will split into trunk branch and soft mask branch.
    trunk_out = _trunk_branch(x,
                              t=t,
                              residual_block=residual_block,
                              name=name + '/trunk')
    soft_mask_out = _soft_mask_branch(x,
                                      r=r,
                                      residual_block=residual_block,
                                      residual_blk_out_filters=residual_blk_out_filters,
                                      name=name + '/mask')

    # Soft mask's branch output will be used to attend trunk branch's output.
    x = layers.Multiply(name=name + '/apply_mask')([trunk_out, soft_mask_out])

    # We don't want the attended output is worst than the original one.
    x = layers.Add(name=name + '/add_trunk')([trunk_out, x])

    # After merging trunk and soft mask branch's output,
    # we need some more residual blocks.
    for i in range(p):
        x = residual_block(x, name=name + f'/post_residual_blk_{i}')

    return x


def SimplifiedSAM():
    pass


def _trunk_branch(x, *, residual_block, t, name=None):
    for i in range(t):
        x = residual_block(x, name=name + f'/residual_blk_{i}')

    return x


def _soft_mask_branch(x, *, residual_block, residual_blk_out_filters, r, name=None):
    # Down sample.
    orig_x = x
    x = layers.MaxPooling2D((2, 2), 2, name=name + '/max_pool_0')(x)

    for i in range(r):
        x = residual_block(x, name=name + f'/pre/residual_blk_{i}')

    skip_connection = x
    x = layers.MaxPooling2D((2, 2), 2, name=name + '/max_pool_1')(x)

    for i in range(2 * r):
        x = residual_block(x, name=name + f'/downsampled/residual_blk_{i}')

    # Up sample.
    # x = layers.UpSampling2D((2, 2), name=name + '/up_sample_0')(x)
    _, h, w, _ = skip_connection.shape
    x = layers.Resizing(h, w, name=name + '/up_sample_0')(x)

    # Add skip connection.
    x = x + skip_connection

    for i in range(r):
        x = residual_block(x, name=name + f'/post/residual_blk_{i}')

    # x = layers.UpSampling2D((2, 2), name=name + '/up_sample_1')(x)
    _, h, w, _ = orig_x.shape
    x = layers.Resizing(h, w, name=name + '/up_sample_1')(x)

    # Output.
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.Conv2D(residual_blk_out_filters, 1, name=name + '/post/conv1_0')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/post/conv1_bn0')(x)
    x = layers.ReLU(name=name + '/post/conv1_relu0')(x)

    x = layers.Conv2D(residual_blk_out_filters, 1, name=name + '/post/conv1_1')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '/post/conv1_bn1')(x)
    x = layers.Activation('sigmoid', name=name + '/post/sigmoid_out')(x)
    
    return x
