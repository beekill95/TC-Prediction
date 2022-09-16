"""
Vision Transformer as described in
[An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929v2)
"""
import keras_nlp
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers


def ViT(*, input_shape=None,
        input_tensor=None,
        sequence_length=None,
        N=6,
        model_dim=768,
        attention_heads=12,
        include_top=True, logits=True, classes=1, name='ViT'):
    """
    Implement Vision Transformer.

    Parameters
    ----------
    input_shape: tuple of shape (T, size).
    input_tensor: input tensor to the network.
    N: number of encoder blocks. Default is 6.
    model_dim: dimension of the output of each encoder block.
    attention_heads: number of heads in multihead attention layer in each encoder block.
    include_top: should the top MLP layer be appended on top of the encoder. Default to True.
    logits: (only valid with include_top=True)
            should the output of the final MLP layer be logits or not. Default to True.
    classes: (only valid with include_top=True)
             number of output classes. Default to 1.
    name: name of the model.
    """
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # _, T, *_ = input_tensor.shape
    # print(input_tensor.shape)
    # We should flatten the input tensor if necessary.
    # if input_shape is not None:
    #     x = layers.Reshape((T, -1), name=name + '/reshape')(img_input)
    # else:
    #     x = img_input

    # Learn embedding for each image patch.
    x = img_input
    x = PatchEncoder(num_patches=sequence_length, projection_dim=model_dim)(x)
    # x = layers.TimeDistributed(
    #         layers.Dense(model_dim, name=name + '/embedding'),
    #         name=name + '/time_distributed',
    #     )(x)

    # Add positional encoding.
    # TODO: is this the correct layer to use for positional encoding.
    # pe = keras_nlp.layers.PositionEmbedding(sequence_length=sequence_length, name=name + '/pe')(x)
    # x = layers.Add(name=name + '/add_pe')([x, pe])

    # Construct model.
    for i in range(N):
        x = _encoder_block(x,
                output_size=model_dim,
                attention_heads=attention_heads,
                name=name + f'/encoder_{i}')

    # Add final MLP layer if necessary.
    if include_top:
        # Experiments:
        # Get the first element in the patch dimension.
        # This will learn the context of the whole image.
        # x = x[:, 0, :]

        # Or, we can just take the average across the second dimension.
        x = layers.GlobalAveragePooling1D(name=name + '/avg_patches')(x)

        # Pass this context vector to the MLP layer to learn the class output.
        x = layers.Dense(classes, name=name + '/output')(x)
        if not logits:
            x = layers.Activation('softmax', name=name + '/output/softmax')(x)

    # Create model.
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Return the model.
    model = keras.Model(inputs, x, name=name)
    return model


def _encoder_block(x, *, output_size=None, attention_heads=12, name=None):
    """
    Implement encoder transformer block as described in
    Figure 1 of [An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929v2)

    Parameters
    ----------
    x: input tensor of shape (B, T, S) where B is batch size,
       T is number of image patches, and S is size per patch.
    output_size: the size of the output of this block. If it is None,
                 then it will default to S.
    attention_heads: number of heads in the multihead attention layer.
    name: name of this block.
    """
    S = x.shape[-1]
    if output_size is None:
        output_size = S

    # Multihead attention.
    skip_conn = x
    x = layers.LayerNormalization(name=name + '/ln_1')(x)
    x = layers.MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=S, # TODO
        name=name + '/multihead_att',
    )(x, x)
    x = layers.Add(name=name + '/residual_conn_1')([x, skip_conn])

    # Linear layer.
    skip_conn = x
    x = layers.LayerNormalization(name=name + 'ln_2')(x)
    x = layers.Dense(output_size, name=name + '/dense')(x)

    # Output
    x = layers.Add(name=name + '/residual_conn_2')([x, skip_conn])

    return x


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
