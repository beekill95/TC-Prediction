from .cbam import CBAM
from .unet import encoder_block, decoder_block

import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def UnetCBAM(
        input_shape=None,
        input_tensor=None,
        filters_block=[64, 128, 256, 512, 1024],
        output_classes=2,
        classifier_activation='softmax',
        decoder_shortcut_mode='concat', # 2 possible modes: 'concat' and 'add'
        include_top=True,
        model_name=None):

    if input_tensor is None:
        input_img = layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            input_img = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            input_img = input_tensor

    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    # Encoder part.
    encoder_blocks = []
    x = input_img
    for i, filters in enumerate(filters_block):
        orig = encoder_block(
                x,
                filters,
                pooling=(i != 0), # Shouldn't do pooling in the first encoder layer.
                has_shortcut=False, # Shortcut right now is broken!
                name=f'encoder_blk_{i}')
        x = CBAM(orig, gate_channels=filters, name=model_name + f'/encoder_{i}/SAM_{i}')

        # Skip connection,
        # we don't want the CBAM learned attention deteriorates original features.
        x = layers.Add(name=model_name + f'/encoder_{i}/residual_{i}')([x, orig])

        encoder_blocks.append(x)

    # Decoder part.
    for i, (encoder_output, filters) in enumerate(zip(encoder_blocks[:-1][::-1], filters_block[:-1][::-1])):
        orig = decoder_block(
                x,
                encoder_output,
                filters,
                decoder_shortcut_mode=decoder_shortcut_mode,
                has_shortcut=False, # Shortcut right now is broken!
                name=f'decoder_blk_{i}')
        x = CBAM(orig, gate_channels=filters, name=model_name + f'/decoder_{i}/SAM_{i}')

        # Skip connection,
        # we don't want the CBAM learned attention deteriorates original features.
        x = layers.Add(name=model_name + f'/decoder_{i}/residual_{i}')([x, orig])

    # The output part.
    if include_top:
        x = layers.Conv2D(
                output_classes,
                3,
                padding='SAME',
                name='out_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='out_bn')(x)

        if classifier_activation:
            x = layers.Activation(classifier_activation, name='activation_out')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = input_img

    model = keras.Model(inputs, outputs=x, name=model_name)
    return model
