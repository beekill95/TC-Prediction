import tensorflow as tf
import tc_formation.models.unet as unet
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def UnetPriorTCProb(
        input_shape=None,
        input_tensor=None,
        filters_block=[64, 128, 256, 512, 1024],
        output_classes=2,
        classifier_activation='softmax',
        decoder_shortcut_mode='add', # 2 possible modes: 'concat' and 'add'
        model_name=None):
    # TODO: should we include Batch Normalization before the activation layer?
    # bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    base_net = unet.Unet(
        input_shape=input_shape,
        input_tensor=input_tensor,
        output_classes=output_classes,
        filters_block=filters_block,
        classifier_activation=classifier_activation,
        decoder_shortcut_mode=decoder_shortcut_mode,
        include_top=True,
    )

    x = base_net.get_layer('encoder_blk_4_2_relu').output
    x = layers.GlobalAveragePooling2D()(x)
    x1 = layers.Dense(output_classes, activation=classifier_activation, name='tc_prior')(x)
    x1_ = tf.expand_dims(x1, axis=-1)
    x1_ = tf.expand_dims(x1_, axis=-1)
    x2 = tf.squeeze(x1_ * base_net.outputs, axis=0)

    model = keras.Model(base_net.inputs, outputs={'tc_prior': x1, 'tc_posterior': x2}, name=model_name)
    return model

