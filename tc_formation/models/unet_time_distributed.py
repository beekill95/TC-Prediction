import tc_formation.models.unet as unet
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def UnetTimeDistributed(
        input_shape=None,
        filters_block=[64, 128, 256, 512, 1024],
        output_classes=2,
        classifier_activation='softmax',
        decoder_shortcut_mode='add', # 2 possible modes: 'concat' and 'add'
        model_name=None):
    bn_axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1

    base_net = unet.Unet(
            input_shape=input_shape[1:],
            filters_block=filters_block,
            classifier_activation='softmax',
            decoder_shortcut_mode=decoder_shortcut_mode,
            include_top=False)

    inputs = layers.Input(input_shape)
    x = layers.TimeDistributed(base_net)(inputs)

    # Concatenate along the time dimension.
    timestep_embeddings = [x[:, i] for i in range(input_shape[0])]
    x = layers.Concatenate()(timestep_embeddings)

    # Then, add the final output channel.
    x = layers.Conv2D(
            output_classes,
            3,
            # kernel_regularizer=keras.regularizers.l2(0.01),
            padding='SAME',
            name='out_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='out_bn')(x)
    if classifier_activation:
        x = layers.Activation(classifier_activation, name='activation_out')(x)

    model = keras.Model(inputs, outputs=x, name=model_name)
    return model
