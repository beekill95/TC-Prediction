import tensorflow as tf
import tensorflow.keras.layers as layers


class Patches(layers.Layer):
    def __init__(self, patch_size, flatten=True, padding='SAME', *args, **kwargs):
        super(Patches, self).__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.flatten = flatten
        self.padding = padding

    def call(self, images):
        batch_size = tf.shape(images)[0]
        channels = images.shape[-1]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding,
        )

        if self.flatten:
            patch_dims = self.patch_size * self.patch_size * channels
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        else:
            patches = tf.reshape(patches, [batch_size, -1, self.patch_size, self.patch_size, channels])
        return patches

