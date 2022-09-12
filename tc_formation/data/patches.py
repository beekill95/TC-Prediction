import tensorflow as tf
import tensorflow.keras.layers as layers


class PatchesLayer(layers.Layer):
    def __init__(self, patch_size, flatten=True, padding='SAME'):
        super(PatchesLayer, self).__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.padding = padding

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding,
        )

        if self.flatten:
            patches = tf.reshape(patches, [batch_size, -1, self.patch_size * self.patch_size])
        else:
            patches = tf.reshape(patches, [batch_size, -1, self.patch_size, self.patch_size])

        return patches
