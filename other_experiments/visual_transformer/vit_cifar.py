# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %cd ../..

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tc_formation.models.vision_transformer import ViT
from tc_formation.models.patches_layer import Patches
import tensorflow_addons as tfa

# # Visual Transformer Test with Cifar

# +
num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# + tags=[]
image_size = 64
patch_size = 6

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

inputs = layers.Input(shape=input_shape)
# Augment data.
augmented = data_augmentation(inputs)
# Create patches.
patches = Patches(patch_size, flatten=True)(augmented)

vit = ViT(
    input_tensor=patches,
    sequence_length=121,
    N=8,
    model_dim=64,
    attention_heads=4,
    classes=num_classes)
vit.summary()

# + tags=[]
optimizer = tfa.optimizers.AdamW(
    learning_rate=0.001, weight_decay=0.0001
)

vit.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

vit.fit(
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=100,
    validation_split=0.1,
    callbacks=[],
)
# -

_, accuracy, top_5_accuracy = vit.evaluate(x_test, y_test)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
