import tensorflow as tf

print('Config Tensorflow to use memory as needed.')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for dev in gpu_devices:
    tf.config.experimental.set_memory_growth(dev, True)

