import tensorflow as tf

"""
Dealing with dictionary tensor.
Copied from: https://github.com/tensorflow/tensorflow/issues/27679#issuecomment-522578000
"""
def new_py_function(func, inp, Tout, name=None):
    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(inp, flat_inp,
                expand_composites=True)
        out = func(*reconstructed_inp)
        return tf.nest.flatten(out, expand_composites=True)

    flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
    flat_out = tf.py_function(
            func=wrapped_func, 
            inp=tf.nest.flatten(inp, expand_composites=True),
            Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
            name=name)
    spec_out = tf.nest.map_structure(_dtype_to_tensor_spec, Tout, 
            expand_composites=True)
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out

def _dtype_to_tensor_spec(v):
    return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v

def _tensor_spec_to_dtype(v):
    return v.dtype if isinstance(v, tf.TensorSpec) else v


