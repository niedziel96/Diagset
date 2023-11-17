import tensorflow as tf


class Squeeze(tf.layers.Layer):
    def __init__(self, axis=(1, 2), name=None):
        super().__init__(name=name)

        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.squeeze(inputs, self.axis)
