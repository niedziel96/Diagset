import inspect
import tensorflow as tf

from diaglib import config


class VDSR:
    def __init__(self, name, n_layers=10, k=3, n_filters=64):
        self.name = name
        self.inputs = tf.placeholder(tf.float32, [None, None, None, 3])
        self.n_layers = n_layers
        self.k = k
        self.n_filters = n_filters

        self.outputs = self.inputs
        self.layers = []
        self.is_training = tf.placeholder_with_default(False, [])

        for i in range(n_layers):
            if i == n_layers - 1:
                filters = 1
            else:
                filters = n_filters

            if i < self.n_layers - 1:
                activation = tf.nn.relu
            else:
                activation = None

            layer = tf.layers.Conv2D(filters, k, padding='same', activation=activation)

            self.add(layer)

    def add(self, layer):
        self.layers.append(layer)

        if 'training' in inspect.getfullargspec(layer.call).args:
            self.outputs = layer(self.outputs, training=self.is_training)
        else:
            self.outputs = layer(self.outputs)

        return self.outputs

    def restore(self, session, model_name=None):
        if model_name is None:
            model_name = self.name

        model_path = config.MODELS_PATH / model_name / ('%s.ckpt' % model_name)

        tf.train.Saver().restore(session, str(model_path))

    def get_weights(self):
        return [layer.kernel for layer in self.layers if hasattr(layer, 'kernel')]
