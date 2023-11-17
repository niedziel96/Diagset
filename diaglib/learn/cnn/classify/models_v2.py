import inspect
import re
import tensorflow as tf

from abc import ABC, abstractmethod
from diaglib import config
from diaglib.learn.cnn.classify import layers_v2


class AbstractModel(ABC):
    def __init__(self, name, output_shape, input_shape=None, inputs=None):
        assert inputs is not None or input_shape is not None

        self.name = name
        self.output_shape = output_shape

        if inputs is not None:
            self.inputs = inputs
        else:
            self.inputs = tf.placeholder(tf.float32, shape=[None] + list(input_shape))

        self.outputs = self.inputs
        self.layers = []
        self.end_points = []
        self.is_training = tf.placeholder_with_default(False, [])

        with tf.variable_scope(self.name):
            self.setup()

    def add(self, layer, inputs=None):
        if inputs is None:
            inputs = self.outputs

        self.layers.append(layer)

        if 'training' in inspect.getfullargspec(layer.call).args:
            self.outputs = layer(inputs, training=self.is_training)
        else:
            self.outputs = layer(inputs)

        return self.outputs

    def restore(self, session, model_name=None, mapping=None, head=True, exclude_training=True):
        if model_name is None:
            model_name = self.name

        model_path = config.MODELS_PATH / model_name / ('%s.ckpt' % model_name)

        if mapping is None:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s/' % self.name)

            if not head:
                variables = [var for var in variables if not var.op.name.startswith('%s/head' % self.name)]

            if exclude_training:
                variables = [
                    var for var in variables if not (
                            'Adam' in var.op.name or
                            'Momentum' in var.op.name or
                            'ExponentialMovingAverage' in var.op.name
                    )
                ]

            mapping = {re.sub('^%s' % self.name, model_name, var.op.name): var for var in variables}

        tf.train.Saver(mapping).restore(session, str(model_path))

    def get_weights(self):
        return [layer.kernel for layer in self.layers if hasattr(layer, 'kernel')]

    @abstractmethod
    def setup(self):
        pass


class VGG16v2(AbstractModel):
    # Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition."
    # arXiv preprint arXiv:1409.1556 (2014).

    def setup(self):
        self.add(tf.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu, name='conv_1a'))
        self.add(tf.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu, name='conv_1b'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_1'))

        self.add(tf.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu, name='conv_2a'))
        self.add(tf.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu, name='conv_2b'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_2'))

        self.add(tf.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, name='conv_3a'))
        self.add(tf.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, name='conv_3b'))
        self.add(tf.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, name='conv_3c'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_3'))

        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_4a'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_4b'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_4c'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_4'))

        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_5a'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_5b'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_5c'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_5'))

        self.add(tf.layers.Conv2D(4096, 7, padding='valid', activation=tf.nn.relu, name='fc_6'))
        self.add(tf.layers.Dropout(0.5, name='drop_6'))

        self.add(tf.layers.Conv2D(4096, 1, padding='same', activation=tf.nn.relu, name='fc_7'))
        self.add(tf.layers.Dropout(0.5, name='drop_7'))

        with tf.variable_scope('head'):
            self.add(tf.layers.Conv2D(self.output_shape[0], 1, name='fc_8'))
            self.add(layers_v2.Squeeze(name='fc_8sqz'))


class VGG19v2(AbstractModel):
    # Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition."
    # arXiv preprint arXiv:1409.1556 (2014).

    def setup(self):
        self.add(tf.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu, name='conv_1a'))
        self.add(tf.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu, name='conv_1b'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_1'))

        self.add(tf.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu, name='conv_2a'))
        self.add(tf.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu, name='conv_2b'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_2'))

        self.add(tf.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, name='conv_3a'))
        self.add(tf.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, name='conv_3b'))
        self.add(tf.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, name='conv_3c'))
        self.add(tf.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, name='conv_3d'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_3'))

        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_4a'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_4b'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_4c'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_4d'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_4'))

        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_5a'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_5b'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_5c'))
        self.add(tf.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, name='conv_5d'))
        self.add(tf.layers.MaxPooling2D(2, 2, name='pool_5'))

        self.add(tf.layers.Conv2D(4096, 7, padding='valid', activation=tf.nn.relu, name='fc_6'))
        self.add(tf.layers.Dropout(0.5, name='drop_6'))

        self.add(tf.layers.Conv2D(4096, 1, padding='same', activation=tf.nn.relu, name='fc_7'))
        self.add(tf.layers.Dropout(0.5, name='drop_7'))

        with tf.variable_scope('head'):
            self.add(tf.layers.Conv2D(self.output_shape[0], 1, name='fc_8'))
            self.add(layers_v2.Squeeze(name='fc_8sqz'))


class ResNet50v2(AbstractModel):
    # He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference
    # on computer vision and pattern recognition. 2016.

    def _add_bottleneck(self, output_depth, inside_depth, stride, epsilon=1e-5, name=None):
        with tf.variable_scope(name):
            inputs = self.outputs
            input_depth = inputs.get_shape()[-1]

            preactivation = self.add(tf.layers.BatchNormalization(epsilon=epsilon, name='preactivation_batch_norm'))

            self.outputs = tf.nn.relu(preactivation)

            self.add(tf.layers.Conv2D(inside_depth, 1, strides=1, padding='same', use_bias=False, name='conv_1'))
            self.add(tf.layers.BatchNormalization(epsilon=epsilon, name='batch_norm_1'))
            self.outputs = tf.nn.relu(self.outputs)

            self.add(tf.layers.Conv2D(inside_depth, 3, strides=stride, padding='same', use_bias=False, name='conv_2'))
            self.add(tf.layers.BatchNormalization(epsilon=epsilon, name='batch_norm_2'))
            self.outputs = tf.nn.relu(self.outputs)

            residual = self.add(tf.layers.Conv2D(output_depth, 1, strides=1, padding='same', name='conv_3'))

            if input_depth == output_depth:
                if stride == 1:
                    shortcut = inputs
                else:
                    shortcut = self.add(tf.layers.MaxPooling2D(1, strides=stride, padding='valid', name='shortcut'), inputs=inputs)
            else:
                shortcut = self.add(
                    tf.layers.Conv2D(output_depth, 1, strides=stride, padding='same', name='shortcut'), inputs=preactivation
                )

            self.outputs = shortcut + residual

    def _add_block(self, depth, n_units, stride, name=None):
        with tf.variable_scope(name):
            for i in range(n_units):
                self._add_bottleneck(depth * 4, depth, stride if i == n_units - 1 else 1, name='unit_%d' % (i + 1))

    def setup(self):
        self.add(tf.layers.Conv2D(64, 7, strides=2, padding='same', name='input_conv'))
        self.add(tf.layers.MaxPooling2D(3, strides=2, padding='valid', name='input_pool'))

        self._add_block(64, 3, 2, 'block_1')
        self._add_block(128, 4, 2, 'block_2')
        self._add_block(256, 6, 2, 'block_3')
        self._add_block(512, 3, 1, 'block_4')

        self.add(tf.layers.BatchNormalization(epsilon=1e-5, name='output_batch_norm'))
        self.outputs = tf.nn.relu(self.outputs)
        self.outputs = tf.reduce_mean(self.outputs, [1, 2], name='output_pool', keepdims=True)

        with tf.variable_scope('head'):
            self.add(tf.layers.Conv2D(self.output_shape[0], 1, padding='same', name='logits'))
            self.add(layers_v2.Squeeze(name='logits_squeezed'))
