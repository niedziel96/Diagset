import re
import tensorflow as tf

from abc import ABC, abstractmethod
from diaglib import config
from diaglib.learn.cnn.classify.layers import *


class Network(ABC):
    def __init__(self, name, output_shape, input_shape=None, inputs=None):
        assert inputs is not None or input_shape is not None

        self.name = name
        self.output_shape = output_shape

        if inputs is not None:
            self.inputs = inputs
        else:
            self.inputs = tf.placeholder(tf.float32, shape=[None] + list(input_shape))

        self.outputs = self.inputs
        self.is_training = tf.placeholder_with_default(False, [])
        self.n_layers = 0

        with tf.variable_scope(self.name):
            self.setup()

    def restore(self, session, model_name=None, head=True):
        if model_name is None:
            model_name = self.name

        model_path = config.MODELS_PATH / model_name / ('%s.ckpt' % model_name)

        if not head:
            scope = '%s/finetunable' % self.name
        else:
            scope = self.name

        variables = {re.sub('^%s' % self.name, model_name, var.op.name): var
                     for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)}

        tf.train.Saver(variables).restore(session, str(model_path))

    def add(self, layer, **kwargs):
        layer_name = 'layer_%d' % self.n_layers

        self.n_layers += 1

        self.outputs = layer(inputs=self.outputs, layer_name=layer_name, **kwargs)

        return self

    def get_weights(self):
        return tf.get_collection('decayable_weights')

    @abstractmethod
    def setup(self):
        pass


class AlexNet(Network):
    # Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet classification with deep convolutional
    # neural networks." Advances in neural information processing systems. 2012.

    def setup(self):
        with tf.variable_scope('finetunable'):
            self.add(convolution, k=11, channels=96, stride=4).\
                add(local_response_normalization).\
                add(pooling, k=3, stride=2, padding='VALID').\
                add(convolution, k=5, channels=256).\
                add(local_response_normalization).\
                add(pooling, k=3, stride=2, padding='VALID').\
                add(convolution, k=3, channels=384).\
                add(convolution, k=3, channels=384).\
                add(convolution, k=3, channels=256).\
                add(pooling, k=3, stride=2, padding='VALID')

        self.add(flattening).\
            add(fully_connected, nodes=4096, activation=tf.nn.relu).\
            add(dropout, keep_probability=0.5, is_training=self.is_training).\
            add(fully_connected, nodes=4096, activation=tf.nn.relu).\
            add(dropout, keep_probability=0.5, is_training=self.is_training).\
            add(fully_connected, nodes=self.output_shape[0])


class VGG16(Network):
    # Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition."
    # arXiv preprint arXiv:1409.1556 (2014).

    def setup(self):
        with tf.variable_scope('finetunable'):
            self.add(convolution, k=3, channels=64).\
                add(convolution, k=3, channels=64).\
                add(pooling, k=2, stride=2).\
                add(convolution, k=3, channels=128).\
                add(convolution, k=3, channels=128).\
                add(pooling, k=2, stride=2).\
                add(convolution, k=3, channels=256).\
                add(convolution, k=3, channels=256).\
                add(convolution, k=3, channels=256).\
                add(pooling, k=2, stride=2).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(pooling, k=2, stride=2).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(pooling, k=2, stride=2)

        self.add(flattening).\
            add(fully_connected, nodes=4096, activation=tf.nn.relu).\
            add(dropout, keep_probability=0.5, is_training=self.is_training).\
            add(fully_connected, nodes=4096, activation=tf.nn.relu).\
            add(dropout, keep_probability=0.5, is_training=self.is_training).\
            add(fully_connected, nodes=self.output_shape[0])


class VGG19(Network):
    # Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition."
    # arXiv preprint arXiv:1409.1556 (2014).

    def setup(self):
        with tf.variable_scope('finetunable'):
            self.add(convolution, k=3, channels=64).\
                add(convolution, k=3, channels=64).\
                add(pooling, k=2, stride=2).\
                add(convolution, k=3, channels=128).\
                add(convolution, k=3, channels=128).\
                add(pooling, k=2, stride=2).\
                add(convolution, k=3, channels=256).\
                add(convolution, k=3, channels=256).\
                add(convolution, k=3, channels=256).\
                add(convolution, k=3, channels=256).\
                add(pooling, k=2, stride=2).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(pooling, k=2, stride=2).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(convolution, k=3, channels=512).\
                add(pooling, k=2, stride=2)

        self.add(flattening).\
            add(fully_connected, nodes=4096, activation=tf.nn.relu).\
            add(dropout, keep_probability=0.5, is_training=self.is_training).\
            add(fully_connected, nodes=4096, activation=tf.nn.relu).\
            add(dropout, keep_probability=0.5, is_training=self.is_training).\
            add(fully_connected, nodes=self.output_shape[0])


class ResNet50(Network):
    # He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference
    # on computer vision and pattern recognition. 2016.

    def setup(self):
        with tf.variable_scope('finetunable'):
            self.add(convolution, k=7, channels=64, stride=2, activation=None).\
                add(batch_normalization).\
                add(relu).\
                add(pooling, k=3, stride=2, pooling_type='MAX')

            block_parameters = [
                [3, 64, 256],
                [4, 128, 512],
                [6, 256, 1024],
                [3, 512, 2048]
            ]

            for n_layers, inside_channels, output_channels in block_parameters:
                for i in range(n_layers):
                    if i == 0:
                        stride = 2
                        identity = False
                    else:
                        stride = 1
                        identity = True

                    self.add(residual_block,
                             inside_channels=inside_channels,
                             output_channels=output_channels,
                             stride=stride,
                             identity=identity,
                             is_training=self.is_training)

            self.add(pooling, k=7, stride=1, pooling_type='AVG')

        self.add(flattening).add(fully_connected, nodes=self.output_shape[0])


class ResNet152(Network):
    # He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference
    # on computer vision and pattern recognition. 2016.

    def setup(self):
        with tf.variable_scope('finetunable'):
            self.add(convolution, k=7, channels=64, stride=2, activation=None).\
                add(batch_normalization).\
                add(relu).\
                add(pooling, k=3, stride=2, pooling_type='MAX')

            block_parameters = [
                [3, 64, 256],
                [8, 128, 512],
                [36, 256, 1024],
                [3, 512, 2048]
            ]

            for n_layers, inside_channels, output_channels in block_parameters:
                for i in range(n_layers):
                    if i == 0:
                        stride = 2
                        identity = False
                    else:
                        stride = 1
                        identity = True

                    self.add(residual_block,
                             inside_channels=inside_channels,
                             output_channels=output_channels,
                             stride=stride,
                             identity=identity,
                             is_training=self.is_training)

            self.add(pooling, k=7, stride=1, pooling_type='AVG')

        self.add(flattening).add(fully_connected, nodes=self.output_shape[0])


class BreakHisNet(Network):
    # Spanhol, Fabio Alexandre, et al. "Breast cancer histopathological image classification using convolutional
    # neural networks." Neural Networks (IJCNN), 2016 International Joint Conference on. IEEE, 2016.

    def setup(self):
        with tf.variable_scope('finetunable'):
            self.add(convolution, k=5, channels=32).\
                add(pooling, k=3, stride=2, pooling_type='MAX').\
                add(convolution, k=5, channels=32).\
                add(pooling, k=3, stride=2, pooling_type='AVG').\
                add(convolution, k=5, channels=64).\
                add(pooling, k=3, stride=2, pooling_type='AVG')

        self.add(flattening).\
            add(fully_connected, nodes=64, activation=tf.nn.relu).\
            add(fully_connected, nodes=self.output_shape[0])
