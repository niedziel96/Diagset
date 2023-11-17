import numpy as np
import tensorflow as tf


def convolution(inputs, k, channels, stride=1, padding='SAME', weight_decay=True,
                activation=tf.nn.relu, layer_name=None):
    input_channels = inputs.get_shape()[3]

    weights_shape = [k, k, input_channels, channels]

    weights = get_variable('%s_weights' % layer_name, weights_shape, weight_decay)
    biases = get_variable('%s_biases' % layer_name, [channels])

    outputs = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1], padding=padding)

    if activation is not None:
        outputs = tf.nn.relu(tf.nn.bias_add(outputs, biases))

    return outputs


def pooling(inputs, k, stride, pooling_type='MAX', padding='SAME', layer_name=None):
    outputs = tf.nn.pool(inputs, pooling_type=pooling_type, window_shape=[k, k],
                         strides=[stride, stride], padding=padding, name=layer_name)

    return outputs


def fully_connected(inputs, nodes, activation=None, weight_decay=True, layer_name=None):
    input_nodes = inputs.get_shape()[1]

    weights = get_variable('%s_weights' % layer_name, [input_nodes, nodes], weight_decay)
    biases = get_variable('%s_biases' % layer_name, [nodes])

    outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)

    if activation is not None:
        outputs = activation(outputs)

    return outputs


def flattening(inputs, layer_name=None):
    flattened_shape = int(np.prod(inputs.get_shape()[1:]))

    outputs = tf.reshape(inputs, [-1, flattened_shape], name=layer_name)

    return outputs


def dropout(inputs, keep_probability, is_training=False, layer_name=None):
    outputs = tf.layers.dropout(inputs, keep_probability, training=is_training, name=layer_name)

    return outputs


def local_response_normalization(inputs, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, layer_name=None):
    outputs = tf.nn.local_response_normalization(inputs, depth_radius=depth_radius, alpha=alpha,
                                                 beta=beta, bias=bias, name=layer_name)

    return outputs


def batch_normalization(inputs, momentum=0.997, epsilon=1e-5, is_training=False, layer_name=None):
    outputs = tf.layers.batch_normalization(inputs, axis=3, momentum=momentum, epsilon=epsilon,
                                            training=is_training, name=layer_name)

    return outputs


def relu(inputs, layer_name=None):
    return tf.nn.relu(inputs, name=layer_name)


def residual_block(inputs, inside_channels, output_channels, identity=True, k=3, stride=2, is_training=False,
                   layer_name=None):
    outputs = convolution(inputs, 1, inside_channels, stride=stride, activation=None, layer_name=('%s_0' % layer_name))
    outputs = batch_normalization(outputs, is_training=is_training, layer_name=('%s_0' % layer_name))
    outputs = tf.nn.relu(outputs)
    outputs = convolution(outputs, k, inside_channels, stride=1, activation=None, layer_name=('%s_1' % layer_name))
    outputs = batch_normalization(outputs, is_training=is_training, layer_name=('%s_1' % layer_name))
    outputs = tf.nn.relu(outputs)
    outputs = convolution(outputs, 1, output_channels, stride=1, activation=None, layer_name=('%s_2' % layer_name))
    outputs = batch_normalization(outputs, is_training=is_training, layer_name=('%s_2' % layer_name))

    if not identity:
        inputs = convolution(inputs, 1, output_channels, stride=stride,
                             activation=None, layer_name=('%s_3' % layer_name))
        inputs = batch_normalization(inputs, is_training=is_training, layer_name=('%s_3' % layer_name))

    outputs = outputs + inputs
    outputs = tf.nn.relu(outputs)

    return outputs


def get_variable(variable_name, shape, weight_decay=False, initializer=None):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()

    variable = tf.get_variable(variable_name, shape, initializer=initializer)

    if weight_decay:
        tf.add_to_collection('decayable_weights', variable)

    return variable
