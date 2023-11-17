import json
import logging
import numpy as np
import tensorflow as tf

from diaglib import config
from skimage.transform import resize
from tqdm import tqdm


class Trainer:
    def __init__(self, network, train_set, epochs, learning_rate, weight_decay, validation_set=None, decay_step=None,
                 decay_rate=0.1, optimizer='adam', momentum=0.9, pretrained_model=None, image_summary_size=(960, 960),
                 params=None):
        assert optimizer in ['adam', 'sgd']

        self.network = network
        self.train_set = train_set
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.validation_set = validation_set
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.pretrained_model = pretrained_model
        self.image_summary_size = image_summary_size
        self.params = params

        self.global_step = tf.get_variable('%s_global_step' % network.name, [],
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.get_variable('%s_learning_rate' % network.name, [],
                                             initializer=tf.constant_initializer(learning_rate), trainable=False)
        self.ground_truth = tf.placeholder(tf.float32, [None, None, None, 1])
        self.base_loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(self.ground_truth, network.outputs)))
        self.weight_decay_loss = weight_decay * tf.add_n([tf.nn.l2_loss(w) for w in network.get_weights()])
        self.total_loss = self.base_loss + self.weight_decay_loss

        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=momentum)
        else:
            raise NotImplementedError

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(self.update_ops):
            self.train_step = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

        self.saver = tf.train.Saver()

        self.mean_base_loss = tf.placeholder(tf.float32)
        self.mean_weight_decay_loss = tf.placeholder(tf.float32)
        self.mean_total_loss = tf.placeholder(tf.float32)

        tf.summary.scalar('train/base_loss', self.mean_base_loss)
        tf.summary.scalar('train/weight_decay_loss', self.mean_weight_decay_loss)
        tf.summary.scalar('train/total_loss', self.mean_total_loss)
        tf.summary.scalar('train/learning_rate', self.learning_rate)

        if validation_set is not None:
            self.validation_inputs = tf.placeholder(tf.uint8, [None, None, None, 3])
            self.validation_outputs = tf.placeholder(tf.uint8, [None, None, None, 1])

            tf.summary.image('validation/inputs', self.validation_inputs, max_outputs=validation_set.length)
            tf.summary.image('validation/outputs', self.validation_outputs, max_outputs=validation_set.length)

        self.summary_step = tf.summary.merge_all()

        self.model_path = config.MODELS_PATH / network.name
        self.checkpoint_path = self.model_path / ('%s.ckpt' % network.name)
        self.log_path = config.LOGS_PATH / network.name

        self.summary_writer = tf.summary.FileWriter(str(self.log_path))

        for path in [config.MODELS_PATH, config.LOGS_PATH, self.model_path, self.log_path]:
            if not path.exists():
                path.mkdir()

    def train(self):
        self._validate_and_save_params()

        with tf.Session() as session:
            checkpoint = tf.train.get_checkpoint_state(str(self.model_path))
            session.run(tf.global_variables_initializer())

            if checkpoint and checkpoint.model_checkpoint_path:
                logging.getLogger('diaglib').info('Restoring the model...')

                self.saver.restore(session, checkpoint.model_checkpoint_path)

            batches_processed = tf.train.global_step(session, self.global_step)
            batches_per_epoch = int(np.ceil(self.train_set.length / self.train_set.batch_size))
            epochs_processed = int(batches_processed / batches_per_epoch)

            if batches_processed == 0 and self.pretrained_model is not None:
                logging.getLogger('diaglib').info('Using pretrained model "%s"...' % self.pretrained_model)

                self.network.restore(session, self.pretrained_model, head=False)

            for epoch in range(epochs_processed, self.epochs):
                logging.getLogger('diaglib').info('Processing epoch #%d...' % (epoch + 1))

                if self.decay_step is not None and epoch % self.decay_step == 0 and epoch > 0:
                    logging.getLogger('diaglib').info('Decaying learning rate...')

                    session.run(self.learning_rate.assign(self.learning_rate * self.decay_rate))

                base_losses = []
                weight_decay_losses = []
                total_losses = []

                for _ in tqdm(range(0, batches_per_epoch)):
                    inputs, outputs = self.train_set.batch()
                    feed_dict = {
                        self.network.inputs: inputs, self.ground_truth: outputs,
                        self.network.is_training: True
                    }
                    _, base_loss, weight_decay_loss, total_loss = session.run(
                        [self.train_step, self.base_loss, self.weight_decay_loss, self.total_loss],
                        feed_dict=feed_dict
                    )

                    base_losses.append(base_loss)
                    weight_decay_losses.append(weight_decay_loss)
                    total_losses.append(total_loss)

                feed_dict = {
                    self.mean_base_loss: np.mean(base_losses),
                    self.mean_weight_decay_loss: np.mean(weight_decay_losses),
                    self.mean_total_loss: np.mean(total_losses)
                }

                if self.validation_set is not None:
                    logging.getLogger('diaglib').info('Evaluating model on validation dataset...')

                    validation_inputs = []
                    validation_outputs = []

                    for validation_input in self.validation_set.inputs:
                        validation_output = self.network.outputs.eval(
                            feed_dict={self.network.inputs: [validation_input]}, session=session
                        )[0]
                        validation_output = np.clip(validation_output, 0, 255).astype(np.uint8)

                        validation_inputs.append(self._resize_validation_image(validation_input))
                        validation_outputs.append(self._resize_validation_image(validation_output) * 255)

                    validation_inputs = np.array(validation_inputs).astype(np.uint8)
                    validation_outputs = np.array(validation_outputs).astype(np.uint8)

                    feed_dict[self.validation_inputs] = validation_inputs
                    feed_dict[self.validation_outputs] = validation_outputs

                summary = session.run(self.summary_step, feed_dict=feed_dict)

                self.summary_writer.add_summary(summary, epoch + 1)
                self.saver.save(session, str(self.checkpoint_path))

    def _validate_and_save_params(self):
        if self.params is not None:
            assert self.params.get('name') is not None

            if not config.PARAMS_PATH.exists():
                config.PARAMS_PATH.mkdir(parents=True, exist_ok=True)

            path = config.PARAMS_PATH / ('%s.json' % self.params['name'])

            if path.exists():
                with open(path, 'r') as f:
                    saved_params = json.load(f)

                if saved_params != self.params:
                    raise ValueError('Passed parameters differ from the ones already stored for this model.')
            else:
                with open(path, 'w') as f:
                    json.dump(self.params, f)

    def _resize_validation_image(self, image, cval=0):
        ratio = np.min([n / i for n, i in zip(self.image_summary_size, image.shape)])

        interim_shape = np.rint([s * ratio for s in image.shape[:2]]).astype(np.int)
        interim_image = resize(image, interim_shape, mode='constant', cval=cval)

        resized_image = np.empty(list(self.image_summary_size) + [image.shape[2]], dtype=interim_image.dtype)
        resized_image.fill(cval)

        x_start = (self.image_summary_size[0] - interim_shape[0]) // 2
        y_start = (self.image_summary_size[1] - interim_shape[1]) // 2

        resized_image[x_start:(x_start + interim_shape[0]), y_start:(y_start + interim_shape[1])] = interim_image

        return resized_image
