import json
import logging
import numpy as np
import re
import tensorflow as tf
import textwrap

from diaglib import config
from diaglib.learn import metrics
from itertools import product
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


_CONFUSION_MATRIX_FIGURE_SIZE = (7, 7)
_CONFUSION_MATRIX_DPI = 128
_CONFUSION_MATRIX_IMAGE_SIZE = (
    _CONFUSION_MATRIX_FIGURE_SIZE[1] * _CONFUSION_MATRIX_DPI,
    _CONFUSION_MATRIX_FIGURE_SIZE[0] * _CONFUSION_MATRIX_DPI,
    3
)


class Trainer:
    def __init__(self, network, train_set, epochs, learning_rate, weight_decay, validation_set=None, test_set=None,
                 evaluation_train_set=None, decay_step=None, decay_on_validation_error=False, decay_rate=0.1,
                 minimum_learning_rate=None, optimizer='adam', momentum=0.9, early_stopping=False,
                 compute_train_accuracy=False, pretrained_model=None, plot_confusion_matrix=False, params=None):
        assert optimizer in ['adam', 'sgd']
        assert decay_step is None or not decay_on_validation_error

        self.network = network
        self.train_set = train_set
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.validation_set = validation_set
        self.test_set = test_set
        self.evaluation_train_set = evaluation_train_set
        self.decay_step = decay_step
        self.decay_on_validation_error = decay_on_validation_error
        self.decay_rate = decay_rate
        self.minimum_learning_rate = minimum_learning_rate
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.compute_train_accuracy = compute_train_accuracy
        self.pretrained_model = pretrained_model
        self.plot_confusion_matrix = plot_confusion_matrix
        self.params = params

        self.global_step = tf.get_variable('%s_global_step' % network.name, [],
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.latest_score = tf.get_variable('%s_latest_score' % network.name, [],
                                            initializer=tf.constant_initializer(-np.inf), trainable=False)
        self.learning_rate = tf.get_variable('%s_learning_rate' % network.name, [],
                                             initializer=tf.constant_initializer(learning_rate), trainable=False)
        self.ground_truth = tf.placeholder(tf.int64, shape=[None])
        self.base_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth,
                                                                                       logits=network.outputs))
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

        if compute_train_accuracy:
            self.train_accuracy = tf.placeholder(tf.float32)
            self.train_average_accuracy = tf.placeholder(tf.float32)
            self.train_class_balance_accuracy = tf.placeholder(tf.float32)
            self.train_geometric_average_of_recall = tf.placeholder(tf.float32)

            tf.summary.scalar('train/accuracy', self.train_accuracy)
            tf.summary.scalar('train/average_accuracy', self.train_average_accuracy)
            tf.summary.scalar('train/class_balance_accuracy', self.train_class_balance_accuracy)
            tf.summary.scalar('train/geometric_average_of_recall', self.train_geometric_average_of_recall)

            if self.plot_confusion_matrix:
                self.train_confusion_matrix_image = tf.placeholder(tf.uint8)

                tf.summary.image('train/confusion_matrix', self.train_confusion_matrix_image)

        if validation_set is not None:
            self.validation_accuracy = tf.placeholder(tf.float32)
            self.validation_average_accuracy = tf.placeholder(tf.float32)
            self.validation_class_balance_accuracy = tf.placeholder(tf.float32)
            self.validation_geometric_average_of_recall = tf.placeholder(tf.float32)

            tf.summary.scalar('validation/accuracy', self.validation_accuracy)
            tf.summary.scalar('validation/average_accuracy', self.validation_average_accuracy)
            tf.summary.scalar('validation/class_balance_accuracy', self.validation_class_balance_accuracy)
            tf.summary.scalar('validation/geometric_average_of_recall', self.validation_geometric_average_of_recall)

            if self.plot_confusion_matrix:
                self.validation_confusion_matrix_image = tf.placeholder(tf.uint8)

                tf.summary.image('validation/confusion_matrix', self.validation_confusion_matrix_image)

        self.summary_step = tf.summary.merge_all()

        if test_set is not None:
            self.test_accuracy = tf.placeholder(tf.float32)
            self.test_average_accuracy = tf.placeholder(tf.float32)
            self.test_class_balance_accuracy = tf.placeholder(tf.float32)
            self.test_geometric_average_of_recall = tf.placeholder(tf.float32)

            self.test_summary = tf.summary.merge([
                tf.summary.scalar('test/accuracy', self.test_accuracy),
                tf.summary.scalar('test/average_accuracy', self.test_average_accuracy),
                tf.summary.scalar('test/class_balance_accuracy', self.test_class_balance_accuracy),
                tf.summary.scalar('test/geometric_average_of_recall', self.test_geometric_average_of_recall)
            ])

            if self.plot_confusion_matrix:
                self.test_confusion_matrix_image = tf.placeholder(tf.uint8)

                self.test_summary = tf.summary.merge(
                    [self.test_summary, tf.summary.image('test/confusion_matrix', self.test_confusion_matrix_image)]
                )

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

                if self.compute_train_accuracy:
                    logging.getLogger('diaglib').info('Evaluating model on train dataset...')

                    if self.evaluation_train_set is None:
                        train_set = self.train_set
                    else:
                        train_set = self.evaluation_train_set

                    train_ground_truth, train_predictions = self._get_ground_truth_and_predictions(train_set, session)
                    train_accuracy = metrics.accuracy(train_ground_truth, train_predictions)
                    train_average_accuracy = metrics.average_accuracy(train_ground_truth, train_predictions)
                    train_class_balance_accuracy = metrics.class_balance_accuracy(train_ground_truth, train_predictions)
                    train_geometric_average_of_recall = metrics.geometric_average_of_recall(train_ground_truth, train_predictions)

                    logging.getLogger('diaglib').info('Observed train accuracy = %.4f, average accuracy = %.4f.' %
                                                      (train_accuracy, train_average_accuracy))

                    feed_dict[self.train_accuracy] = train_accuracy
                    feed_dict[self.train_average_accuracy] = train_average_accuracy
                    feed_dict[self.train_class_balance_accuracy] = train_class_balance_accuracy
                    feed_dict[self.train_geometric_average_of_recall] = train_geometric_average_of_recall

                    if self.plot_confusion_matrix:
                        feed_dict[self.train_confusion_matrix_image] = self._get_confusion_matrix_image(
                            train_ground_truth, train_predictions
                        )

                if self.validation_set is not None:
                    logging.getLogger('diaglib').info('Evaluating model on validation dataset...')

                    validation_ground_truth, validation_predictions = self._get_ground_truth_and_predictions(
                        self.validation_set, session
                    )
                    validation_accuracy = metrics.accuracy(validation_ground_truth, validation_predictions)
                    validation_average_accuracy = metrics.average_accuracy(validation_ground_truth, validation_predictions)
                    validation_class_balance_accuracy = metrics.class_balance_accuracy(validation_ground_truth, validation_predictions)
                    validation_geometric_average_of_recall = metrics.geometric_average_of_recall(validation_ground_truth, validation_predictions)

                    logging.getLogger('diaglib').info('Observed validation accuracy = %.4f, average accuracy = %.4f.' %
                                                      (validation_accuracy, validation_average_accuracy))

                    feed_dict[self.validation_accuracy] = validation_accuracy
                    feed_dict[self.validation_average_accuracy] = validation_average_accuracy
                    feed_dict[self.validation_class_balance_accuracy] = validation_class_balance_accuracy
                    feed_dict[self.validation_geometric_average_of_recall] = validation_geometric_average_of_recall

                    if self.plot_confusion_matrix:
                        feed_dict[self.validation_confusion_matrix_image] = self._get_confusion_matrix_image(
                            validation_ground_truth, validation_predictions
                        )

                    current_score = validation_accuracy

                    if session.run(self.latest_score) >= current_score:
                        decayable = self.minimum_learning_rate is None or \
                                    session.run(self.learning_rate) > self.minimum_learning_rate

                        if self.decay_on_validation_error and decayable:
                            logging.getLogger('diaglib').info('Decaying learning rate...')

                            session.run(self.learning_rate.assign(self.learning_rate * self.decay_rate))

                        if self.early_stopping:
                            if self.minimum_learning_rate is None:
                                logging.getLogger('diaglib').info('Stopping training early due to achieving '
                                                                  'worse validation accuracy.')

                                break
                            elif session.run(self.learning_rate) < self.minimum_learning_rate:
                                logging.getLogger('diaglib').info('Stopping training early due to surpassing '
                                                                  'minimum learning rate.')

                                break

                    session.run(self.latest_score.assign(current_score))

                summary = session.run(self.summary_step, feed_dict=feed_dict)

                self.summary_writer.add_summary(summary, epoch + 1)
                self.saver.save(session, str(self.checkpoint_path))

            if self.test_set is not None:
                logging.getLogger('diaglib').info('Evaluating model on test dataset...')

                test_ground_truth, test_predictions = self._get_ground_truth_and_predictions(self.test_set, session)
                test_accuracy = metrics.accuracy(test_ground_truth, test_predictions)
                test_average_accuracy = metrics.average_accuracy(test_ground_truth, test_predictions)
                test_class_balance_accuracy = metrics.class_balance_accuracy(test_ground_truth, test_predictions)
                test_geometric_average_of_recall = metrics.geometric_average_of_recall(test_ground_truth, test_predictions)

                logging.getLogger('diaglib').info('Observed test accuracy = %.4f, average accuracy = %.4f.' %
                                                  (test_accuracy, test_average_accuracy))

                feed_dict = {
                    self.test_accuracy: test_accuracy,
                    self.test_average_accuracy: test_average_accuracy,
                    self.test_class_balance_accuracy: test_class_balance_accuracy,
                    self.test_geometric_average_of_recall: test_geometric_average_of_recall
                }

                if self.plot_confusion_matrix:
                    feed_dict[self.test_confusion_matrix_image] = self._get_confusion_matrix_image(
                        test_ground_truth, test_predictions
                    )

                summary = session.run(self.test_summary, feed_dict=feed_dict)

                self.summary_writer.add_summary(summary, self.epochs)

    def _get_ground_truth_and_predictions(self, dataset, session):
        ground_truth = np.empty(dataset.length, np.int64)
        predictions = np.empty(dataset.length, np.int64)

        current_index = 0

        batches_per_epoch = int(np.ceil(dataset.length / dataset.batch_size))

        for _ in tqdm(range(batches_per_epoch)):
            batch_inputs, batch_ground_truth = dataset.batch()
            batch_predictions = np.argmax(
                self.network.outputs.eval(feed_dict={self.network.inputs: batch_inputs}, session=session),
                axis=-1
            )

            for gt, p in zip(batch_ground_truth, batch_predictions):
                ground_truth[current_index] = gt
                predictions[current_index] = p

                current_index += 1

        return ground_truth, predictions

    def _get_confusion_matrix_image(self, ground_truth, predictions):
        labels = []

        for numeric_label in sorted(set(self.train_set.label_dictionary.values())):
            text_labels_for_numeric_label = []

            for text_label in self.train_set.label_dictionary.keys():
                if self.train_set.label_dictionary[text_label] == numeric_label:
                    text_labels_for_numeric_label.append(text_label)

            labels.append('+'.join(text_labels_for_numeric_label))

        cm = confusion_matrix(ground_truth, predictions, labels=sorted(set(self.train_set.label_dictionary.values())))
        cm = cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)

        fig = Figure(figsize=_CONFUSION_MATRIX_FIGURE_SIZE, dpi=_CONFUSION_MATRIX_DPI, facecolor='w', edgecolor='k')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cm, cmap='Oranges')

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
        classes = ['\n'.join(textwrap.wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predictions', fontsize=12)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('Ground truth', fontsize=12)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=9, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], '.2f') if cm[i, j] != 0 else '.', horizontalalignment='center', fontsize=9,
                    verticalalignment='center', color='black')

        fig.set_tight_layout(True)

        canvas.draw()

        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((1, ) + _CONFUSION_MATRIX_IMAGE_SIZE)

        return image

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
