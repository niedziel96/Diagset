import argparse
import logging
import os

from diaglib.data.imagenet.containers import ImageNetDataset
from diaglib.learn.cnn.classify.model_classes import MODEL_CLASSES
from diaglib.learn.cnn.classify.trainers import Trainer

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-compute_train_accuracy', type=bool, default=False)
parser.add_argument('-decay_on_validation_error', type=bool, default=False)
parser.add_argument('-decay_rate', type=float, default=0.1)
parser.add_argument('-decay_step', type=int, default=30)
parser.add_argument('-early_stopping', type=bool, default=False)
parser.add_argument('-epochs', type=int, default=120)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-learning_rate', type=float, default=0.025)
parser.add_argument('-minimum_learning_rate', type=float)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-name', type=str, required=True)
parser.add_argument('-network', type=str, choices=MODEL_CLASSES.keys(), required=True)
parser.add_argument('-optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
parser.add_argument('-weight_decay', type=float, default=0.0004)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

logging.info('Constructing model...')

model_class = MODEL_CLASSES[args.network]
network = model_class(args.name, [1000], [224, 224, 3])

logging.info('Loading training dataset...')

train_set = ImageNetDataset('train', args.batch_size)

if args.compute_train_accuracy:
    logging.info('Loading training dataset (evaluation)...')

    evaluation_train_set = ImageNetDataset('train', args.batch_size, augment=False, shuffling=False)
else:
    evaluation_train_set = None

logging.info('Loading validation dataset...')

validation_set = ImageNetDataset('validation', args.batch_size, augment=False, shuffling=False)

trainer = Trainer(network, train_set, args.epochs, args.learning_rate, args.weight_decay, validation_set,
                  evaluation_train_set=evaluation_train_set, decay_step=args.decay_step,
                  decay_on_validation_error=args.decay_on_validation_error, decay_rate=args.decay_rate,
                  minimum_learning_rate=args.minimum_learning_rate, optimizer=args.optimizer,
                  momentum=args.momentum, early_stopping=args.early_stopping, params=vars(args))
trainer.train()
