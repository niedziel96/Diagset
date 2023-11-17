import argparse
import logging
import os

from diaglib.data.segset.containers import TrainingSegSetDataset, TestSegSetDataset
from diaglib.learn.cnn.segment.models import VDSR
from diaglib.learn.cnn.segment.trainers import Trainer


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-decay_rate', type=float, default=0.1)
parser.add_argument('-decay_step', type=int, default=200)
parser.add_argument('-downsample', type=float, default=8.0)
parser.add_argument('-epochs', type=int, default=600)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-k', type=int, default=3)
parser.add_argument('-learning_rate', type=float, default=0.0001)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-n_filters', type=int, default=64)
parser.add_argument('-n_layers', type=int, default=10)
parser.add_argument('-name', type=str, required=True)
parser.add_argument('-optimizer', type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument('-patch_size', type=int, default=21)
parser.add_argument('-pretrained_model', type=str)
parser.add_argument('-weight_decay', type=float, default=0.0001)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

logging.info('Constructing model...')

network = VDSR(args.name, args.n_layers, args.k, args.n_filters)

logging.info('Loading training dataset...')

train_set = TrainingSegSetDataset(batch_size=args.batch_size, patch_size=args.patch_size, downsample=args.downsample)

logging.info('Loading test dataset...')

test_set = TestSegSetDataset(downsample=args.downsample)

trainer = Trainer(
    network,
    train_set,
    args.epochs,
    args.learning_rate,
    args.weight_decay,
    test_set,
    decay_step=args.decay_step,
    decay_rate=args.decay_rate,
    optimizer=args.optimizer,
    momentum=args.momentum,
    pretrained_model=args.pretrained_model,
    params=vars(args)
)

trainer.train()
