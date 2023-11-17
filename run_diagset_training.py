import argparse
import logging
import os

from diaglib import config
from diaglib.data.diagset.containers import TrainingDiagSetDataset, EvaluationDiagSetDataset
from diaglib.learn.cnn.classify.model_classes import MODEL_CLASSES
from diaglib.learn.cnn.classify.trainers import Trainer


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-compute_train_accuracy', type=bool, default=False)
parser.add_argument('-decay_on_validation_error', type=bool, default=False)
parser.add_argument('-decay_rate', type=float, default=0.1)
parser.add_argument('-decay_step', type=int, default=20)
parser.add_argument('-early_stopping', type=bool, default=False)
parser.add_argument('-epochs', type=int, default=50)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-learning_rate', type=float, default=0.0001)
parser.add_argument('-magnification', type=int, choices=[5, 10, 20, 40], required=True)
parser.add_argument('-minimum_learning_rate', type=float)
parser.add_argument('-minimum_overlap', type=float, default=0.75)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-name', type=str, required=True)
parser.add_argument('-network', type=str, choices=MODEL_CLASSES.keys(), required=True)
parser.add_argument('-optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
parser.add_argument('-patch_size', type=int, default=224)
parser.add_argument('-pretrained_model', type=str)
parser.add_argument('-scan_subset', type=float, default=1.0)
parser.add_argument('-stride', type=int, default=128)
parser.add_argument('-tissue_tag', type=str, choices=config.TISSUE_TAGS, required=True)
parser.add_argument('-use_unannotated_scans', type=bool, default=False)
parser.add_argument('-weight_decay', type=float, default=0.0005)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

label_dictionary = config.LABEL_DICTIONARIES[args.tissue_tag]

logging.info('Constructing model...')

model_class = MODEL_CLASSES[args.network]
network = model_class(args.name, [len(set(label_dictionary.values()))], [args.patch_size, args.patch_size, 3])

logging.info('Loading training dataset...')

if args.use_unannotated_scans:
    train_partitions = ['train', 'unannotated']
else:
    train_partitions = ['train']

train_set = TrainingDiagSetDataset(
    tissue_tag=args.tissue_tag,
    partitions=train_partitions,
    magnification=args.magnification,
    batch_size=args.batch_size,
    patch_size=(args.patch_size, args.patch_size),
    image_size=(args.image_size, args.image_size),
    stride=(args.stride, args.stride),
    minimum_overlap=args.minimum_overlap,
    label_dictionary=label_dictionary,
    scan_subset=args.scan_subset
)

if args.compute_train_accuracy:
    logging.info('Loading training dataset (evaluation)...')

    evaluation_train_set = EvaluationDiagSetDataset(
        tissue_tag=args.tissue_tag,
        partitions=train_partitions,
        magnification=args.magnification,
        batch_size=args.batch_size,
        patch_size=(args.patch_size, args.patch_size),
        image_size=(args.image_size, args.image_size),
        stride=(args.stride, args.stride),
        minimum_overlap=args.minimum_overlap,
        label_dictionary=label_dictionary,
        scan_subset=train_set.scan_names,
        augment=False,
        shuffling=False
    )
else:
    evaluation_train_set = None

logging.info('Loading validation dataset...')

validation_set = EvaluationDiagSetDataset(
    tissue_tag=args.tissue_tag,
    partitions=['validation'],
    magnification=args.magnification,
    batch_size=args.batch_size,
    patch_size=(args.patch_size, args.patch_size),
    image_size=(args.image_size, args.image_size),
    stride=(args.stride, args.stride),
    minimum_overlap=args.minimum_overlap,
    label_dictionary=label_dictionary,
    augment=False,
    shuffling=False
)

logging.info('Loading test dataset...')

test_set = EvaluationDiagSetDataset(
    tissue_tag=args.tissue_tag,
    partitions=['test'],
    magnification=args.magnification,
    batch_size=args.batch_size,
    patch_size=(args.patch_size, args.patch_size),
    image_size=(args.image_size, args.image_size),
    stride=(args.stride, args.stride),
    minimum_overlap=args.minimum_overlap,
    label_dictionary=label_dictionary,
    augment=False,
    shuffling=False
)

trainer = Trainer(
    network,
    train_set,
    args.epochs,
    args.learning_rate,
    args.weight_decay,
    validation_set,
    test_set,
    evaluation_train_set,
    decay_step=args.decay_step,
    decay_on_validation_error=args.decay_on_validation_error,
    decay_rate=args.decay_rate,
    minimum_learning_rate=args.minimum_learning_rate,
    optimizer=args.optimizer,
    momentum=args.momentum,
    early_stopping=args.early_stopping,
    compute_train_accuracy=args.compute_train_accuracy,
    pretrained_model=args.pretrained_model,
    plot_confusion_matrix=True,
    params=vars(args)
)

trainer.train()
