import argparse
import logging
import os
import pandas as pd

from diaglib import config
from diaglib.learn.cnn.classify.model_classes import MODEL_CLASSES
from diaglib.predict.maps.ensemble import produce_maps


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-buffer_size', type=int, default=128)
parser.add_argument('-ensemble_name', type=str)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-magnifications', type=int, choices=[5, 10, 20, 40], nargs='+', required=True)
parser.add_argument('-model_classes', type=str, choices=MODEL_CLASSES.keys(), nargs='+', required=True)
parser.add_argument('-model_names', type=str, nargs='+', required=True)
parser.add_argument('-partitions', type=str, nargs='+')
parser.add_argument('-patch_size', type=int, default=224)
parser.add_argument('-postprocess_background', type=bool, default=True)
parser.add_argument('-scan_ids', type=str, nargs='+')
parser.add_argument('-segment_foreground', type=bool, default=True)
parser.add_argument('-tissue_tag', type=str, choices=config.TISSUE_TAGS, required=True)

args = parser.parse_args()

assert (args.partitions is not None) != (args.scan_ids is not None)

if args.partitions is not None:
    scan_ids = sum([
        list(pd.read_csv(config.DIAGSET_PARTITIONS_PATH / args.tissue_tag / ('%s.csv' % partition))['scan_id'])
        for partition in args.partitions
    ], [])
else:
    scan_ids = args.scan_ids

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

for i, scan_id in enumerate(scan_ids):
    logging.info('Processing scan %d/%d with ID = %s...' % (i + 1, len(scan_ids), scan_id))

    model_classes = [MODEL_CLASSES[cls] for cls in args.model_classes]

    if args.ensemble_name is None:
        ensemble_name = ','.join(args.model_names)
    else:
        ensemble_name = args.ensemble_name

    produce_maps(
        scan_id=scan_id, tissue_tag=args.tissue_tag,
        magnifications=args.magnifications, patch_size=args.patch_size,
        batch_size=args.batch_size, buffer_size=args.buffer_size,
        ensemble_name=ensemble_name, model_classes=model_classes,
        model_names=args.model_names, segment_foreground=args.segment_foreground,
        postprocess_background=args.postprocess_background
    )
