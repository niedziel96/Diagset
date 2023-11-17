import argparse
import logging
import pandas as pd

from diaglib import config
from diaglib.data.diagset.preparation import process_scan


def get_scan_ids_from_csv(csv_path):
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(exist_ok=True, parents=True)

    if not csv_path.exists():
        pd.DataFrame({'scan_id': []}).to_csv(csv_path, index=False)

    return list(pd.read_csv(csv_path)['scan_id'])


def add_scan_id_to_csv(csv_path, scan_id):
    if not csv_path.exists():
        scan_ids = []
    else:
        scan_ids = list(pd.read_csv(csv_path)['scan_id'])

    if scan_id not in scan_ids:
        scan_ids.append(scan_id)

    pd.DataFrame({'scan_id': scan_ids}).to_csv(csv_path, index=False)


def get_annotated_scan_ids():
    annotated_scan_ids = []

    for tissue_tag in config.TISSUE_TAGS:
        for partition in ['train', 'validation', 'test']:
            csv_path = config.DIAGSET_PARTITIONS_PATH / tissue_tag / ('%s.csv' % partition)
            annotated_scan_ids += get_scan_ids_from_csv(csv_path)

    annotated_scan_ids += get_scan_ids_from_csv(config.DIAGSET_PARTITIONS_PATH / 'ignored.csv')

    return annotated_scan_ids


def get_unannotated_scan_ids(tissue_tag):
    organs = [organ for organ, tag in config.ORGAN_TO_TAG_MAPPING.items() if tag == tissue_tag]

    df = pd.read_excel(config.DIAGSET_SCAN_INFO_FILE_PATH)

    df = df[df['narzad'].isin(organs)]
    df = df[df['br'] == 1]

    df['ID'] = df['ID'].str.strip()
    df = df[~df['ID'].isin(get_annotated_scan_ids())]

    return list(df['ID'])


parser = argparse.ArgumentParser()

parser.add_argument('-blob_size', type=int, default=128)
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-levels', type=int, nargs='+', default=[0, 1, 2, 3])
parser.add_argument('-stride', type=int, default=128)
parser.add_argument('-tissue_tags', type=str, nargs='+', default=config.TISSUE_TAGS)

args = parser.parse_args()

fh = logging.StreamHandler()
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

logger = logging.getLogger('diaglib')
logger.setLevel(level=logging.DEBUG)
logger.addHandler(fh)

for tissue_tag in args.tissue_tags:
    unannotated_scan_ids = get_unannotated_scan_ids(tissue_tag)

    logger.info('Found %d valid, unannotated scans for tissue tag "%s".' % (len(unannotated_scan_ids), tissue_tag))

    csv_path = config.DIAGSET_PARTITIONS_PATH / tissue_tag / 'unannotated.csv'

    for scan_id in unannotated_scan_ids:
        add_scan_id_to_csv(csv_path, scan_id)

logger.info('Generating blobs...')

for tissue_tag in args.tissue_tags:
    for scan_id in get_unannotated_scan_ids(tissue_tag):
        for level in args.levels:
            process_scan(
                tissue_tag, scan_id=scan_id, level=level, image_size=args.image_size,
                stride=args.stride, labeling_type='segmentation', blob_size=args.blob_size
            )
