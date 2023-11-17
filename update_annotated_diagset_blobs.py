import argparse
import logging
import pandas as pd

from copy import deepcopy
from diaglib import config
from diaglib.data.diagset.loading import db
from diaglib.data.diagset.loading.common import extract_labels
from diaglib.data.diagset.preparation import process_scan


PARTITIONS = ['train', 'validation', 'test']


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


def get_assigned_scan_ids():
    """Return dictionary containing only scan IDs already present in one of the CSV partition files."""
    assigned_scan_id_dict = {tag: {} for tag in config.TISSUE_TAGS}

    for tissue_tag in config.TISSUE_TAGS:
        for partition in PARTITIONS:
            csv_path = config.DIAGSET_PARTITIONS_PATH / tissue_tag / ('%s.csv' % partition)
            assigned_scan_id_dict[tissue_tag][partition] = get_scan_ids_from_csv(csv_path)

    return assigned_scan_id_dict


def get_tissue_tag_from_scan_info_file(scan_id):
    scan_info = pd.read_excel(config.DIAGSET_SCAN_INFO_FILE_PATH)
    scan_info['ID'] = scan_info['ID'].str.strip()

    selection = scan_info[scan_info['ID'] == scan_id]

    assert len(selection) <= 1

    if len(selection) == 0:
        return None
    else:
        organ = selection.iloc[0]['narzad'].strip()

        return config.ORGAN_TO_TAG_MAPPING[organ]


def get_tissue_specific_labels():
    tissue_specific_labels = deepcopy(config.EXTRACTED_LABELS)

    for tissue_tag in config.TISSUE_TAGS:
        for label in config.EXTRACTED_LABELS[tissue_tag]:
            for other_tissue_tag in config.TISSUE_TAGS:
                if tissue_tag == other_tissue_tag:
                    continue

                if label in config.EXTRACTED_LABELS[other_tissue_tag]:
                    while label in tissue_specific_labels[tissue_tag]:
                        tissue_specific_labels[tissue_tag].remove(label)

    return tissue_specific_labels


def estimate_tissue_tag(scan_id):
    possible_tissue_tags = []
    annotations = db.fetch_xml_annotations_for_scan(scan_id)
    labels = extract_labels(annotations)
    tissue_specific_labels = get_tissue_specific_labels()

    for label in labels:
        for tissue_tag in config.TISSUE_TAGS:
            if label in tissue_specific_labels[tissue_tag]:
                possible_tissue_tags.append(tissue_tag)

    possible_tissue_tags = set(possible_tissue_tags)

    if len(possible_tissue_tags) == 1:
        return list(possible_tissue_tags)[0]
    else:
        return None


def calculate_current_proportion(assigned_scan_id_dict, tissue_tag):
    lengths = [len(assigned_scan_id_dict[tissue_tag][partition]) for partition in PARTITIONS]

    if sum(lengths) == 0:
        return [0.0, 0.0, 0.0]
    else:
        return [l / sum(lengths) for l in lengths]


parser = argparse.ArgumentParser()

parser.add_argument('-blob_size', type=int, default=128)
parser.add_argument('-image_size', type=int, default=256)
parser.add_argument('-levels', type=int, nargs='+', default=[0, 1, 2, 3])
parser.add_argument('-maximum_validation_test', type=int, default=40)
parser.add_argument('-minimum_overlap', type=float, default=0.75)
parser.add_argument('-ratio', type=float, nargs='+', default=[0.5, 0.25, 0.25],
                    help='Ratio of train/validation/test scans.')
parser.add_argument('-stride', type=int, default=128)
parser.add_argument('-tissue_tags', type=str, nargs='+', default=config.TISSUE_TAGS)

args = parser.parse_args()

assert sum(args.ratio) == 1.0

fh = logging.StreamHandler()
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

logger = logging.getLogger('diaglib')
logger.setLevel(level=logging.DEBUG)
logger.addHandler(fh)

scan_info = db.fetch_scans_with_annotations()

logger.info('Found %d available scans with annotations.' % len(scan_info))

ignored_csv_path = config.DIAGSET_PARTITIONS_PATH / 'ignored.csv'
ignored_scan_ids = get_scan_ids_from_csv(ignored_csv_path)

scan_info = [(scan_id, scan_name) for scan_id, scan_name in scan_info if scan_id not in ignored_scan_ids]

assigned_scan_id_dict = get_assigned_scan_ids()
assigned_scan_count = 0

for tissue_tag in config.TISSUE_TAGS:
    for partition in PARTITIONS:
        assigned_scan_count += len(assigned_scan_id_dict[tissue_tag][partition])

logger.info('Found %d already assigned scans.' % assigned_scan_count)
logger.info('Found %d ignored scans.' % len(ignored_scan_ids))

unassigned_scan_id_list = []

for scan_id, _ in scan_info:
    assignments = []

    for tissue_tag in config.TISSUE_TAGS:
        for partition in PARTITIONS:
            if scan_id in assigned_scan_id_dict[tissue_tag][partition]:
                assignments.append((tissue_tag, partition))

    if len(assignments) == 0:
        unassigned_scan_id_list.append(scan_id)
    elif len(assignments) > 1:
        unassigned_scan_id_list.append(scan_id)

        logger.info('Scan with ID "%s" was automatically assigned to multiple partitions: %s.' % (scan_id, assignments))

if len(unassigned_scan_id_list):
    logger.info('Assigning scans...')

for scan_id in unassigned_scan_id_list:
    logger.info('Assigning scan with ID "%s"...' % scan_id)
    logger.info('Scan can be found at http://%s/ndp/serve/view?objectid=%s.' % (config.NDP_SERVER_ADDRESS, scan_id))
    logger.info('Should this scan be [A]ssigned or [I]gnored?')

    command = None

    while command not in ['A', 'I']:
        command = input('> ')

    if command == 'I':
        add_scan_id_to_csv(ignored_csv_path, scan_id)

        logger.info('Scan with ID "%s" was ignored.' % scan_id)

        continue

    scan_info_tissue_tag = get_tissue_tag_from_scan_info_file(scan_id)
    estimated_tissue_tag = estimate_tissue_tag(scan_id)

    if scan_info_tissue_tag is not None:
        tissue_tag = scan_info_tissue_tag

        logger.info('Using scan info tissue tag "%s" for scan with ID "%s".' % (scan_info_tissue_tag, scan_id))
    elif estimated_tissue_tag is not None:
        tissue_tag = estimated_tissue_tag

        logger.info('Using estimated tissue tag "%s" for scan with ID "%s".' % (estimated_tissue_tag, scan_id))
    else:
        scan_name = db.fetch_scan_name(scan_id)
        annotations = db.fetch_xml_annotations_for_scan(scan_id)
        labels = extract_labels(annotations)

        logger.info('Unable to automatically detect tissue tag for scan with ID "%s".' % scan_id)
        logger.info('Scan name: "%s".' % scan_name)
        logger.info('Found following labels: %s.' % labels)
        logger.info('What tissue tag should be assigned? [%s]' % '/'.join(config.TISSUE_TAGS))

        tissue_tag = None

        while tissue_tag not in config.TISSUE_TAGS:
            tissue_tag = input('> ')

        logger.info('Tissue tag "%s" was assigned.' % tissue_tag)

    current_proportion = calculate_current_proportion(assigned_scan_id_dict, tissue_tag)

    if current_proportion[1] <= args.ratio[1] and \
            len(assigned_scan_id_dict[tissue_tag]['validation']) <= args.maximum_validation_test:
        target_partition = 'validation'
    elif current_proportion[2] <= args.ratio[2] and \
            len(assigned_scan_id_dict[tissue_tag]['test']) <= args.maximum_validation_test:
        target_partition = 'test'
    else:
        target_partition = 'train'

    assigned_scan_id_dict[tissue_tag][target_partition].append(scan_id)

    csv_path = config.DIAGSET_PARTITIONS_PATH / tissue_tag / ('%s.csv' % target_partition)
    add_scan_id_to_csv(csv_path, scan_id)

logger.info('Generating blobs...')

for tissue_tag in args.tissue_tags:
    for partition in PARTITIONS:
        for scan_id in assigned_scan_id_dict[tissue_tag][partition]:
            for level in args.levels:
                process_scan(tissue_tag, scan_id=scan_id, level=level, image_size=args.image_size, stride=args.stride,
                             minimum_overlap=args.minimum_overlap, blob_size=args.blob_size)
