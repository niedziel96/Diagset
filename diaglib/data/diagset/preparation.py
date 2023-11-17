import imageio
import json
import logging
import numpy as np

from diaglib import config
from diaglib.data.diagset.containers import Scan
from diaglib.data.diagset.paths import get_nested_path
from itertools import product
from tqdm import tqdm


def process_scan(tissue_tag, scan_name=None, scan_id=None, level=0, image_size=(256, 256), stride=(128, 128),
                 labeling_type='annotations', minimum_overlap=0.75, blob_size=128, debug=True, debug_scaling=8,
                 source='server'):
    """
    Extract images and associated class labels from a scan with the given name. Extracted data is stored under
    config.DATA_PATH in a form of binary data blobs (each containing multiple images), in the NPY format [1].

    [1] https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html

    :param tissue_tag: type of tissue in the specified scan, value from config.TISSUE_TAGS
    :param scan_name: name of the scan, either name or ID must be provided
    :param scan_id: ID of the scan, either name or ID must be provided
    :param level: scan level at which the images are extracted. The magnification of the scan can be calculated as
           BASE_MAGNIFICATION / 2 ** level, with BASE_MAGNIFICATION being the scanner parameter (40x by default).
           That is, at level = 0 we use magnification = 40x, at level = 1 - 20x, and so on
    :param image_size: tuple containing width and height of the extracted images
    :param stride: tuple containing horizontal and vertical stride of the extracted images
    :param labeling_type: either 'annotations' (default), in which case the image classes are assigned based on the
           available annotations, or 'segmentation', in which case VDSR network is used to segment valid tissue and all
           of the underlying images are classified as 'normal' (to be used when it is certain that the given scan
           consists only of healthy tissue)
    :param minimum_overlap: minimum ratio of the number pixels inside annotation to the number of pixels inside whole
           image necessary to classify a given image as belonging to a given annotation
    :param blob_size: the number of images in a single data blob
    :param debug: if True, in addition to data blobs, image file with a thumbnail and corresponding binary annotation
           masks (calculated with no stride) will be stored in config.DEBUG_PATH
    :param debug_scaling: a scaling factor by which the size of the debug images will be increased
    :param source: either 'server' (default), in which case the data will be automatically fetched from NDP API and
           database, or local, in which case the .NDPA scan must be stored in config.SCANS_PATH, and the .NDPI
           annotations in config.ANNOTATIONS_PATH
    """
    assert tissue_tag in config.TISSUE_TAGS
    assert (scan_name is None) != (scan_id is None)
    assert labeling_type in ['annotations', 'segmentation']
    assert 0.0 <= minimum_overlap <= 1.0
    assert source in ['local', 'server']

    if type(image_size) is int:
        image_size = (image_size, image_size)

    if type(stride) is int:
        stride = (stride, stride)

    scan = Scan(tissue_tag, level, scan_id, scan_name, source)

    output_dirs = {}
    root_paths = {
        'blobs': config.DIAGSET_BLOBS_PATH,
        'positions': config.DIAGSET_POSITIONS_PATH,
        'distributions': config.DIAGSET_DISTRIBUTIONS_PATH,
        'debug': config.DIAGSET_DEBUG_PATH,
        'metadata': config.DIAGSET_METADATA_PATH
    }

    for name, root_path in root_paths.items():
        output_dirs[name] = get_nested_path(
            root_path, tissue_tag, scan.magnification
        )

        if name in ['blobs', 'positions', 'debug']:
            output_dirs[name] /= scan.scan_title

    if output_dirs['blobs'].exists() and len([x for x in output_dirs['blobs'].iterdir()]) > 0:
        logging.getLogger('diaglib').info('Blob directory "%s" is already occupied. Omitting scan "%s".'
                                          % (output_dirs['blobs'], scan.scan_title))

        return

    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    logging.getLogger('diaglib').info('Processing scan "%s" at the magnification %dx...' %
                                      (scan.scan_title, scan.magnification))
    logging.getLogger('diaglib').info('Extracting annotation polygons...')

    scan.load_annotation_polygons()

    if labeling_type == 'annotations' and all(polygon.is_empty for polygon in scan.polygons.values()):
        logging.getLogger('diaglib').warning('Found no valid annotation polygons for the given scan.')

        return

    images = {label: [] for label in config.USABLE_LABELS[tissue_tag]}
    positions = {label: [] for label in config.USABLE_LABELS[tissue_tag]}

    if debug:
        logging.getLogger('diaglib').info('Extracting debug images...')

        thumbnail = scan.get_thumbnail(image_size, debug_scaling)

        if labeling_type == 'annotations':
            ground_truth_maps = scan.get_ground_truth_maps(image_size, minimum_overlap, debug_scaling)
        elif labeling_type == 'segmentation':
            ground_truth_maps = scan.get_foreground_as_normal_map(image_size, scaling=debug_scaling)
        else:
            raise NotImplementedError

        imageio.imwrite(str(output_dirs['debug'] / 'thumbnail.jpg'), thumbnail)

        for label, ground_truth_map in ground_truth_maps.items():
            imageio.imwrite(str(output_dirs['debug'] / ('%s.jpg' % label)), (ground_truth_map * 255).astype(np.uint8))

    logging.getLogger('diaglib').info('Extracting images...')

    if labeling_type == 'annotations':
        x_step = int(stride[0] * scan.downsample)
        y_step = int(stride[1] * scan.downsample)

        x_start, x_end, y_start, y_end = scan.get_polygon_bounds(x_step, y_step)

        x_range = range(x_start, x_end, x_step)
        y_range = range(y_start, y_end, y_step)

        for (x, y) in tqdm(product(x_range, y_range), total=(len(x_range) * len(y_range))):
            label = scan.get_patch_ground_truth_label(x, y, image_size, minimum_overlap)

            if label is not None:
                images[label].append(scan.get_patch(x, y, image_size))
                positions[label].append([x, y])
    elif labeling_type == 'segmentation':
        ground_truth_maps = scan.get_foreground_as_normal_map(image_size)

        x_step = int(image_size[0] * scan.downsample)
        y_step = int(image_size[1] * scan.downsample)

        x_start, x_end, y_start, y_end = 0, scan.width, 0, scan.height

        x_range = range(x_start, x_end, x_step)
        y_range = range(y_start, y_end, y_step)

        for (x, y) in tqdm(product(x_range, y_range), total=(len(x_range) * len(y_range))):
            if ground_truth_maps['N'][y // image_size[1], x // image_size[0]] == 1.0:
                images['N'].append(scan.get_patch(x, y, image_size))
                positions['N'].append([x, y])
    else:
        raise NotImplementedError

    n_extracted_images = sum([len(images[label]) for label in config.USABLE_LABELS[tissue_tag]])

    logging.getLogger('diaglib').info('Extracted %d images.' % n_extracted_images)

    for label in config.USABLE_LABELS[tissue_tag]:
        class_blobs_path = output_dirs['blobs'] / label
        class_positions_path = output_dirs['positions'] / label

        for path in [class_blobs_path, class_positions_path]:
            path.mkdir(parents=True, exist_ok=True)

    logging.getLogger('diaglib').info('Saving extracted images...')

    for label in config.USABLE_LABELS[tissue_tag]:
        class_blobs_path = output_dirs['blobs'] / label
        class_positions_path = output_dirs['positions'] / label

        logging.getLogger('diaglib').info('Found %d images from class "%s".' % (len(images[label]), label))

        if len(images[label]) == 0:
            continue

        logging.getLogger('diaglib').info('Saving blobs for class "%s"...' % label)

        n_blobs = int(np.ceil(len(images[label]) / blob_size))
        indices = list(range(len(images[label])))

        np.random.shuffle(indices)

        for i in tqdm(range(n_blobs)):
            blob_indices = indices[(i * blob_size):((i + 1) * blob_size)]
            blob_images = np.array([images[label][j] for j in blob_indices])
            blob_positions = np.array([positions[label][j] for j in blob_indices])

            np.save(str(class_blobs_path / ('%s.blob.%d.npy' % (scan.scan_title, i))), blob_images)
            np.save(str(class_positions_path / ('%s.blob.%d.npy' % (scan.scan_title, i))), blob_positions)

    logging.getLogger('diaglib').info('Saving class distribution...')

    distribution = {label: len(images[label]) for label in config.USABLE_LABELS[tissue_tag]}

    with open(output_dirs['distributions'] / ('%s.json' % scan.scan_title), 'w') as f:
        json.dump(distribution, f)

    logging.getLogger('diaglib').info('Saving metadata...')

    metadata = {
        'scan_name': scan_name,
        'scan_id': scan_id,
        'level': level,
        'image_size': image_size,
        'stride': stride,
        'minimum_overlap': minimum_overlap,
        'blob_size': blob_size,
        'scan_width': scan.width,
        'scan_height': scan.height
    }

    with open(output_dirs['metadata'] / ('%s.json' % scan.scan_title), 'w') as f:
        json.dump(metadata, f)
