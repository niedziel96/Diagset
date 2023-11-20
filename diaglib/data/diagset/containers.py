import json
import logging
import numpy as np
import pandas as pd
import os 
from abc import ABC, abstractmethod
from diaglib import config
# from diaglib.data.diagset.loading import ndp, db, local
from diaglib.data.diagset.loading.common import prepare_multipolygons
from diaglib.data.diagset.paths import get_nested_path
from diaglib.data.imagenet.containers import IMAGENET_IMAGE_MEAN
from diaglib.predict.maps.common import segment_foreground_vdsr
from queue import Queue
from shapely.geometry.polygon import Polygon
from threading import Thread
from pathlib import Path

class AbstractDiagSetDataset(ABC):
    def __init__(self, tissue_tag, partitions, magnification=40, batch_size=32, patch_size=(224, 224),
                 image_size=(256, 256), stride=(128, 128), minimum_overlap=0.75, augment=True, subtract_mean=True,
                 label_dictionary=None, shuffling=True, class_ratios=None, scan_subset=None, buffer_size=64):
        assert tissue_tag in config.TISSUE_TAGS

        for partition in partitions:
            assert partition in ['train', 'validation', 'test', 'unannotated']

        self.tissue_tag = tissue_tag
        self.partitions = partitions
        self.magnification = magnification
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.stride = stride
        self.minimum_overlap = minimum_overlap
        self.augment = augment
        self.subtract_mean = subtract_mean
        self.shuffling = shuffling
        self.scan_subset = scan_subset
        self.buffer_size = buffer_size

        if label_dictionary is None:
            logging.getLogger('diaglib').info('Using default label dictionary...')

            self.label_dictionary = config.LABEL_DICTIONARIES[tissue_tag]
        else:
            self.label_dictionary = label_dictionary

        self.numeric_labels = list(set(self.label_dictionary.values()))

        self.buffers = {}
        self.blob_paths = {}
        self.class_distribution = {}

        for numeric_label in self.numeric_labels:
            self.buffers[numeric_label] = Queue(buffer_size)
            self.blob_paths[numeric_label] = []
            self.class_distribution[numeric_label] = 0

        self.length = 0

        self.root_blobs_path = get_nested_path(
            config.DIAGSET_BLOBS_PATH, tissue_tag, magnification
        )
        self.root_distributions_path = get_nested_path(
            config.DIAGSET_DISTRIBUTIONS_PATH, tissue_tag, magnification
        )
        self.root_distributions_path = Path(os.path.join(self.root_distributions_path, '256x256', '128x128','0.75'))
        self.root_blobs_path = Path(os.path.join(self.root_blobs_path, '256x256', '128x128','0.75'))
        print(f" ********** {self.root_blobs_path} ********")
        assert self.root_blobs_path.exists()

        self.scan_names = [path.name for path in self.root_blobs_path.iterdir()]
        
        partition_scan_names = []

        for partition in self.partitions:
            partition_path = config.DIAGSET_PARTITIONS_PATH / tissue_tag / ('%s.csv' % partition)

            if partition_path.exists():
                df = pd.read_csv(partition_path)
                partition_scan_names += df['scan_id'].astype(str).tolist()
            else:
                raise ValueError('Partition file not found under "%s".' % partition_path)

        self.scan_names = [scan_name for scan_name in self.scan_names if scan_name in partition_scan_names]

        if self.scan_subset is not None and self.scan_subset != 1.0:
            if type(self.scan_subset) is list:
                logging.getLogger('diaglib').info('Using given %d out of %d scans...' %
                                                  (len(self.scan_subset), len(self.scan_names)))

                self.scan_names = self.scan_subset
            else:
                if type(self.scan_subset) is float:
                    n_scans = int(self.scan_subset * len(self.scan_names))
                else:
                    n_scans = self.scan_subset

                assert n_scans > 0
                assert n_scans <= len(self.scan_names)

                logging.getLogger('diaglib').info('Randomly selecting %d out of %d scans...' %
                                                  (n_scans, len(self.scan_names)))

                self.scan_names = list(np.random.choice(self.scan_names, n_scans, replace=False))

        logging.getLogger('diaglib').info('Loading blob paths...')

        errored_dist = []
        for scan_name in self.scan_names:
            ignored = []
            for string_label in config.USABLE_LABELS[tissue_tag]:
                numeric_label = self.label_dictionary[string_label]
                pth = Path((f'{self.root_blobs_path}/{scan_name}/{string_label}'))

                if pth.exists():
                    blbs_list = [x for x in pth.iterdir()]
                    blob_names = map(lambda x: x.name, sorted(blbs_list))
                else:
                    print(f'Not found for: {string_label}')
                    
                    ignored.append(string_label)
                    continue 
                
                for blob_name in blob_names:
                    self.blob_paths[numeric_label].append(self.root_blobs_path / scan_name / string_label / blob_name)

            with open(self.root_distributions_path / ('%s.json' % scan_name), 'r') as f:
                scan_class_distribution = json.load(f)

            
            self.length += sum(scan_class_distribution.values())

            for string_label in config.USABLE_LABELS[tissue_tag]:
                print(string_label)
                numeric_label = self.label_dictionary[string_label]

                try:
                    self.class_distribution[numeric_label] += scan_class_distribution[string_label]
                except:
                    print('*** no class found ***')
                    continue
                    
        if class_ratios is None:
            self.class_ratios = {}

            for numeric_label in self.numeric_labels:
                print(self.length)
                self.class_ratios[numeric_label] = self.class_distribution[numeric_label] / self.length
        else:
            self.class_ratios = class_ratios

        print(self.class_ratios)

        logging.getLogger('diaglib').info('Found %d patches.' % self.length)

        class_distribution_text = ', '.join(['%s: %.2f%%' % (label, count / self.length * 100)
                                             for label, count in self.class_distribution.items()])
        logging.getLogger('diaglib').info('Class distribution: %s.' % class_distribution_text)

        if self.shuffling:
            for numeric_label in self.numeric_labels:
                np.random.shuffle(self.blob_paths[numeric_label])

        for numeric_label in self.numeric_labels:
            if len(self.blob_paths[numeric_label]) > 0:
                Thread(target=self.fill_buffer, daemon=True, args=(numeric_label, )).start()

    @abstractmethod
    def batch(self):
        return

    def fill_buffer(self, numeric_label):
        while True:
            for blob_path in self.blob_paths[numeric_label]:
                images = self.prepare_images(blob_path)

                for image in images:
                    self.buffers[numeric_label].put(image)

            if self.shuffling:
                np.random.shuffle(self.blob_paths[numeric_label])

    def prepare_images(self, blob_path):
        images = np.load(blob_path)

        if self.shuffling:
            np.random.shuffle(images)

        prepared_images = []

        for i in range(len(images)):
            image = images[i].astype(np.float32)

            if self.augment:
                image = self._augment(image)
            else:
                x = (image.shape[0] - self.patch_size[0]) // 2
                y = (image.shape[1] - self.patch_size[1]) // 2

                image = image[x:(x + self.patch_size[0]), y:(y + self.patch_size[1])]

            if self.subtract_mean:
                image -= IMAGENET_IMAGE_MEAN

            prepared_images.append(image)

        prepared_images = np.array(prepared_images)

        return prepared_images

    def _augment(self, image):
        x_max = image.shape[0] - self.patch_size[0]
        y_max = image.shape[1] - self.patch_size[1]

        x = np.random.randint(x_max)
        y = np.random.randint(y_max)

        image = image[x:(x + self.patch_size[0]), y:(y + self.patch_size[1])]

        if np.random.choice([True, False]):
            image = np.fliplr(image)

        image = np.rot90(image, k=np.random.randint(4))

        return image


class TrainingDiagSetDataset(AbstractDiagSetDataset):
    def batch(self):
        probabilities = [self.class_ratios[label] for label in self.numeric_labels]

        labels = np.random.choice(self.numeric_labels, self.batch_size, p=probabilities)
        images = np.array([self.buffers[label].get() for label in labels])

        return images, labels


class EvaluationDiagSetDataset(AbstractDiagSetDataset):
    def __init__(self, **kwargs):
        assert kwargs.get('augment', False) is False
        assert kwargs.get('shuffling', False) is False
        assert kwargs.get('class_ratios') is None

        kwargs['augment'] = False
        kwargs['shuffling'] = False
        kwargs['class_ratios'] = None

        self.current_numeric_label_index = 0
        self.current_batch_index = 0

        super().__init__(**kwargs)

    def batch(self):
        labels = []
        images = []

        for _ in range(self.batch_size):
            label = self.numeric_labels[self.current_numeric_label_index]

            while len(self.blob_paths[label]) == 0:
                self.current_numeric_label_index = (self.current_numeric_label_index + 1) % len(self.numeric_labels)

                label = self.numeric_labels[self.current_numeric_label_index]

            image = self.buffers[label].get()

            labels.append(label)
            images.append(image)

            self.current_batch_index += 1

            if self.current_batch_index >= self.class_distribution[label]:
                self.current_batch_index = 0
                self.current_numeric_label_index += 1

                if self.current_numeric_label_index >= len(self.numeric_labels):
                    self.current_numeric_label_index = 0

                    break

        labels = np.array(labels)
        images = np.array(images)

        return images, labels


class Scan:
    def __init__(self, tissue_tag, level=0, scan_id=None, scan_name=None, source='server'):
        assert tissue_tag in config.TISSUE_TAGS
        assert (scan_name is not None) or (scan_id is not None)
        assert source in ['local', 'server']

        self.tissue_tag = tissue_tag
        self.scan_id = scan_id
        self.scan_name = scan_name
        self.scan_title = scan_id or scan_name
        self.level = level
        self.source = source

        if source == 'server':
            self.slide = None

            if scan_name is None:
                self.scan_name = db.fetch_scan_name(scan_id)

            if scan_id is None:
                self.scan_id = db.fetch_scan_id(scan_name)

            (self.width, self.height), self.offset, self.mpp, self.lens = ndp.fetch_metadata(scan_id=scan_id)
        elif source == 'local':
            assert scan_name is not None

            self.slide = local.read_slide(scan_name)
            self.scan_id = None

            (self.width, self.height), self.offset, self.mpp, self.lens = local.extract_metadata(self.slide)
        else:
            raise NotImplementedError

        self.downsample = 2 ** level
        self.magnification = self.lens / self.downsample
        self.polygons = None
        self.zero_level_scan = None

    def load_annotation_polygons(self):
        if self.source == 'server':
            annotations = db.fetch_xml_annotations_for_scan(self.scan_id)
        elif self.source == 'local':
            annotations = local.read_annotations(self.scan_name)
        else:
            raise NotImplementedError

        self.polygons = prepare_multipolygons(
            annotations, self.tissue_tag, (self.width, self.height), self.offset, self.mpp
        )

    def get_patch(self, x, y, patch_size):
        if type(patch_size) is int:
            patch_size = (patch_size, patch_size)

        if self.source == 'server':
            source_roi_size = (int(patch_size[0] * self.downsample), int(patch_size[1] * self.downsample))
            centre_coordinate = (x + source_roi_size[0] // 2, y + source_roi_size[1] // 2)
            patch = ndp.fetch_region(patch_size,
                                     centre_coordinate, source_roi_size,
                                     scan_id=self.scan_id)
        elif self.source == 'local':
            patch = np.array(self.slide.read_region((x, y), self.level, patch_size))[:, :, :3]
        else:
            raise NotImplementedError

        assert patch.shape[0] == patch_size[0]
        assert patch.shape[1] == patch_size[1]

        return patch

    def get_patch_overlap_ratios(self, x, y, patch_size):
        if type(patch_size) is int:
            patch_size = (patch_size, patch_size)

        if self.polygons is None:
            self.load_annotation_polygons()

        patch_polygon = Polygon((
            (x, y),
            (x, y + int(patch_size[1] * self.downsample)),
            (x + int(patch_size[0] * self.downsample), y + int(patch_size[1] * self.downsample)),
            (x + int(patch_size[0] * self.downsample), y)
        ))

        overlap_ratios = {}

        for label in config.USABLE_LABELS[self.tissue_tag]:
            if label in self.polygons.keys():
                intersection = patch_polygon.intersection(self.polygons[label])
                overlap_ratios[label] = intersection.area / patch_polygon.area
            else:
                overlap_ratios[label] = 0.0

        return overlap_ratios

    def get_patch_ground_truth_label(self, x, y, patch_size, minimum_overlap=0.75):
        if self.level == 0:
            return self._get_zero_level_patch_ground_truth_label(x, y, patch_size, minimum_overlap)
        else:
            return self._get_higher_level_patch_ground_truth_label(x, y, patch_size, minimum_overlap)

    def _get_zero_level_patch_ground_truth_label(self, x, y, patch_size, minimum_overlap=0.75):
        if type(patch_size) is int:
            patch_size = (patch_size, patch_size)

        overlap_ratios = self.get_patch_overlap_ratios(x, y, patch_size)

        sufficiently_overlapping_labels = [
            label for label in overlap_ratios.keys() if overlap_ratios[label] >= minimum_overlap
        ]

        if len(sufficiently_overlapping_labels) == 0:
            return None
        elif len(sufficiently_overlapping_labels) == 1:
            return sufficiently_overlapping_labels[0]
        else:
            if self.scan_id is not None:
                identifier_string = 'ID = "%s"' % self.scan_id
            elif self.scan_name is not None:
                identifier_string = 'name = "%s"' % self.scan_name
            else:
                raise ValueError

            logging.getLogger('diaglib').warning(
                'Found multiple possible labels (%s) for scan with %s at level = %d, '
                'position = (%d, %d) and patch size = %s. Setting label to None.' % (
                    ', '.join(sufficiently_overlapping_labels), identifier_string, self.level, x, y, patch_size
                )
            )

            return None

    def _get_higher_level_patch_ground_truth_label(self, x, y, patch_size, minimum_overlap=0.75):
        if type(patch_size) is int:
            patch_size = (patch_size, patch_size)

        if self.zero_level_scan is None:
            self.zero_level_scan = Scan(
                tissue_tag=self.tissue_tag,
                level=0,
                scan_id=self.scan_id,
                scan_name=self.scan_name,
                source=self.source
            )

        zero_level_labels = []

        for x_zero in range(x, x + int(patch_size[0] * self.downsample), patch_size[0]):
            for y_zero in range(y, y + int(patch_size[1] * self.downsample), patch_size[1]):
                zero_level_label = self.zero_level_scan._get_zero_level_patch_ground_truth_label(
                    x_zero, y_zero, patch_size, minimum_overlap
                )

                if zero_level_label is not None:
                    zero_level_labels.append(zero_level_label)

        if len(zero_level_labels) == 0:
            return None

        for label in config.LABEL_ORDER[self.tissue_tag]:
            if label in zero_level_labels:
                return label

        return None

    def get_thumbnail(self, patch_size=(224, 224), scaling=1):
        assert type(scaling) is int and scaling >= 1

        if type(patch_size) is int:
            patch_size = (patch_size, patch_size)

        width = self.width - self.width % patch_size[0]
        height = self.height - self.height % patch_size[1]

        if self.source == 'server':
            thumbnail = ndp.fetch_region(
                (width // patch_size[0] * scaling, height // patch_size[1] * scaling),
                (width // 2, height // 2), (width, height),
                scan_id=self.scan_id
            )
        else:
            raise NotImplementedError

        return thumbnail

    def get_polygon_bounds(self, x_step, y_step):
        if self.polygons is None:
            self.load_annotation_polygons()

        non_empty_bounds = [polygon.bounds for polygon in self.polygons.values() if polygon.bounds]

        x_start = int(np.min([bounds[0] for bounds in non_empty_bounds]))
        y_start = int(np.min([bounds[1] for bounds in non_empty_bounds]))

        x_start = x_start - x_start % x_step
        y_start = y_start - y_start % y_step

        x_end = int(np.max([bounds[2] for bounds in non_empty_bounds]))
        y_end = int(np.max([bounds[3] for bounds in non_empty_bounds]))

        x_end = x_end + (x_step - x_end % x_step)
        y_end = y_end + (y_step - y_end % y_step)

        return x_start, x_end, y_start, y_end

    def get_ground_truth_maps(self, patch_size=(224, 224), minimum_overlap=0.75, scaling=1):
        assert type(scaling) is int and scaling >= 1

        if type(patch_size) is int:
            patch_size = (patch_size, patch_size)

        if self.polygons is None:
            self.load_annotation_polygons()

        scan_width = self.width - self.width % patch_size[0]
        scan_height = self.height - self.height % patch_size[1]

        map_width = scan_width // patch_size[0] * scaling
        map_height = scan_height // patch_size[1] * scaling

        ground_truth_maps = {
            label: np.zeros((map_height, map_width)) for label in config.USABLE_LABELS[self.tissue_tag]
        }

        x_step = int(patch_size[0] * self.downsample)
        y_step = int(patch_size[1] * self.downsample)

        x_start, x_end, y_start, y_end = self.get_polygon_bounds(x_step, y_step)

        for x in range(x_start, x_end, x_step):
            for y in range(y_start, y_end, y_step):
                label = self.get_patch_ground_truth_label(x, y, patch_size, minimum_overlap)

                if label is not None:
                    map_x_start = x // patch_size[0]
                    map_y_start = y // patch_size[1]

                    map_x_end = map_x_start + int(self.downsample)
                    map_y_end = map_y_start + int(self.downsample)

                    map_x_start *= scaling
                    map_y_start *= scaling

                    map_x_end *= scaling
                    map_y_end *= scaling

                    ground_truth_maps[label][map_y_start:map_y_end, map_x_start:map_x_end] = 1.0

        return ground_truth_maps

    def get_foreground_as_normal_map(self, patch_size=(224, 224), model_name='SegSet_VDSR', reset_graph=True, scaling=1):
        assert type(scaling) is int and scaling >= 1

        if type(patch_size) is int:
            patch_size = (patch_size, patch_size)

        thumbnail = self.get_thumbnail(patch_size)
        foreground = segment_foreground_vdsr(thumbnail, model_name, reset_graph)

        scan_width = self.width - self.width % patch_size[0]
        scan_height = self.height - self.height % patch_size[1]

        map_width = scan_width // patch_size[0] * scaling
        map_height = scan_height // patch_size[1] * scaling

        ground_truth_maps = {
            label: np.zeros((map_height, map_width)) for label in config.USABLE_LABELS[self.tissue_tag]
        }

        for x in range(foreground.shape[0]):
            for y in range(foreground.shape[1]):
                foreground_region = foreground[x:(x + int(self.downsample)), y:(y + int(self.downsample))]

                if np.max(foreground_region) == 1.0:
                    map_x_start = x
                    map_y_start = y

                    map_x_end = map_x_start + int(self.downsample)
                    map_y_end = map_y_start + int(self.downsample)

                    map_x_start *= scaling
                    map_y_start *= scaling

                    map_x_end *= scaling
                    map_y_end *= scaling

                    ground_truth_maps['N'][map_x_start:map_x_end, map_y_start:map_y_end] = 1.0

        return ground_truth_maps
