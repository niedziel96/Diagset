import json
import logging
import numpy as np

from diaglib import config
from diaglib.utils import shuffle_combined
from pathlib import Path
from queue import Queue
from threading import Thread


IMAGENET_IMAGE_MEAN = [123.68, 116.779, 103.939]
TRANSLATIONS_PATH = Path(__file__).parent / 'translations.json'


class ImageNetDataset:
    def __init__(self, partition, batch_size, patch_size=(224, 224), augment=True, subtract_mean=True, shuffling=True,
                 translate_labels=False, offset_labels=False, buffer_size=1):
        assert partition in ['train', 'validation']

        self.partition = partition
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.augment = augment
        self.subtract_mean = subtract_mean
        self.shuffling = shuffling
        self.translate_labels = translate_labels
        self.offset_labels = offset_labels
        self.buffer_size = buffer_size

        self.buffer = Queue(buffer_size)
        self.current_blob = None
        self.current_blob_index = 0
        self.current_batch_index = 0
        self.length = 0

        with open(TRANSLATIONS_PATH, 'r') as f:
            string_translations = json.load(f)

            self.label_translations = {int(k): v for k, v in string_translations.items()}

        self.partition_path = config.IMAGENET_BLOBS_PATH / partition

        logging.getLogger('diaglib').info('Loading blob paths...')

        image_paths = []
        label_paths = []

        for path in sorted((self.partition_path / 'images').iterdir()):
            image_paths.append(path)

        for path in sorted((self.partition_path / 'labels').iterdir()):
            label_paths.append(path)
            labels = np.load(str(path))

            self.length += len(labels)

        self.inputs = np.array(image_paths)
        self.outputs = np.array(label_paths)

        logging.getLogger('diaglib').info('Found %d images.' % self.length)

        if self.shuffling:
            self.shuffle()

        Thread(target=self.fill_buffer, daemon=True).start()

    def fill_buffer(self):
        while True:
            image_blob_path = self.inputs[self.current_blob_index]
            label_blob_path = self.outputs[self.current_blob_index]

            images = np.load(image_blob_path)
            labels = np.load(label_blob_path)

            blob = self.prepare_blob((images, labels))

            self.current_blob_index += 1

            if self.current_blob_index >= len(self.inputs):
                self.current_blob_index = 0

                if self.shuffling:
                    self.shuffle()

            self.buffer.put(blob)

    def batch(self):
        if self.current_blob is None:
            self.current_blob = self.buffer.get()

        images = self.current_blob[0][self.current_batch_index:(self.current_batch_index + self.batch_size)]
        labels = self.current_blob[1][self.current_batch_index:(self.current_batch_index + self.batch_size)]

        self.current_batch_index += self.batch_size

        if self.current_batch_index >= len(self.current_blob[0]):
            self.current_batch_index = 0
            self.current_blob = None

        return images, labels

    def shuffle(self):
        self.inputs, self.outputs = shuffle_combined(self.inputs, self.outputs)

    def prepare_blob(self, blob):
        images, labels = blob
        images, labels = shuffle_combined(images, labels)

        transformed_images = []

        for i in range(len(images)):
            if self.translate_labels:
                labels[i] = self.label_translations[labels[i]]

            if self.offset_labels:
                labels[i] += 1

            image = images[i].astype(np.float32)

            if self.augment:
                image = self._augment(image)
            else:
                x = (image.shape[0] - self.patch_size[0]) // 2
                y = (image.shape[1] - self.patch_size[1]) // 2

                image = image[x:(x + self.patch_size[0]), y:(y + self.patch_size[1])]

            if self.subtract_mean:
                image -= IMAGENET_IMAGE_MEAN

            transformed_images.append(image)

        transformed_images = np.array(transformed_images)

        return transformed_images, labels

    def _augment(self, image):
        x_max = image.shape[0] - self.patch_size[0]
        y_max = image.shape[1] - self.patch_size[1]

        x = np.random.randint(x_max)
        y = np.random.randint(y_max)

        image = image[x:(x + self.patch_size[0]), y:(y + self.patch_size[1])]

        if np.random.choice([True, False]):
            image = np.fliplr(image)

        return image
