import imageio
import logging
import numpy as np
import scipy.io
import skimage.transform

from diaglib import config
from diaglib.utils import shuffle_combined
from tqdm import tqdm


def prepare_imagenet_blobs(blob_size=4096, image_size=(256, 256)):
    blobs_path = config.IMAGENET_BLOBS_PATH

    for partition in ['train', 'validation']:
        logging.getLogger('diaglib').info('Preparing %s partition...' % partition)

        partition_path = blobs_path / partition
        partition_images_path = partition_path / 'images'
        partition_labels_path = partition_path / 'labels'

        for path in [partition_images_path, partition_labels_path]:
            path.mkdir(parents=True, exist_ok=True)

        logging.getLogger('diaglib').info('Loading image paths...')

        if partition == 'train':
            train_set_path = config.IMAGENET_ORIGINAL_TRAIN_SET_PATH
            metadata_file_path = config.IMAGENET_ORIGINAL_METADATA_PATH

            synset_dictionary = {synset[0][1][0]: synset[0][0][0][0] - 1 for synset in
                                 scipy.io.loadmat(str(metadata_file_path))['synsets']}
            synset_paths = [path for path in train_set_path.iterdir() if path.is_dir()]

            image_paths = []
            labels = []

            for synset_path in tqdm(synset_paths):
                synset = synset_path.stem
                label = synset_dictionary[synset]

                for image_path in synset_path.iterdir():
                    image_paths.append(image_path)
                    labels.append(label)
        elif partition == 'validation':
            validation_set_path = config.IMAGENET_ORIGINAL_VALIDATION_SET_PATH
            labels_path = config.IMAGENET_ORIGINAL_LABELS_PATH

            image_paths = []

            with open(labels_path) as f:
                labels = list(map(lambda x: int(x) - 1, f.readlines()))

            for i in range(50000):
                image_paths.append(validation_set_path / ('ILSVRC2012_val_%.8d.JPEG' % (i + 1)))
        else:
            raise NotImplementedError

        image_paths = np.array(image_paths)
        labels = np.array(labels, dtype=np.uint16)

        image_paths, labels = shuffle_combined(image_paths, labels)

        n_blobs = int(np.ceil(len(image_paths) / blob_size))

        logging.getLogger('diaglib').info('Extracting blobs...')

        for i in tqdm(range(n_blobs)):
            blob_image_paths = image_paths[(i * blob_size):((i + 1) * blob_size)]
            blob_labels = labels[(i * blob_size):((i + 1) * blob_size)]

            blob_images = []

            for image_path in blob_image_paths:
                try:
                    image = imageio.imread(str(image_path))

                    assert image.dtype == np.uint8
                    assert len(image.shape) in [2, 3]

                    if len(image.shape) == 3:
                        assert image.shape[2] == 3

                    if len(image.shape) == 2:
                        image = np.stack((image,) * 3, axis=-1)

                    if image.shape[0] != image.shape[1]:
                        x, y = image.shape[:2]
                        cropped_size = np.min((x, y))
                        x_start = (x - cropped_size) // 2
                        x_end = (x - cropped_size) // 2 + cropped_size
                        y_start = (y - cropped_size) // 2
                        y_end = (y - cropped_size) // 2 + cropped_size

                        image = image[x_start:x_end, y_start:y_end, :]

                    image = skimage.transform.resize(image, image_size, mode='wrap')
                    image *= 255
                    image = image.astype(np.uint8)

                    blob_images.append(image)
                except AssertionError:
                    image = imageio.imread(str(image_path))

                    logging.getLogger('diaglib').warning('Failed to load image from "%s". '
                                                         'Got image with data type "%s" and shape "%s".'
                                                         % (image_path, image.dtype, image.shape))

            blob_images = np.array(blob_images)

            np.save(str(partition_images_path / ('ImageNet.train.blob.%d.npy' % i)), blob_images)
            np.save(str(partition_labels_path / ('ImageNet.train.blob.%d.npy' % i)), blob_labels)
