import imageio
import numpy as np

from diaglib import config
from diaglib.utils import shuffle_combined
from skimage import transform


class TrainingSegSetDataset:
    def __init__(self, batch_size=64, patch_size=21, downsample=8.0, shuffling=True):
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.downsample = downsample
        self.shuffling = shuffling

        self.scan_ids = [path.stem.replace('_mask', '') for path in config.SEGSET_TRAIN_SET_PATH.glob('*_mask.jpg')]

        self.inputs = []
        self.outputs = []

        for scan_id in self.scan_ids:
            thumbnail = imageio.imread(str(config.SEGSET_TRAIN_SET_PATH / ('%s.jpg' % scan_id)))
            mask = imageio.imread(str(config.SEGSET_TRAIN_SET_PATH / ('%s_mask.jpg' % scan_id)))[:, :, :1]

            if downsample != 1.0:
                thumbnail = transform.rescale(thumbnail, 1.0 / downsample) * 255
                mask = transform.rescale(mask, 1.0 / downsample) * 255

            for x in range(0, thumbnail.shape[0] - thumbnail.shape[0] % patch_size, patch_size):
                for y in range(0, thumbnail.shape[1] - thumbnail.shape[1] % patch_size, patch_size):
                    self.inputs.append(thumbnail[x:(x + patch_size), y:(y + patch_size)])
                    self.outputs.append(mask[x:(x + patch_size), y:(y + patch_size)])

        self.inputs = np.array(self.inputs)
        self.outputs = np.array(self.outputs)

        self.length = len(self.inputs)
        self.current_index = 0

        if self.shuffling:
            self.shuffle()

    def batch(self):
        batch_inputs = self.inputs[self.current_index:(self.current_index + self.batch_size)]
        batch_outputs = self.outputs[self.current_index:(self.current_index + self.batch_size)]

        self.current_index += self.batch_size

        if self.current_index >= self.length:
            self.current_index = 0

            if self.shuffling:
                self.shuffle()

        return batch_inputs, batch_outputs

    def shuffle(self):
        self.inputs, self.outputs = shuffle_combined(self.inputs, self.outputs)


class TestSegSetDataset:
    def __init__(self, downsample=8.0):
        self.downsample = downsample

        self.scan_ids = []
        self.inputs = []

        for path in config.SEGSET_TEST_SET_PATH.glob('*.jpg'):
            self.scan_ids.append(path.stem)
            self.inputs.append(transform.rescale(imageio.imread(str(path)), 1.0 / downsample) * 255)

        self.length = len(self.inputs)
        self.current_index = 0

    def get(self):
        scan_id = self.scan_ids[self.current_index]
        thumbnail = self.inputs[self.current_index]

        self.current_index += 1

        if self.current_index >= self.length:
            self.current_index = 0

        return scan_id, thumbnail
