import imageio
import numpy as np
import tensorflow as tf

from diaglib import config
from diaglib.data.diagset.loading import ndp
from diaglib.learn.cnn.segment.models import VDSR
from itertools import product
from queue import Queue
from threading import Thread


def initialize_scan_predictions(patch_size, n_labels, width, height):
    scan_predictions = np.zeros((height // patch_size, width // patch_size, n_labels), dtype=np.float32)

    return scan_predictions


def get_thumbnail(scan_id, patch_size, width, height):
    thumbnail = ndp.fetch_region(
        (width // patch_size, height // patch_size),
        (width // 2, height // 2), (width, height),
        scan_id=scan_id
    )

    return thumbnail


def save_thumbnail(thumbnail, scan_id, model_name):
    prediction_directory = _get_prediction_directory(scan_id, model_name)

    imageio.imwrite(str(prediction_directory / 'thumbnail.png'), thumbnail)


def segment_foreground_color(image, lower_bounds=(45, 45, 45), upper_bounds=(210, 210, 210)):
    """
    Given an image, return binary mask with the same width and height, and mask[i][j] == 1.0 indicating that pixel
    at position (i, j) was recognized as foreground. Pixel is recognized as background if its RGB values are either
    lower than lower bounds, or higher than upper bounds.
    """
    output = np.zeros(image.shape[:2])

    pixels_below = (image[:, :, 0] < lower_bounds[0]) & (image[:, :, 1] < lower_bounds[1]) & (
                image[:, :, 2] < lower_bounds[2])
    pixels_above = (image[:, :, 0] > upper_bounds[0]) & (image[:, :, 1] > upper_bounds[1]) & (
                image[:, :, 2] > upper_bounds[2])

    output[~(pixels_above | pixels_below)] = 1.0

    return output


def segment_foreground_vdsr(image, model_name='SegSet_VDSR', reset_graph=True):
    if reset_graph:
        tf.reset_default_graph()

    network = VDSR(model_name)

    with tf.Session() as session:
        network.restore(session)

        output = network.outputs.eval(feed_dict={network.inputs: [image]}, session=session)[0, :, :, 0]

    output[output <= 127] = 0.0
    output[output > 127] = 1.0

    return output


def save_foreground(foreground, scan_id, model_name):
    foreground_png = _convert_probability_to_int(foreground, np.uint16)

    prediction_directory = _get_prediction_directory(scan_id, model_name)

    imageio.imwrite(str(prediction_directory / 'foreground.png'), foreground_png)


def accumulate_patch_positions(patch_size, width, height, downsample, foreground=None, scan_predictions=None,
                               background_label=None, n_labels=None):
    """
    Accumulate positions (upper left corners) of patches for which predictions should be made. If foreground is given,
    also segment foreground and automatically set predictions at background positions.
    """
    if foreground is not None:
        assert scan_predictions is not None
        assert background_label is not None
        assert n_labels is not None

    patch_positions = []

    x_step, y_step = int(patch_size * downsample), int(patch_size * downsample)
    x_range, y_range = range(0, width, x_step), range(0, height, y_step)

    if foreground is not None:
        for (x, y) in product(x_range, y_range):
            x_, y_ = x // patch_size, y // patch_size

            foreground_region = foreground[y_:(y_ + int(downsample)), x_:(x_ + int(downsample))]

            if np.max(foreground_region) == 1.0:
                patch_positions.append((x, y))
            else:
                batch_predictions = [np.zeros(n_labels)]
                batch_predictions[0][background_label] = 1.0
                scan_predictions = update_scan_predictions(
                    scan_predictions, batch_predictions, [(x, y)], patch_size, downsample
                )
    else:
        for (x, y) in product(x_range, y_range):
            patch_positions.append((x, y))

    return patch_positions


def get_scan_dimensions(scan_id, magnification, patch_size):
    (width, height), _, _, lens = ndp.fetch_metadata(scan_id=scan_id)
    width, height = width - width % patch_size, height - height % patch_size
    downsample = lens / magnification

    return width, height, downsample


def get_n_labels(tissue_tag):
    return len(set(config.LABEL_DICTIONARIES[tissue_tag].values()))


def get_background_label(tissue_tag):
    return config.LABEL_DICTIONARIES[tissue_tag]['BG']


def setup_map_production(scan_id, tissue_tag, magnification, patch_size, model_name, segment_foreground=True,
                         segmentation_function=segment_foreground_vdsr):
    width, height, downsample = get_scan_dimensions(scan_id, magnification, patch_size)
    n_labels = get_n_labels(tissue_tag)
    background_label = get_background_label(tissue_tag)

    scan_predictions = initialize_scan_predictions(patch_size, n_labels, width, height)

    thumbnail = get_thumbnail(scan_id, patch_size, width, height)
    save_thumbnail(thumbnail, scan_id, model_name)

    if segment_foreground:
        foreground = segmentation_function(thumbnail)
        save_foreground(foreground, scan_id, model_name)

        patch_positions = accumulate_patch_positions(
            patch_size, width, height, downsample, foreground, scan_predictions, background_label, n_labels
        )
    else:
        patch_positions = accumulate_patch_positions(patch_size, width, height, downsample)

    return scan_predictions, patch_positions


def fill_buffer(buffer, patch_positions, scan_id, patch_size, downsample):
    """
    Given a buffer object, such as Queue, and a list of (x, y) positions, fill the buffer with tuples containing said
    positions paired with scan patches, fetched from NDP server. Used to fetch patches asynchronously during prediction.
    """
    x_step, y_step = int(patch_size * downsample), int(patch_size * downsample)

    for (x, y) in patch_positions:
        centre_coordinate = (x + x_step // 2, y + y_step // 2)
        source_roi_size = (x_step, y_step)
        patch = ndp.fetch_region((patch_size, patch_size),
                                 centre_coordinate, source_roi_size,
                                 scan_id=scan_id)

        buffer.put([(x, y), patch])


def update_scan_predictions(scan_predictions, batch_predictions, batch_positions, patch_size, downsample):
    """
    Given a possibly empty matrix of scan predictions, with dimensions equal to that of the underlying scan
    divided by patch size, as well as the lists of predictions made by the classifier for a batch and positions
    at which said predictions were made, fill the scan predictions matrix.
    """
    for prediction, (x, y) in zip(batch_predictions, batch_positions):
        x_, y_ = x // patch_size, y // patch_size
        scan_predictions[y_:(y_ + int(downsample)), x_:(x_ + int(downsample))] = prediction

    return scan_predictions


def clear_background(scan_predictions, scan_id, tissue_tag, patch_size, magnification=40,
                     segmentation_function=segment_foreground_vdsr):
    """
    Postprocess scan predictions to remove background. Used for predictions generated by ensembles using multiple
    magnifications, when in some cases tissue edges can be incorrectly classified for low magnifications.
    """
    width, height, downsample = get_scan_dimensions(scan_id, magnification, patch_size)
    n_labels = get_n_labels(tissue_tag)
    background_label = get_background_label(tissue_tag)
    thumbnail = get_thumbnail(scan_id, patch_size, width, height)
    foreground = segmentation_function(thumbnail)

    x_step, y_step = int(patch_size * downsample), int(patch_size * downsample)
    x_range, y_range = range(0, width, x_step), range(0, height, y_step)

    for (x, y) in product(x_range, y_range):
        x_, y_ = x // patch_size, y // patch_size

        foreground_region = foreground[y_:(y_ + int(downsample)), x_:(x_ + int(downsample))]

        if np.max(foreground_region) == 0.0:
            batch_predictions = [np.zeros(n_labels)]
            batch_predictions[0][background_label] = 1.0
            scan_predictions = update_scan_predictions(
                scan_predictions, batch_predictions, [(x, y)], patch_size, downsample
            )

    return scan_predictions


def initialize_buffer(buffer_size):
    buffer = Queue(buffer_size)

    return buffer


def start_daemon(buffer, patch_positions, scan_id, patch_size, downsample):
    Thread(target=fill_buffer, daemon=True, args=(buffer, patch_positions, scan_id, patch_size, downsample)).start()


def save_maps(scan_id, model_name, tissue_tag, scan_predictions):
    n_labels = get_n_labels(tissue_tag)

    prediction_directory = _get_prediction_directory(scan_id, model_name)

    for i in range(n_labels):
        label_dictionary = config.LABEL_DICTIONARIES[tissue_tag]
        label_string = '+'.join([k for k, v in label_dictionary.items() if v == i])

        probability_map = scan_predictions[:, :, i].copy()
        probability_map_png = _convert_probability_to_int(probability_map, np.uint16)

        imageio.imwrite(str(prediction_directory / ('%s.png' % label_string)), probability_map_png)

        binarized_prediction = np.zeros(probability_map.shape)
        binarized_prediction[np.argmax(scan_predictions, axis=2) == i] = 1.0
        binarized_prediction_png = _convert_probability_to_int(binarized_prediction, np.uint16)

        imageio.imwrite(str(prediction_directory / ('%s_binarized.png' % label_string)), binarized_prediction_png)


def _get_prediction_directory(scan_id, model_name):
    prediction_directory = config.PREDICTIONS_PATH / model_name / scan_id
    prediction_directory.mkdir(parents=True, exist_ok=True)

    return prediction_directory


def _convert_probability_to_int(array, dtype):
    return (array * np.iinfo(dtype).max).round().astype(dtype)
