import numpy as np
import tensorflow as tf

from diaglib.data.imagenet.containers import IMAGENET_IMAGE_MEAN
from diaglib.predict.maps import common
from tqdm import tqdm


def predict_batch(network, session, batch_patches, subtract_mean=True):
    batch_inputs = np.array(batch_patches, dtype=np.float32)

    if subtract_mean:
        batch_inputs -= IMAGENET_IMAGE_MEAN

    batch_predictions = tf.nn.softmax(network.outputs).eval(
        feed_dict={network.inputs: batch_inputs}, session=session
    )

    return batch_predictions


def restore_network(model_class, model_name, session, patch_size, n_labels):
    network = model_class(model_name, [n_labels], [patch_size, patch_size, 3])
    network.restore(session, model_name)

    return network


def fill_scan_predictions(network, session, patch_positions, scan_predictions, buffer, batch_size,
                          patch_size, downsample, cache_patches=False, cache=None):
    batch_patches, batch_positions = [], []

    using_previously_cached_patches = (cache_patches and cache is not None)
    currently_caching_patches = (cache_patches and cache is None)

    if currently_caching_patches:
        try:
            cache = np.empty((len(patch_positions), patch_size, patch_size, 3), dtype=np.uint8)
        except MemoryError:
            currently_caching_patches = False

    for i, (x, y) in enumerate(tqdm(patch_positions)):
        if using_previously_cached_patches:
            patch = cache[i]
        else:
            _, patch = buffer.get()

            if currently_caching_patches:
                cache[i] = patch

        batch_patches.append(patch)
        batch_positions.append((x, y))

        if len(batch_patches) >= batch_size:
            batch_predictions = predict_batch(network, session, batch_patches)
            scan_predictions = common.update_scan_predictions(
                scan_predictions, batch_predictions, batch_positions, patch_size, downsample
            )

            batch_patches, batch_positions = [], []

    if len(batch_patches) > 0:
        batch_predictions = predict_batch(network, session, batch_patches)
        scan_predictions = common.update_scan_predictions(
            scan_predictions, batch_predictions, batch_positions, patch_size, downsample
        )

    return scan_predictions, cache


def produce_maps(scan_id, tissue_tag, magnification, patch_size, batch_size, buffer_size, model_class, model_name,
                 segment_foreground=True, segmentation_function=common.segment_foreground_vdsr,
                 reset_graph=True):
    scan_predictions, patch_positions = common.setup_map_production(
        scan_id, tissue_tag, magnification, patch_size, model_name, segment_foreground, segmentation_function
    )

    _, _, downsample = common.get_scan_dimensions(scan_id, magnification, patch_size)
    n_labels = common.get_n_labels(tissue_tag)

    buffer = common.initialize_buffer(buffer_size)
    common.start_daemon(buffer, patch_positions, scan_id, patch_size, downsample)

    if reset_graph:
        tf.reset_default_graph()

    with tf.Session() as session:
        network = restore_network(model_class, model_name, session, patch_size, n_labels)

        scan_predictions, _ = fill_scan_predictions(
            network, session, patch_positions, scan_predictions, buffer,
            batch_size, patch_size, downsample
        )

    common.save_maps(scan_id, model_name, tissue_tag, scan_predictions)
