import numpy as np
import tensorflow as tf

from diaglib.predict.maps import cnn, common


def merge_model_scan_predictions(model_scan_predictions):
    merged = np.array(model_scan_predictions)

    return np.mean(merged, axis=0)


def produce_maps(scan_id, tissue_tag, magnifications, patch_size, batch_size, buffer_size,
                 ensemble_name, model_classes, model_names, segment_foreground=True,
                 segmentation_function=common.segment_foreground_vdsr, postprocess_background=True,
                 reset_graph=True):
    n_labels = common.get_n_labels(tissue_tag)

    model_scan_predictions = []

    for model_class, model_name, magnification in zip(model_classes, model_names, magnifications):
        scan_predictions, patch_positions = common.setup_map_production(
            scan_id, tissue_tag, magnification, patch_size,
            ensemble_name, segment_foreground, segmentation_function
        )

        _, _, downsample = common.get_scan_dimensions(scan_id, magnification, patch_size)

        individual_model_scan_predictions = scan_predictions.copy()

        buffer = common.initialize_buffer(buffer_size)
        common.start_daemon(buffer, patch_positions, scan_id, patch_size, downsample)

        if reset_graph:
            tf.reset_default_graph()

        with tf.Session() as session:
            network = cnn.restore_network(model_class, model_name, session, patch_size, n_labels)

            individual_model_scan_predictions, patch_cache = cnn.fill_scan_predictions(
                network, session, patch_positions, individual_model_scan_predictions,
                buffer, batch_size, patch_size, downsample
            )

        model_scan_predictions.append(individual_model_scan_predictions)

        common.save_maps('%s/%s' % (scan_id, model_name), ensemble_name, tissue_tag, individual_model_scan_predictions)

    scan_predictions = merge_model_scan_predictions(model_scan_predictions)

    if postprocess_background and len(magnifications) > 1:
        scan_predictions = common.clear_background(
            scan_predictions, scan_id, tissue_tag, patch_size, np.max(magnifications), segmentation_function
        )

    common.save_maps(scan_id, ensemble_name, tissue_tag, scan_predictions)
