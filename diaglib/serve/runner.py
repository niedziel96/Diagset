import argparse
import logging
import os
import time

from diaglib import config
from diaglib.learn.cnn.classify.model_classes import MODEL_CLASSES
from diaglib.predict.maps.ensemble import produce_maps
from diaglib.serve import db


fh = logging.FileHandler(config.REQUESTS_RUNNER_LOG_PATH)
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

logger = logging.getLogger('diaglib')
logger.setLevel(level=logging.INFO)
logger.addHandler(fh)


def run(batch_size=32, buffer_size=128, gpu=0, clear_on_start=True):
    if clear_on_start:
        db.clear_processed_requests()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    while True:
        request = db.pull_request()

        if request is None:
            time.sleep(config.REQUESTS_MIN_DELAY_BETWEEN_DB_CHECKS_SECONDS)

            continue

        logger.info('Processing request for scan with ID = %s...' % request.scan_id)

        ensemble_parameters = config.ENSEMBLE_PARAMETERS.get(request.tissue_tag)

        if ensemble_parameters is None:
            logger.info(
                'Failed to process request for scan with ID = %s due to '
                'unconfigured config.ENSEMBLE_PARAMETERS["%s"].' % (request.scan_id, request.tissue_tag))

            db.finalize_request(request.id, 'failed')

            continue

        ensemble_name = None
        model_class_string = None
        magnification_string = None
        patch_size = None

        try:
            ensemble_name = ensemble_parameters['ensemble_name']
            model_class_string = ','.join(ensemble_parameters['model_classes'])
            magnification_string = ','.join([
                str(magnification) for magnification in ensemble_parameters['magnifications']
            ])
            patch_size = ensemble_parameters['patch_size']

            model_classes = [MODEL_CLASSES[cls] for cls in ensemble_parameters['model_classes']]

            produce_maps(
                scan_id=request.scan_id, tissue_tag=request.tissue_tag,
                magnifications=ensemble_parameters['magnifications'],
                patch_size=ensemble_parameters['patch_size'],
                ensemble_name=ensemble_parameters['ensemble_name'],
                model_names=ensemble_parameters['model_names'],
                model_classes=model_classes, segment_foreground=True,
                batch_size=batch_size, buffer_size=buffer_size,
                postprocess_background=True
            )
        except Exception as e:
            logger.info('Failed to process request for scan with ID = %s.' % request.scan_id)
            logger.info(e)

            status = 'failed'
        else:
            logger.info('Successfully processed request for scan with ID = %s.' % request.scan_id)

            status = 'finished'

        db.finalize_request(request.id, status, ensemble_name, model_class_string, magnification_string, patch_size)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-buffer_size', type=int, default=128)
    parser.add_argument('-gpu', type=int, default=0)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    run(**vars(args))
