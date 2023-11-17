import imageio
import numpy as np

from diaglib import config
from diaglib.serve import db


class Report:
    def __init__(self, scan_id):
        request_status = db.get_request_status(scan_id)

        if request_status != 'finished':
            raise ValueError(
                'Unable to create report for scan with ID %s: expected status "finished", received "%s".' %
                (scan_id, request_status)
            )

        model_name = db.find_requests(scan_id=scan_id)[0].model_name
        prediction_path = config.PREDICTIONS_PATH / model_name / scan_id

        assert prediction_path.exists()

        self.body = {'n_patches': {}, 'percentage': {}}

        def read_map(path):
            _map = imageio.imread(str(path))

            return _map / np.iinfo(_map.dtype).max

        map_paths = prediction_path.glob('*.png')
        map_paths = [map_path for map_path in map_paths if map_path.stem not in ['foreground', 'thumbnail']]
        map_paths = [map_path for map_path in map_paths if not map_path.stem.endswith('_binarized')]

        foreground = read_map(prediction_path / 'foreground.png')

        maps = {}

        for map_path in map_paths:
            maps[map_path.stem] = read_map(map_path) * foreground

        scan_predictions = np.array([maps[k] for k in sorted(maps.keys())])

        for i, k in enumerate(sorted(maps.keys())):
            binarized_map = np.zeros(scan_predictions.shape[1:])
            binarized_map[(np.argmax(scan_predictions, axis=0) == i) & (foreground == 1.0)] = 1.0

            self.body['n_patches'][k] = int(np.sum(binarized_map))

        total_n_patches = np.sum([v for k, v in self.body['n_patches'].items() if k != 'BG'])

        for k in sorted(maps.keys()):
            if k == 'BG':
                continue

            self.body['percentage'][k] = np.round(self.body['n_patches'][k] / total_n_patches * 100, 4)

    def get(self):
        return self.body
