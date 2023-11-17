import base64
import flask

from diaglib import config
from diaglib.predict.reports import Report
from diaglib.serve import db
from flask import jsonify


app = flask.Flask(__name__)


@app.route('/api/v1/requests/<scan_id>/<organ>', methods=['POST'])
def create_request(scan_id, organ):
    if len(db.find_requests(scan_id=scan_id)) > 0:
        return jsonify({'error': 'Request already exists.'}), 400
    elif organ not in config.ORGAN_TO_TAG_MAPPING.keys():
        return jsonify({'error': 'Unrecognized organ type.'}), 400
    else:
        db.add_request(scan_id, organ)

        return jsonify({'message': 'Request successfully created.'}), 200


@app.route('/api/v1/requests/<scan_id>', methods=['GET'])
def get_request(scan_id):
    return jsonify({'status': db.get_request_status(scan_id)}), 200


@app.route('/api/v1/predictions/maps/<scan_id>', methods=['GET'])
def get_probability_maps(scan_id):
    if db.get_request_status(scan_id) != 'finished':
        return jsonify({'error': 'Request unfinished.'}), 400

    request = db.find_requests(scan_id=scan_id)[0]

    prediction_path = config.PREDICTIONS_PATH / request.model_name / scan_id

    if not prediction_path.exists():
        return jsonify({'error': 'Request marked as finished but predictions were not found.'}), 400

    map_paths = prediction_path.glob('*.png')
    map_paths = [map_path for map_path in map_paths if map_path.stem not in ['foreground', 'thumbnail']]
    map_paths = [map_path for map_path in map_paths if not map_path.stem.endswith('_binarized')]

    result = {}

    for map_path in map_paths:
        if config.SERVE_BINARY_PREDICTIONS:
            map_path_string = str(map_path).replace('.png', '_binarized.png')
        else:
            map_path_string = str(map_path)

        with open(map_path_string, 'rb') as f:
            result[map_path.stem] = base64.b64encode(f.read()).decode('utf-8')

    return jsonify(result), 200


@app.route('/api/v1/predictions/reports/<scan_id>', methods=['GET'])
def get_reports(scan_id):
    try:
        report = Report(scan_id)

        return jsonify(report.get()), 200
    except ValueError:
        return jsonify({'error': 'Report unavailable.'}), 400


@app.route('/api/v1/predictions/diagnosis/<scan_id>', methods=['GET'])
def get_diagnosis(scan_id):
    pass


@app.route('/api/v1/revisions/<scan_id>/<organ>', methods=['POST'])
def create_revision(scan_id, organ):
    if organ not in config.ORGAN_TO_TAG_MAPPING.keys():
        return jsonify({'error': 'Unrecognized organ type.'}), 400
    else:
        pass

        return jsonify({'message': 'Revision successfully created.'}), 200


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run(host=config.REQUESTS_HOST, port=config.REQUESTS_PORT)
