import numpy as np
import requests
import time
import xmltodict
import xml.etree.ElementTree as ET

from diaglib import config
from diaglib.utils import get_ndp_credentials
from io import BytesIO
from PIL import Image


def sign_in():
    credentials = get_ndp_credentials()

    username = credentials['username']
    password = credentials['password']

    response = requests.get(config.NDP_BASE_API_URL + 'signin', data={'username': username, 'password': password})
    status = response.status_code

    assert status == 200

    body = xmltodict.parse(response.text)
    sessionid = body['signin']['sessionid']
    userid = body['signin']['userid']

    return sessionid, userid


def call(base_url, function_name, data=None):
    if (not hasattr(call, 'session')) or (time.time() - call.session.get('timestamp', 0) > config.NDP_SESSION_TIMEOUT_SECONDS):
        sessionid, userid = sign_in()

        call.session = {'sessionid': sessionid, 'userid': userid, 'timestamp': time.time()}
    else:
        call.session['timestamp'] = time.time()

    if data is None:
        data = {}
    else:
        data = data.copy()

    data['sessionid'] = call.session['sessionid']
    data['userid'] = call.session['userid']

    return requests.get(base_url + str(function_name), data=data, stream=True)


def call_api(function_name, data=None):
    return call(config.NDP_BASE_API_URL, function_name, data)


def call_image_server(function_name, data=None):
    return call(config.NDP_BASE_IMAGE_SERVER_URL, function_name, data)


def find_scan_id(scan_name):
    search_results = ET.fromstring(call_api('search', {'for': scan_name}).text.encode('latin-1')).findall('object')

    if len(search_results) == 0:
        raise ValueError('Error searching for a scan with a name "%s": '
                         'unable to find a scan with a given name.' % scan_name)
    elif len(search_results) > 1:
        raise ValueError('Error searching for a scan with a name "%s": '
                         'search query returned multiple results for a given name.' % scan_name)

    return search_results[0].find('id').text


def fetch_annotations(scan_name=None, scan_id=None):
    assert (scan_name is None) != (scan_id is None)

    if scan_id is None:
        scan_id = find_scan_id(scan_name)

    response = call_api('getchildren', {'parentid': scan_id}).text.encode('latin-1')
    xml = ET.fromstring(response)
    annotations = xml.findall('./object/content/ndpviewstate')

    return annotations


def download_annotations(output_file=None, scan_name=None, scan_id=None):
    assert (scan_name is None) != (scan_id is None)

    if scan_id is None:
        scan_id = find_scan_id(scan_name)

    if output_file is None:
        output_file = '%s.ndpa' % scan_name

    response = call_api('getchildren', {'parentid': scan_id}).text.encode('latin-1')

    with open(output_file, 'wb') as f:
        f.write(response)


def fetch_region(output_image_size, centre_coordinate, source_roi_size, scan_name=None, scan_id=None):
    assert (scan_name is None) != (scan_id is None)

    if scan_id is None:
        scan_id = find_scan_id(scan_name)

    response = call_image_server('getregion', {
        'objectid': scan_id,
        'width': output_image_size[0],
        'height': output_image_size[1],
        'x': centre_coordinate[0],
        'y': centre_coordinate[1],
        'sourcewidth': source_roi_size[0],
        'sourceheight': source_roi_size[1]
    })

    return np.array(Image.open(BytesIO(response.content)))


def fetch_metadata(scan_name=None, scan_id=None):
    assert (scan_name is None) != (scan_id is None)

    if scan_id is None:
        scan_id = find_scan_id(scan_name)

    image_info = call_image_server('getimageinfo', {'objectid': scan_id}).text
    xml = ET.fromstring(image_info)

    dimensions = [int(xml.find('./pixeldimensions/%s' % attribute).text) for attribute in ['width', 'height']]
    offset = [int(xml.find('./physicalbounds/%s' % attribute).text) for attribute in ['x', 'y']]
    physical_bounds = [int(xml.find('./physicalbounds/%s' % attribute).text) for attribute in ['width', 'height']]
    mpp = [x / (y * 1000) for (x, y) in zip(physical_bounds, dimensions)]
    lens = float(xml.find('./metadata[@name="ndpImage.Lens"]').text)

    return dimensions, offset, mpp, lens
