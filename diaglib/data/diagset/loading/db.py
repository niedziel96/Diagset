import pyodbc
import re
import xml.etree.ElementTree as ET

from diaglib import config
from diaglib.utils import get_db_credentials


def connect():
    """Connect to the database and return connection object."""
    credentials = get_db_credentials()

    uid = credentials['uid']
    pwd = credentials['pwd']

    connection = pyodbc.connect(
        'DRIVER={%s};'
        'SERVER=%s;'
        'DATABASE=%s;'
        'UID=%s;'
        'PWD=%s' % (config.DB_DRIVER, config.DB_SERVER_ADDRESS, config.DB_DATABASE_NAME, uid, pwd)
    )

    return connection


def fetch_scans_with_annotations(connection=None):
    """Find all scans with at least one annotation, return their IDs and names."""
    if connection is None:
        connection = connect()

    cursor = connection.cursor()

    rows = cursor.execute('SELECT DISTINCT s.ID, s.Name '
                          'FROM objObjects o, objObjects s, objObjectRelations r '
                          'WHERE o.ObjectType = \'annotation\' AND r.ChildID = o.ID AND s.ID = r.ParentID '
                          'ORDER BY s.ID').fetchall()

    for i in range(len(rows)):
        rows[i][0] = rows[i][0].rstrip()

    return rows


def fetch_all_annotations(connection=None):
    """Find all annotations, return scan names, scan IDs, IDs of the annotations and annotations content."""
    if connection is None:
        connection = connect()

    cursor = connection.cursor()

    rows = cursor.execute('SELECT DISTINCT s.Name, s.ID, o.ID, o.Content '
                          'FROM objObjects o, objObjects s, objObjectRelations r '
                          'WHERE o.ObjectType = \'annotation\' AND r.ChildID = o.ID AND s.ID = r.ParentID '
                          'ORDER BY s.ID, o.ID').fetchall()

    for i in range(len(rows)):
        rows[i][1] = rows[i][1].rstrip()
        rows[i][2] = rows[i][2].rstrip()

    return rows


def fetch_annotations_for_scan(scan_id, connection=None):
    """Find annotations for the scan with a given ID, return annotation ID and content."""
    if connection is None:
        connection = connect()

    cursor = connection.cursor()

    rows = cursor.execute('SELECT DISTINCT o.ID, o.Content '
                          'FROM objObjects o, objObjectRelations r '
                          'WHERE o.ObjectType = \'annotation\' AND r.ChildID = o.ID AND r.ParentID = \'%s\' '
                          'ORDER BY o.ID' % scan_id).fetchall()

    for i in range(len(rows)):
        rows[i][0] = rows[i][0].rstrip()

    return rows


def fetch_xml_annotations_for_scan(scan_id, connection=None):
    """Find annotations for the scan with a given ID, return a list of XML objects."""
    annotations = fetch_annotations_for_scan(scan_id, connection)

    return [ET.fromstring(annotation[1]) for annotation in annotations]


def fetch_scan_id(scan_name, connection=None):
    """Find scan ID for a given scan name."""
    if connection is None:
        connection = connect()

    cursor = connection.cursor()

    rows = cursor.execute('SELECT DISTINCT o.ID '
                          'FROM objObjects o '
                          'WHERE o.ObjectType = \'slide\' AND o.Name = \'%s\' '
                          'ORDER BY o.ID' % scan_name).fetchall()

    if len(rows) == 0:
        raise ValueError('Error searching for a scan with a name "%s": '
                         'unable to find a scan with a given name.' % scan_name)
    elif len(rows) > 1:
        raise ValueError('Error searching for a scan with a name "%s": '
                         'search query returned multiple results for a given name.' % scan_name)

    scan_id = rows[0][0].rstrip()

    return scan_id


def fetch_scan_name(scan_id, connection=None):
    """Find scan name for a given scan ID."""
    if connection is None:
        connection = connect()

    cursor = connection.cursor()

    rows = cursor.execute('SELECT DISTINCT o.Name '
                          'FROM objObjects o '
                          'WHERE o.ObjectType = \'slide\' AND o.ID = \'%s\' '
                          'ORDER BY o.Name' % scan_id).fetchall()

    if len(rows) == 0:
        raise ValueError('Error searching for a scan with an ID "%s": '
                         'unable to find a scan with a given name.' % scan_id)
    elif len(rows) > 1:
        raise ValueError('Error searching for a scan with an ID "%s": '
                         'search query returned multiple results for a given name.' % scan_id)

    scan_name = rows[0][0].rstrip()

    return scan_name


def fetch_all_scans(connection=None):
    """Find all scans, return their IDs and names."""
    if connection is None:
        connection = connect()

    cursor = connection.cursor()

    rows = cursor.execute('SELECT DISTINCT o.ID, o.Name '
                          'FROM objObjects o '
                          'WHERE o.ObjectType = \'slide\' '
                          'ORDER BY o.ID').fetchall()

    for i in range(len(rows)):
        rows[i][0] = rows[i][0].rstrip()

    return rows


def fetch_scans_with_tissue_tag_and_label(tissue_tag, label, connection=None):
    """Find all scans matching the pattern '.*-\s*{tissue_tag}\s*-\s*{label}\s*-.*', return their IDs and names."""
    pattern = r'.*-\s*%s\s*-\s*%s\s*-.*' % (tissue_tag, label)
    scans = fetch_all_scans(connection)
    selected_scans = [scan for scan in scans if re.match(pattern, scan[1])]

    return selected_scans
