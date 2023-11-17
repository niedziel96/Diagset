import logging
import shapely.ops

from diaglib import config
from shapely.geometry import MultiPolygon
from shapely.geometry.polygon import Polygon


def extract_polygons(annotations, tissue_tag):
    """Extract polygon dictionary from the given iterable object of XML annotations based on the given tissue tag."""
    if tissue_tag not in config.TISSUE_TAGS:
        raise ValueError('Tissue tag "%s" was not recognized.' % tissue_tag)

    polygons = {label: [] for label in config.EXTRACTED_LABELS[tissue_tag]}

    for annotation in annotations:
        label = annotation.find('title').text

        if label not in config.EXTRACTED_LABELS[tissue_tag]:
            if label not in config.IGNORED_LABELS[tissue_tag]:
                logging.getLogger('diaglib').warning('Unrecognized label "%s" in the annotation file. Ignoring.' % label)

            continue

        points = []

        for point in annotation.find('annotation').find('pointlist').findall('point'):
            x, y = float(point.find('x').text), float(point.find('y').text)
            points.append((x, y))

        if len(points) < 3:
            logging.getLogger('diaglib').warning('Less than 3 coordinates extracted from a point list. Ignoring.')

            continue

        polygon = Polygon(points)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty or not polygon.is_valid:
            logging.getLogger('diaglib').warning('Polygon extracted from annotation file is invalid. Ignoring.')
        else:
            polygons[label].append(polygon)

    return polygons


def transform_polygons(polygons, dimensions, offset, mpp):
    """Rescale the coordinates of the polygons based on the given metadata."""
    def transformation_function(x, y, z=None):
        x_transformed = (((x - offset[0]) / (mpp[0] * 1000)) + dimensions[0] / 2)
        y_transformed = (((y - offset[1]) / (mpp[1] * 1000)) + dimensions[1] / 2)

        return x_transformed, y_transformed

    polygons = polygons.copy()

    for label in polygons.keys():
        for i in range(len(polygons[label])):
            polygons[label][i] = shapely.ops.transform(transformation_function, polygons[label][i])

    return polygons


def translate_tags(polygons, tissue_tag):
    """Combine different types of valid tissue labels into a single label based on config.LABEL_TRANSLATIONS."""
    polygons = polygons.copy()

    for label in config.LABEL_TRANSLATIONS[tissue_tag].keys():
        translate_into = config.LABEL_TRANSLATIONS[tissue_tag][label]

        if label in polygons.keys():
            if translate_into in polygons.keys():
                polygons[translate_into] += polygons[label]

                del polygons[label]
            else:
                polygons[translate_into] = polygons[label]

                del polygons[label]

    return polygons


def convert_into_multipolygons(polygons):
    """Convert dictionary containing list of polygons into a dictionary of multipolygons."""
    multipolygons = {}

    for label in polygons.keys():
        multipolygon = MultiPolygon(polygons[label])

        if not multipolygon.is_valid:
            multipolygon = multipolygon.buffer(0)

        if not multipolygon.is_valid:
            raise ValueError('Merged polygon with label "%s" is invalid.' % label)

        multipolygons[label] = multipolygon

    return multipolygons


def remove_background(multipolygons):
    """Remove background regions (T) overlapping different types of tissues."""
    multipolygons = multipolygons.copy()

    if 'T' in multipolygons.keys():
        for label in multipolygons.keys():
            if label == 'T':
                continue

            multipolygons['T'] = multipolygons['T'].difference(multipolygons[label])

        if not multipolygons['T'].is_valid:
            raise ValueError('Merged polygon with label "T" is invalid after subtraction.')

    return multipolygons


def prepare_multipolygons(annotations, tissue_tag, dimensions, offset, mpp):
    """Extract and process polygons, and afterwards convert them into a multipolygons, in a single step."""
    polygons = extract_polygons(annotations, tissue_tag)
    polygons = transform_polygons(polygons, dimensions, offset, mpp)
    polygons = translate_tags(polygons, tissue_tag)

    multipolygons = convert_into_multipolygons(polygons)
    multipolygons = remove_background(multipolygons)

    return multipolygons


def extract_labels(annotations):
    labels = []

    for annotation in annotations:
        labels.append(annotation.find('title').text)

    return set(labels)
