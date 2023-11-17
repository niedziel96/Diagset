import openslide
import xml.etree.ElementTree as ET

from diaglib import config


def read_annotations(name):
    """Read annotation file for a scan with the given name from local .NDPA file."""
    annotation_file_name = '%s.ndpa' % name
    annotation_file_path = config.DIAGSET_ANNOTATIONS_PATH / annotation_file_name

    if not annotation_file_path.exists():
        raise ValueError('No annotation file was found under the path "%s".' % annotation_file_path)

    xml = ET.parse(str(annotation_file_path)).getroot()
    annotations = xml.findall('./object/content/ndpviewstate')

    return annotations


def read_slide(name):
    """Read OpenSlide object for a scan with the given name from local .NDPI file."""
    scan_file_name = '%s.ndpi' % name
    scan_file_path = config.DIAGSET_SCANS_PATH / scan_file_name

    if not scan_file_path.exists():
        raise ValueError('No scan file was found under the path "%s".' % scan_file_path)

    return openslide.OpenSlide(str(scan_file_path))


def extract_metadata(slide):
    """Extract scan metadata from the given OpenSlide object."""
    dimensions = list(slide.level_dimensions[0])
    offset = [int(slide.properties['hamamatsu.%sOffsetFromSlideCentre' % coordinate]) for coordinate in ['X', 'Y']]
    mpp = [float(slide.properties['openslide.mpp-%s' % coordinate]) for coordinate in ['x', 'y']]
    lens = slide.properties['hamamatsu.SourceLens']

    return dimensions, offset, mpp, lens
