def get_nested_path(root_path, tissue_tag, magnification):
    nested_path = root_path / tissue_tag / ('%dx' % magnification)

    return nested_path
