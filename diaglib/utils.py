import diaglib.config
import numpy as np


def shuffle_combined(inputs, outputs):
    indices = list(range(len(inputs)))

    np.random.shuffle(indices)

    return inputs[indices], outputs[indices]


def get_ndp_credentials():
    if diaglib.config.NDP_USERNAME is None:
        raise ValueError('NDP_USERNAME was not set in the config file.')

    if diaglib.config.NDP_PASSWORD is None:
        raise ValueError('NDP_PASSWORD was not set in the config file.')

    return {'username': diaglib.config.NDP_USERNAME, 'password': diaglib.config.NDP_PASSWORD}


def get_db_credentials():
    if diaglib.config.DB_UID is None:
        raise ValueError('DB_UID was not set in the config file.')

    if diaglib.config.DB_PWD is None:
        raise ValueError('DB_PWD was not set in the config file.')

    return {'uid': diaglib.config.DB_UID, 'pwd': diaglib.config.DB_PWD}
