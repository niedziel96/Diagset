import datetime

from diaglib import config
from pony import orm


REQUEST_STATUSES = ['unreceived', 'pending', 'processing', 'finished', 'failed']


class DB:
    __instance = None
    __bound = False
    __mapped = False

    @staticmethod
    def get():
        if DB.__instance is None:
            DB.__instance = orm.Database()

        return DB.__instance

    @staticmethod
    def bind_and_map():
        if not DB.__bound:
            config.REQUESTS_ROOT_PATH.mkdir(parents=True, exist_ok=True)

            DB.__instance.bind(provider='sqlite', filename=str(config.REQUESTS_DB_PATH), create_db=True)
            DB.__bound = True

        if not DB.__mapped:
            DB.__instance.generate_mapping(create_tables=True)
            DB.__mapped = True


class Request(DB.get().Entity):
    scan_id = orm.Required(str, unique=True)
    organ = orm.Required(str)
    status = orm.Required(str)
    tissue_tag = orm.Optional(str)
    magnification = orm.Optional(str)
    patch_size = orm.Optional(int)
    model_name = orm.Optional(str)
    model_class = orm.Optional(str)
    created_at = orm.Required(str)
    pulled_at = orm.Optional(str)
    finalized_at = orm.Optional(str)


DB.bind_and_map()


@orm.db_session
def add_request(scan_id, organ):
    if organ not in config.ORGAN_TO_TAG_MAPPING.keys():
        raise ValueError('Unrecognized organ "%s".' % organ)

    return Request(
        scan_id=scan_id,
        organ=organ,
        tissue_tag=config.ORGAN_TO_TAG_MAPPING[organ],
        status='pending',
        created_at=str(datetime.datetime.now())
    )


@orm.db_session
def find_requests(scan_id=None, organ=None, tissue_tag=None, status=None, magnification=None,
                  patch_size=None, model_name=None, model_class=None):
    requests = Request.select()

    if scan_id is not None:
        requests = orm.select(r for r in requests if r.scan_id == scan_id)

    if organ is not None:
        requests = orm.select(r for r in requests if r.organ == organ)

    if tissue_tag is not None:
        requests = orm.select(r for r in requests if r.tissue_tag == tissue_tag)

    if status is not None:
        requests = orm.select(r for r in requests if r.status == status)

    if magnification is not None:
        requests = orm.select(r for r in requests if r.magnification == magnification)

    if patch_size is not None:
        requests = orm.select(r for r in requests if r.patch_size == patch_size)

    if model_name is not None:
        requests = orm.select(r for r in requests if r.model_name == model_name)

    if model_class is not None:
        requests = orm.select(r for r in requests if r.model_class == model_class)

    return requests[:]


@orm.db_session
def pull_request():
    requests = find_requests(status='pending')

    if len(requests) > 0:
        request = requests[0]

        request.status = 'processing'
        request.pulled_at = str(datetime.datetime.now())

        return request
    else:
        return None


@orm.db_session
def finalize_request(request_id, status, model_name=None, model_class=None, magnification=None, patch_size=None):
    assert status in ['finished', 'failed']

    if status != 'failed':
        assert model_name is not None

    request = Request[request_id]

    request.status = status

    if model_name is not None:
        request.model_name = model_name

    if model_class is not None:
        request.model_class = model_class

    if magnification is not None:
        request.magnification = magnification

    if patch_size is not None:
        request.patch_size = patch_size

    request.finalized_at = str(datetime.datetime.now())


@orm.db_session
def get_request_status(scan_id):
    requests = find_requests(scan_id=scan_id)

    if len(requests) == 0:
        return 'unreceived'
    else:
        return requests[0].status


@orm.db_session
def clear_processed_requests(time_passed_in_seconds=config.REQUESTS_PROCESSING_TIMEOUT_SECONDS):
    requests = find_requests(status='processing')
    current_time = datetime.datetime.now()
    n_cleared_requests = 0

    for r in requests:
        delta = current_time - datetime.datetime.strptime(r.pulled_at, '%Y-%m-%d %H:%M:%S.%f')

        if delta.seconds >= time_passed_in_seconds:
            r.status = 'pending'
            r.pulled_at = ''
            n_cleared_requests += 1

    return n_cleared_requests
