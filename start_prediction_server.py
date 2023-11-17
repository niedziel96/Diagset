import argparse

from diaglib import config
from diaglib.serve import runner, server
from threading import Thread


parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-buffer_size', type=int, default=128)
parser.add_argument('-gpu', type=int, default=0)

args = parser.parse_args()

Thread(target=runner.run, daemon=True, args=(
    args.batch_size, args.buffer_size, args.gpu
)).start()

server.app.run(debug=False, host=config.REQUESTS_HOST, port=config.REQUESTS_PORT)
