import io
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import base64
from PIL import Image

def get_logger(name):
        logger = logging.getLogger(name)
        logger.handlers = []
        log_formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d ' + log_context +
                                          ' %(levelname)s %(filename)s:%(lineno)d] %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_formatter)

        if os.environ.get('TORNASOLE_LOG_ALL_TO_STDOUT', default='TRUE').lower() == 'false':
            stderr_handler = logging.StreamHandler(sys.stderr)
            min_level = logging.DEBUG
            # lets through all levels less than ERROR
            stdout_handler.addFilter(MaxLevelFilter(logging.ERROR))
            stdout_handler.setLevel(min_level)

            stderr_handler.setLevel(max(min_level, logging.ERROR))
            stderr_handler.setFormatter(log_formatter)
            logger.addHandler(stderr_handler)

        logger.addHandler(stdout_handler)

def analysis( fname ):
    with open( fname ) as f:
        captures = json.loads(f.read())