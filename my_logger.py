import logging
import pathlib
from datetime import datetime

import utils


@utils.singleton
class MyLogger(object):
    logger = None
    def __init__(self):
        MyLogger.logger = None
        self.__get_logger()

    def __get_logger(self):
        MyLogger.logger = logging.getLogger('')

        # create a logger in the form [time] [filename] [line number] [log level] [message]
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s')

        # create a file handler
        if not pathlib.Path('./logs').exists():
            pathlib.Path('./logs').mkdir(parents=True, exist_ok=True)
        time_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        fh = logging.FileHandler(f'logs/log-{time_str}.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        MyLogger.logger.addHandler(fh)

        # create a console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        MyLogger.logger.addHandler(ch)

        # set the log level
        MyLogger.logger.setLevel(logging.DEBUG)

    @classmethod
    def get_logger(cls):
        if not MyLogger.logger:
            MyLogger()
        return MyLogger.logger
