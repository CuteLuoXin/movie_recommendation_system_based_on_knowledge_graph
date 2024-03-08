import sys
import os
import logging


class Logger(object):

    def __init__(self, filename):

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s: %(message)s',
                                      datefmt='%Y-%m-%d %H-%M-%S')

        # write into file
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # show on console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)


    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()