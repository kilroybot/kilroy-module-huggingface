from contextlib import contextmanager

from transformers import logging


@contextmanager
def suppress_huggingface_logging(level=logging.ERROR):
    current_level = logging.get_verbosity()
    logging.set_verbosity(level)
    yield
    logging.set_verbosity(current_level)
