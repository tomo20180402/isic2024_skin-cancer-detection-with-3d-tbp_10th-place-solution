from logging import Logger, getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
from scripts.config import Config as cfg


def create_logger(fname: str = cfg.LOG_NAME) -> Logger:
    """logger
    """
    log_file = ("{}.txt".format(fname))

    logger = getLogger(fname)
    logger.setLevel(DEBUG)

    if logger.handlers:
        return logger

    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger