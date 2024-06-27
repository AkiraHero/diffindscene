import logging
from utils.logger.basic_logger import BasicLogger


class DummyLogger(BasicLogger):
    logger = None

    def __init__(self, config):
        return

    def log_config(self, config):
        return

    def log_data(self, data_name, data_content, add_to_tensorboard=False):
        return

    def _add_to_pickle(self, status, data_name, data_content):
        return

    def _log_scalar(self, status, data_name, data_content, add_to_tensorboard=False):
        return

    def log_model_params(self, *args, **argv):
        return

    def register_status_hook(self, fn):
        return

    @classmethod
    def get_logger(cls, config=None):
        if cls.logger is not None:
            if config is not None:
                logging.warning("input config for logger will be ignored")
            return cls.logger
        if config is None:
            raise ValueError("config must be set")
        else:
            cls.logger = DummyLogger(config)
            return cls.logger

    def copy_screen_log(self, file_path):
        return
