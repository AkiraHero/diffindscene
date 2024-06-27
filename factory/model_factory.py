from model import *
from utils.config.Configuration import Configuration

class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_config):
        class_name, paras = Configuration.find_dict_node(model_config, 'model_class')
        all_classes = ModelBase.__subclasses__()
        for cls in all_classes:
            if cls.__name__ == class_name:
                return cls(model_config['config_file']['expanded']) # todo not perfect
        raise TypeError(f'no class named \'{class_name}\' found in model folder')


