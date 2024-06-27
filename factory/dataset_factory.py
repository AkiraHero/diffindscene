from dataset import *


class DatasetFactory:
    singleton_dataset = None
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(data_config):
        class_name = data_config['dataset_class']
        all_classes = DatasetBase.__subclasses__()
        for cls in all_classes:
            if cls.__name__ == class_name:
                return cls(data_config['config_file']['expanded'])
        raise TypeError(f'no class named \'{class_name}\' found in dataset folder')

    @classmethod
    def get_singleton_dataset(cls, data_config=None):
        if data_config is None:
            return cls.singleton_dataset
        if cls.singleton_dataset is None:
            cls.singleton_dataset = cls.get_dataset(data_config)
        return cls.singleton_dataset
