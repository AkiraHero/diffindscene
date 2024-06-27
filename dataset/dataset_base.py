from torch.utils.data import Dataset


class DatasetBase(Dataset):
    def __init__(self):
        super(DatasetBase, self).__init__()
        self.mode ='train'

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_data_loader(self, distributed=False):
        raise NotImplementedError
    
    def set_mode(self, mode):
        assert mode in ['train', 'val', 'test']
        self.mode = mode

    def get_mode(self):
        return self.mode
