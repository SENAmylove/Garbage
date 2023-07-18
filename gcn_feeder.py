import pickle
import torch
from torch.utils.data import DataLoader

class Feeder(DataLoader):

    def __init__(self, data_path: str, train_val = 'train'):
        self.data_path = data_path
        with open(self.data_path, 'rb') as r:
            self.data = pickle.load(r)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass
