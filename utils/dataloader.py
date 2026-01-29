import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('/opt/run')
from utils import center_crop, min_max_normalization


class ThalamusDataset(Dataset):
    def __init__(self, data_path):
        super(ThalamusDataset, self).__init__()
        self.data_path = data_path
        self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = np.expand_dims(nib.load(self.data_path).get_fdata().astype(np.float32), axis=-1)
        data = data.transpose((3, 0, 1, 2))
        data = center_crop(data, crop_size=(96, 96, 96))
        data = min_max_normalization(data)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        return data_tensor


def thalamus_dataloader(data_path, shuffle=False):
    dataset = ThalamusDataset(data_path=data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle)
    return dataloader
