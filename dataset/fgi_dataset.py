import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision import transforms


class FGIDataset(Dataset):
    def __init__(self, mat_path, transform=None):
        self.f = h5py.File(mat_path, 'r')
        self.transform = transform
        self.images = self.f['Data']['data']
        self.labels = self.f['Data']['label']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = np.transpose(self.images[idx], (1, 2, 0))  # [448, 448, 3]
        label = self.labels[idx]
        gaze = label[:2]
        head = label[2:4]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "gaze": torch.tensor(gaze).float(),
            "headpose": torch.tensor(head).float()
        }
