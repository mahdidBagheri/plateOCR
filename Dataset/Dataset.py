import torch
from torch.utils.data import Dataset
from Dataset.DataSynthesis import synthesize_license
from torchvision.transforms import v2
from Config.DatasetConfig import alphabet_length
import numpy as np

class PlateDataset(Dataset):
    def __init__(self, size):
        super(PlateDataset, self).__init__()

        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert item <= self.size

        word, plate  = synthesize_license()

        lable = self.create_lable(word)

        transforms = v2.Compose([ v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), v2.ToTensor()])
        plate_tensor = transforms(plate)

        return plate_tensor, lable

    def create_lable(self, word):
        v = torch.zeros((len(word),alphabet_length))
        for i, c in enumerate(word):
            if c == " ":
                j = 0
            elif c.isdigit():
                j = int(c) + 1
            else:
                j = ord(c) - 55 + 1
            v[i, j] = 1
        return v

