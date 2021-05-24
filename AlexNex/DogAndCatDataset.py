import os
from torch.utils.data import Dataset

class DogAndCatDataset(Dataset):
    def __init__(self, path):
        self.path = path
