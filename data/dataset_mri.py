# Author: Charles R. Clark
# CS 6440 Spring 2024

import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image

class Mri(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df 
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        img_path, img_label = self.df.iloc[idx]
        img = read_image(img_path).float()

        if self.transform:
            img = self.transform(img)

        return img, img_label