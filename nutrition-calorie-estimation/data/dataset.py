import torch
from torch.utils.data import Dataset
import numpy as np


class MultiModalDataset(Dataset):
    def __init__(self, cgm, textual, img, labels):
        self.cgm = cgm
        self.textual = textual
        self.img = img
        self.labels = labels

    def __len__(self):
        return len(self.cgm)

    def __getitem__(self, idx):
        cgm_seq = torch.tensor(self.cgm[idx],dtype=torch.float32)
        textual_features = torch.tensor(self.textual.iloc[idx].values,dtype=torch.float32)
        img_before_breakfast = self.img.iloc[idx]['Image Before Breakfast']
        img_before_lunch = self.img.iloc[idx]['Image Before Lunch']

        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[idx],dtype=torch.float32)
            return cgm_seq, textual_features, img_before_breakfast, img_before_lunch, label
        else:
            return cgm_seq, textual_features, img_before_breakfast, img_before_lunch