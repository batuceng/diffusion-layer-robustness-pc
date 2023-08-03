import torch
import os
import glob
import h5py
import numpy as np
from copy import copy
from torch.utils.data import Dataset
import random


class ModelNet40Attack(Dataset):
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError
        self.filename = os.path.basename(path)
        self.dataname, self.modelname, self.taskname, \
            self.attackname, epsilonkeyword, self.eps = self.filename.split('_')
        self.eps = self.eps.rsplit('.pt')
        self.path = path
        self.transform = False
        self.datalist = self.load_data(self.path)
    
    def load_data(self, path):
        data = torch.load(path)
        return data
    
    def __getitem__(self, idx):
        data = self.datalist[idx]
        return data

    def __len__(self):
        return len(self.datalist)