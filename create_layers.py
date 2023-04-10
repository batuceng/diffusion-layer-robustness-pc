from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# from data import ModelNet40
#from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
#from util import cal_loss, IOStream
import sklearn.metrics as metrics
import models 
from models.dgcnn import DGCNN
from models.autoencoder import AutoEncoder
from utils.dataset import Layer1
from utils.dataset import ModelNet40
from utils.misc import seed_all

seed_all(42)
args = None

# Dataset
train_set = ModelNet40(num_points=1024, partition='train')
train_loader = DataLoader(train_set, batch_size=32, num_workers=0)

test_set = ModelNet40(num_points=1024, partition='test')
test_loader = DataLoader(test_set, batch_size=32, num_workers=0)
print("loaded")

# Load pretrained DGCNN
model =  DGCNN(mode="save_layers")
model_dict = torch.load("model_best_test.pth")["model_state"]
model_keys = list(model_dict.keys())
for key in model_keys:
    if key.startswith("model."):
        keyname = key[6:]
        model_dict[keyname] = model_dict.pop(key)
    else:
        _ = model_dict.pop(key)

model.load_state_dict(model_dict)
model.cuda()
model.eval()

# DENOISER

layer_dict = {"x1":64, "x2":64, "x3":128, "x4":256, "original":3}
layer_dim = None

class Denoiser(nn.Module):
    def __init__(self, args, layer_dim):
        super().__init__()
        assert args.denoiser_cpkt_path is not None
        ckpt = torch.load(args.denoiser_cpkt_path)
        self.diffusion_model =  AutoEncoder(ckpt['args'], layer_dim=layer_dim).cuda()
        self.diffusion_model.load_state_dict(ckpt['state_dict'])
        
    # Do the denoising
    def forward(self, x, t=5):
        x = x.transpose(1, 2)
        code = self.diffusion_model.encode(x)
        x = self.diffusion_model.denoiser(x, t, context=code)
        x = x.transpose(1, 2)
        return x

class Identity_c(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, t=5):
        return x

# denoiser = Denoiser(args=args, layer_dim=layer_dim)
denoiser_list = nn.ModuleList([
    Identity_c(),                  # Input
    Identity_c(),                       # Layer1
    Identity_c(),                  # Layer2
    Identity_c(),                  # Layer3
    Identity_c(),                  # Layer4
])

# t = 10

save_layer = True
loader = train_loader

correct_num, num_samples_test = 0,0
with torch.no_grad():
    for i, (data, labels) in enumerate(loader):
        print("Evaluating batch", i+1, "/", len(loader.dataset), "...")
        
        x, save_copy, sizes = model.forward(data.permute((0,2,1)).cuda(), denoiser=denoiser_list, t=1)
        logits = torch.argmax(x.cpu(), dim=1)
        
        correct_num += torch.sum(logits == labels.squeeze())
        num_samples_test += len(labels)
        print(correct_num, num_samples_test)
    
    