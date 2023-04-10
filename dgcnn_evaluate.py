from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
#from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
#from util import cal_loss, IOStream
import sklearn.metrics as metrics
import models 
from models.dgcnn import DGCNN
from models.autoencoder import AutoEncoder
from utils.dataset import Layer1
from utils.misc import seed_all



# Arguments
parser = argparse.ArgumentParser()
# Denoiser Model arguments
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--denoiser_cpkt_path', type=str, 
                    default="/home/robust/Desktop/diffusion-point-cloud-main/logs_ae/AE_2023_04_07__15_39_28/ckpt_21.095808_61000.pt")
# Training
parser.add_argument('--input_type', type=str, default='x1')  # "x1", "x2", "x3", "x4", "original"
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
seed_all(args.seed)


# Dataset
test_dset = Layer1("original", split="test", standardize=False)
test_loader = DataLoader(test_dset, batch_size=32, num_workers=0)

# Load pretrained DGCNN
model =  DGCNN()
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
layer_dim = layer_dict[args.input_type]

# class Denoiser(nn.Module):
#     def __init__(self, args, layer_dim):
#         super().__init__()
#         assert args.denoiser_cpkt_path is not None
#         ckpt = torch.load(args.denoiser_cpkt_path)
#         self.diffusion_model =  AutoEncoder(ckpt['args'], layer_dim=layer_dim).cuda()
#         self.diffusion_model.load_state_dict(ckpt['state_dict'])
        
#     # Do the denoising
#     def forward(self, x, t=5):
#         x = x.transpose(1, 2)
#         code = self.diffusion_model.encode(x)
#         x = self.diffusion_model.denoiser(x, t, context=code)
#         x = x.transpose(1, 2)
#         return x

class Identity_c(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, t=5, layer_name="original"):
        return x
    def denoise_layer(self, x, t=5, layer_name="original"):
        return x

denoiser = AutoEncoder(args=args, layer_dim=layer_dim)
denoiser_list = nn.ModuleList([
    Identity_c(),                  # Input
    denoiser,                       # Layer1
    Identity_c(),                  # Layer2
    Identity_c(),                  # Layer3
    Identity_c(),                  # Layer4
])

# t = 10
results = {}
for t in range(10):
    t +=1
    # EVALUATE
    num_batches_test = len(test_loader)
    num_samples_test = 0
    correct_num = 0

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            print("Evaluating batch", i+1, "/", num_batches_test, "...")
            
            logits = model.forward(data.cuda(), denoiser=denoiser_list, t=t, layer_name=args.input_type)
            logits = torch.argmax(logits.cpu(), dim=1)
            
            correct_num += torch.sum(logits == labels)
            num_samples_test += len(labels)
        
        total_acc = correct_num/num_samples_test
        print(f"Test accuracy : {total_acc:.4f}, denoising_t:{t}") 
    results[t] = total_acc # Save result for t step denoising
    
print(results)