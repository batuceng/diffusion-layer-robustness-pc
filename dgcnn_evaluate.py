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
from utils.dataset import ModelNet40, ModelNet40C, ModelNet40Attack
from utils.misc import seed_all
from attack import FGM, IFGM, MIFGM, PGD, Identity_Attack
from attack import CWPerturb
from attack import CrossEntropyAdvLoss, LogitsAdvLoss
from attack import ClipPointsL2, L2Dist


# Arguments
parser = argparse.ArgumentParser()
# Denoiser Model arguments
# parser.add_argument('--latent_dim', type=int, default=256)
# parser.add_argument('--num_steps', type=int, default=200)
# parser.add_argument('--beta_1', type=float, default=1e-4)
# parser.add_argument('--beta_T', type=float, default=0.05)
# parser.add_argument('--sched_mode', type=str, default='linear')
# parser.add_argument('--flexibility', type=float, default=0.0)
# parser.add_argument('--residual', type=eval, default=True, choices=[True, False])



#### TO DO LIST
"""
    -Add adversarial attacks
    -Add training epoch loss
    -Normalize ChamferDistance by num of dimension
    -Solve batch size error
    -Cleanup code
    -Move standardization to dataloader
"""


parser.add_argument('--corruption', type=str, default='gaussian', choices=["background", "cutout", "density", "density_inc", 
                                                                         "distortion", "distortion_rbf", "distortion_rbf_inv", 
                                                                         "gaussian", "impulse", "lidar", "occlusion", "rotation", 
                                                                         "shear", "uniform", "upsampling"])
parser.add_argument('--severity', type=int, default=4, choices=[1,2,3,4,5])
parser.add_argument('--denoiser_cpkt_path', type=str, 
                    default="logs_x1/AE_2023_04_13__02_18_24/ckpt_19.849722_181000.pt")
# Training
# parser.add_argument('--input_type', type=str, default='original')  # "x1", "x2", "x3", "x4", "original"
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=42)

# Attack arguments
parser.add_argument('--attack_type', type=str, default="cw_perturb")
parser.add_argument('--budget', type=float, default=0.08,
                        help='FGM attack budget')
parser.add_argument('--num_iter', type=int, default=500, # 50 for pgd, 500 for cw
                    help='IFGM iterate step')
parser.add_argument('--mu', type=float, default=1.,
                    help='momentum factor for MIFGM attack')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
parser.add_argument('--kappa', type=float, default=0.,
                        help='min margin in logits adv loss')
parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in CW optimization')
parser.add_argument('--binary_step', type=int, default=10, metavar='N',
                        help='Binary search step')

args = parser.parse_args()
seed_all(args.seed)

# corruption_list = ["background", "cutout", "density", "density_inc", "distortion", "distortion_rbf", "distortion_rbf_inv", 
#                       "gaussian", "impulse", "lidar", "occlusion", "rotation", "shear", "uniform", "upsampling"]
# severity_list = [1,2,3,4,5]

# Dataset
# test_dset = ModelNet40(num_points=1024, partition='test')

# corruption, severity = ("density", 5)

# test_dset = ModelNet40C(split="test", test_data_path="data/modelnet40_c",corruption=args.corruption,severity=args.severity)
test_dset = ModelNet40Attack(data_root="data/ModelNet40attack", num_points=args.num_points)
# test_dset = ModelNet40(num_points=args.num_points, partition="test")
test_loader = DataLoader(test_dset, batch_size=args.val_batch_size, num_workers=0)

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

# ATTACK

if args.adv_func == 'logits':
    adv_func = LogitsAdvLoss(kappa=args.kappa)
else:
    adv_func = CrossEntropyAdvLoss()
clip_func = ClipPointsL2(budget=args.budget)
dist_func = L2Dist()

delta = args.budget
args.budget = args.budget * np.sqrt(args.num_points * 3)  # \delta * \sqrt(N * d)
args.num_iter = int(args.num_iter)
args.step_size = args.budget / float(args.num_iter)

# args.attack_type = "None"
# 4 variants of FGM
if args.attack_type.lower() == 'fgm':
    attacker = FGM(model, adv_func=adv_func,
                    budget=args.budget, dist_metric='l2')
elif args.attack_type.lower() == 'ifgm':
    attacker = IFGM(model, adv_func=adv_func,
                    clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                    num_iter=args.num_iter, dist_metric='l2')
elif args.attack_type.lower() == 'mifgm':
    attacker = MIFGM(model, adv_func=adv_func,
                        clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                        num_iter=args.num_iter, mu=args.mu, dist_metric='l2')
elif args.attack_type.lower() == 'pgd':
    attacker = PGD(model, adv_func=adv_func,
                    clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                    num_iter=args.num_iter, dist_metric='l2')
elif args.attack_type.lower() == "cw_perturb":
    attacker = CWPerturb(model, adv_func, dist_func,
                         attack_lr=args.attack_lr,
                         init_weight=10., max_weight=80.,
                         binary_step=args.binary_step,
                         num_iter=args.num_iter)
elif args.attack_type.lower() == "none":
    attacker = Identity_Attack()

# DENOISER


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
    
layer_dict = {"original":(0,3), "x1":(1,64), "x2":(2,64), "x3":(3,128), "x4":(4,256)}

assert args.denoiser_cpkt_path is not None
ckpt = torch.load(args.denoiser_cpkt_path)
layer_name = ckpt['args'].input_type
layer_no, layer_dim = layer_dict[layer_name]

denoiser_list = nn.ModuleList([
    Identity_c(),                  # Input
    Identity_c(),                  # Layer1
    Identity_c(),                  # Layer2
    Identity_c(),                  # Layer3
    Identity_c(),                  # Layer4
])
model_paths = ["/ari/users/bcengiz/pc_rob_file/diffusion-point-cloud-main/logs_x1/AE_2023_04_13__02_18_24/ckpt_19.849722_181000.pt",
               "/ari/users/bcengiz/pc_rob_file/diffusion-point-cloud-main/logs_x2/AE_2023_04_13__02_22_07/ckpt_37.653294_181000.pt",
               "/ari/users/bcengiz/pc_rob_file/diffusion-point-cloud-main/logs_x3/AE_2023_04_13__02_28_17/ckpt_113.578575_179000.pt",
               "/ari/users/bcengiz/pc_rob_file/diffusion-point-cloud-main/logs_x4/AE_2023_04_13__02_28_19/ckpt_392.097565_48000.pt"]

for i in range(1, 5):
    ckpt = torch.load(model_paths[i-1])
    layer_name = ckpt['args'].input_type
    layer_no, layer_dim = layer_dict[layer_name]
    denoiser = AutoEncoder(ckpt['args'], layer_dim=layer_dim)
    denoiser.load_state_dict(ckpt['state_dict'])
    denoiser.cuda()
    denoiser.eval()
    denoiser_list[i] = denoiser
    print(f"layer_no:{layer_no}, layer_name:{ckpt['args'].input_type}, ckpt_path:{model_paths[i-1]}")
    
# denoiser = AutoEncoder(ckpt['args'], layer_dim=layer_dim)
# denoiser.load_state_dict(ckpt['state_dict'])
# denoiser.cuda()
# denoiser.eval()
# denoiser_list = nn.ModuleList([
#     Identity_c(),                  # Input
#     Identity_c(),                  # Layer1
#     Identity_c(),                  # Layer2
#     Identity_c(),                  # Layer3
#     Identity_c(),                  # Layer4
# ])
# denoiser_list[layer_no] = denoiser
# print(f"layer_no:{layer_no}, layer_name:{ckpt['args'].input_type}, corruption:{args.corruption}, severity:{args.severity} ckpt_path:{args.denoiser_cpkt_path}")

T = 8
results = {}
for t in range(T, T+1):
    # t +=1
    # EVALUATE
    num_batches_test = len(test_loader)
    num_samples_test = 0
    correct_num = 0
    
    for i, (data, labels, target_labels) in enumerate(test_loader):
        print("Batch", i, "/", len(test_loader))
        best_dist, best_pc, success_num = attacker.attack(data, target_labels)
        best_pc = torch.tensor(best_pc)
        with torch.no_grad():
            # print("Evaluating batch", i+1, "/", num_batches_test, "...")
            logits = model.forward(best_pc.permute((0,2,1)).cuda(), denoiser=denoiser_list, t=t, layer_name=layer_name)
            logits = torch.argmax(logits.cpu(), dim=1)
        
        correct_num += torch.sum(logits == labels.squeeze())
        num_samples_test += len(labels)
        
        total_acc = correct_num/num_samples_test
        print(f"Test accuracy : {total_acc:.4f}, denoising_t:{t}") 
    results[t] = correct_num/num_samples_test # Save result for t step denoising
    
print(results)

