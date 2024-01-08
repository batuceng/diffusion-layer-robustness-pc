import os
import argparse
import torch
import numpy as np
import json
import sys

import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

sys.path.append("../../..")

from util.misc import IOStream, seed_all
from dataset.modelnet40attack import ModelNet40Attack
from models import DGCNN_cls, PointNet2_cls, PointNet_cls, PointMLP_cls, CurveNet_cls, PCT_cls

from opt_defense import defend_point_cloud

import warnings
warnings.filterwarnings("ignore")


# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--save-path', type=str, default='../../../data_defended')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp-logs', type=str, default='experiment_logs')
parser.add_argument('--test_size', type=int, default=128)

# Datasets and loaders
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--scale-mode', type=str, default='shape_unit')

parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'pointnet2', 'dgcnn', 'curvenet', 'pct', 'pointmlp'],
                    help='Model to use, [pointnet, pointnet2, dgcnn, curvenet, pct, pointmlp]')
parser.add_argument('--num-points', type=int, default=1024,
                        help='num of points to use, changes nothing currently')
parser.add_argument('--no-cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--attack', type=str, default="cw",
                    choices=['add', 'cw', 'drop', 'knn', 'pgd', 'pgdl2'],)

parser.add_argument('--attacked_data_folder', type=str, default="../../../data_attacked",
                    help='path to attacked data folder, including json')
args = parser.parse_args()
seed_all(args.seed)

model_dict = {
    'curvenet': CurveNet_cls, #0.938412
    'pct':      PCT_cls, #0.931524
    'pointmlp': PointMLP_cls, #0.940032
    'dgcnn':    DGCNN_cls, #0.928687
    'pointnet2':PointNet2_cls, #0.911264
    'pointnet': PointNet_cls, #0.896677
}

# Get attacked data paths as dictionary
with open(os.path.join(args.attacked_data_folder,"attacked_data_list.json")) as file:
    attacked_data_list = json.load(file)


# Get path to adversarial data & info json from dict
attacked_data_path = "../../../" + attacked_data_list[args.attack][args.model] + ".pt"
attacked_json_path = "../../../" + attacked_data_list[args.attack][args.model] + ".json"
args.attacked_data_path, args.attacked_json_path = attacked_data_path, attacked_json_path

test_loader = DataLoader(ModelNet40Attack(path=attacked_data_path), num_workers=4,
                            batch_size=args.batch_size, shuffle=False, drop_last=False)


device = torch.device("cuda" if args.device else "cpu")
torch.manual_seed(args.seed)

# Function for forward pass
@torch.no_grad()
def get_logits(data, model, shift, scale, device):
    input = (data.to(device) * scale.view(-1,1,1).to(device) + shift.to(device)).detach().permute(0,2,1)
    model = model.to(device)
    logits = model(input)
    return logits.clone().detach().cpu()

@torch.no_grad()
def get_stats(labels, logits):
    preds = logits.max(dim=1)[1]
    acc = accuracy_score(labels.squeeze().numpy(), preds.squeeze().numpy())
    loss = F.cross_entropy(logits.to(torch.float64), labels.squeeze().to(torch.long))
    return acc, loss.item()

# Load classifer model
if args.model == 'pointnet2':
    model = PointNet2_cls().to(device)
elif args.model == 'pointnet':
    model = PointNet_cls().to(device)
elif args.model == 'dgcnn':
    model = DGCNN_cls().to(device)
elif args.model == 'curvenet':
    model = CurveNet_cls().to(device)
elif args.model == 'pct':
    model = PCT_cls().to(device)
elif args.model == 'pointmlp':
    model = PointMLP_cls().to(device)
else:
    raise Exception("Not implemented")
model.eval()

# Load defense model
defense_model = defend_point_cloud

num_classes = 40    
clean_preds_list = torch.tensor([]).view(0,num_classes)
attacked_preds_list = torch.tensor([]).view(0,num_classes)
denoised_preds_list = torch.tensor([]).view(0,num_classes)
defended_preds_list = torch.tensor([]).view(0,num_classes)
true_label_list = torch.tensor([]).view(0,1)

stop_iter = args.test_size // args.batch_size
# stop_iter = 5

defended_data_list = []
    
for i, batch in enumerate(test_loader):
    print(f"batch {i}/{len(test_loader)}")
    
    # Stop at specified batch number
    if i == stop_iter: break
    
    data = batch['pointcloud']
    data_attack = batch['attacked']
    shift = batch['shift']
    scale = batch['scale']
    label = batch["cate"]
    
    # Prediction on original data
    clean_logits = get_logits(data, model, shift, scale, device)
    clean_preds_list = torch.cat((clean_preds_list, clean_logits), dim=0)
    
    # Prediction on attacked data
    attacked_logits = get_logits(data_attack, model, shift, scale, device)
    attacked_preds_list = torch.cat((attacked_preds_list, attacked_logits), dim=0)
    
    # Defensive Prediction on original data
    denoised_data = defense_model(data.clone().cpu().numpy())
    denoised_data = torch.from_numpy(denoised_data).to(args.device)
    denoised_logits = get_logits(denoised_data, model, shift, scale, device)
    denoised_preds_list = torch.cat((denoised_preds_list, denoised_logits), dim=0)
        
    # Defensive Prediction on attacked data
    defended_data = defense_model(data_attack.clone().cpu().numpy())
    defended_data_list.append(defended_data)
    defended_data = torch.from_numpy(defended_data).to(args.device)
    defended_logits = get_logits(defended_data, model, shift, scale, device)
    defended_preds_list = torch.cat((defended_preds_list, defended_logits), dim=0)
    
    # Label
    true_label_list = torch.cat((true_label_list, label), dim=0)

# Print out Results
clean_acc, clean_loss = get_stats(true_label_list, clean_preds_list)
attacked_acc, attacked_loss = get_stats(true_label_list, attacked_preds_list)
denoised_acc, denoised_loss = get_stats(true_label_list, denoised_preds_list)
defended_acc, defended_loss = get_stats(true_label_list, defended_preds_list)
geo_mean = np.sqrt((denoised_acc**2 + defended_acc**2)/2)

defended_data_list = torch.cat(defended_data_list)

result_dict = {}
result_dict["Clean_loss"] = clean_loss
result_dict["Attacked_loss"] = attacked_loss
result_dict["Denoised_loss"] = denoised_loss
result_dict["Defended_loss"] = defended_loss

result_dict["Clean_acc"] = clean_acc
result_dict["Attacked_acc"] = attacked_acc
result_dict["Denoised_acc"] = denoised_acc
result_dict["Defended_acc"] = defended_acc
result_dict["Geo-Mean_acc"] = geo_mean

print("Clean acc:", clean_acc, "Clean loss:", clean_loss)
print("Attacked acc:", attacked_acc, "Attacked loss:", attacked_loss)
print("Denoised acc:", denoised_acc, "Denoised loss:", denoised_loss)
print("Defended acc:", defended_acc, "Defended loss:", defended_loss)
print("Geo-Mean Acc:", geo_mean)

savedir = os.path.join(*[args.save_path, "ifedefense", args.attack, args.model]) if args.save_path is not None else None
filename = os.path.join(savedir, datetime.today().strftime('DEFENSE_%Y_%m_%d__%H_%M_%S')) if savedir is not None else None

os.makedirs(savedir, exist_ok=True)

args_dict = vars(args)
with open(args.attacked_json_path, "r") as file:
    args_dict["attack_args"] = json.load(file)
    file.close()

args_dict |= {"result_path":filename}
args_dict |= result_dict

with open(filename+'.json', 'w') as fp:
    json.dump(args_dict, fp, indent=4)
    fp.close()

torch.save(defended_data_list, f=filename+'.pt')