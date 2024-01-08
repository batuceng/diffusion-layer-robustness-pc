"""
Test denoising model on pre-saved PGD attacked data.
"""


import os
import argparse
import torch
import numpy as np
import json

import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from util.misc import IOStream, seed_all
from dataset.modelnet40attack import ModelNet40Attack
from models.autoencoder import AutoEncoder
from models import DGCNN_cls, PointNet2_cls, PointNet_cls, PointMLP_cls, CurveNet_cls, PCT_cls
from models.denoiser import Identity, Layer_Denoiser, Multiple_Layer_Denoiser
from sklearn.metrics import accuracy_score
from util.evaluation_metrics import EMD_CD

import warnings
warnings.filterwarnings("ignore")


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t_list', type=str, default='0,0,0,0,0')

parser.add_argument('--save-path', type=str, default='./dist_results_maxvals')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp-logs', type=str, default='experiment_logs')
parser.add_argument('--test_size', type=int, default=np.inf)

# Datasets and loaders
parser.add_argument('--dataset', type=str, default='modelnet40attack', metavar='N',
                    choices=['modelnet40attack'])
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

parser.add_argument('--attacked_data_folder', type=str, default="./data_attacked",
                    help='path to attacked data folder, including json')
parser.add_argument('--pretrained_weights_folder', type=str, default="./uhem_trained",
                    help='path to pretrained diffusion models folder, including json')

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

# Get pratrained diffusion model paths as dictionary
with open(os.path.join(args.pretrained_weights_folder,"model_path_dict.json")) as file:
    model_path_dict = json.load(file)
# Get attacked data paths as dictionary
with open(os.path.join(args.attacked_data_folder,"attacked_data_list.json")) as file:
    attacked_data_list = json.load(file)

# Get path to adversarial data & info json from dict
attaked_data_path = attacked_data_list[args.attack][args.model] + ".pt"
attaked_json_path = attacked_data_list[args.attack][args.model] + ".json"
args.attaked_data_path, args.attaked_json_path = attaked_data_path, attaked_json_path

# Decode t_list and get path to pretrained diffusion models from dict
t_list = [int(i.strip()) for i in args.t_list.split(",")]
pretrained_denoiser_paths = model_path_dict[args.model]
args.t_list, args.pretrained_denoiser_paths = t_list, pretrained_denoiser_paths

# Function for forward pass
@torch.no_grad()
def get_logits(data, model, shift, scale, device, denoiser=Identity()):
    input = (data.to(device) * scale.view(-1,1,1).to(device) + shift.to(device)).detach().permute(0,2,1)
    model = model.to(device)
    logits, layers = model.forward_denoised(input, denoiser)
    return logits.clone().detach().cpu(), [l.clone().detach().cpu() for l in layers if l!=None]

# Function to calc Accuracy & loss
@torch.no_grad()
def get_stats(labels, logits):
    preds = logits.max(dim=1)[1]
    acc = accuracy_score(labels.squeeze().numpy(), preds.squeeze().numpy())
    loss = F.cross_entropy(logits.to(torch.float64), labels.squeeze().to(torch.long))
    return acc, loss.item()

# Function to calc distances such as L2, Linf, Chamfer
@torch.no_grad()
def get_layer_distances(layerA, layerB, metrics):
    # ObjA Clean Pc, ObjB Denoised PC
    def get_distances(objA, objB, metrics=metrics):
        l2dist, linfdist, cd = torch.tensor([0], dtype=objA.dtype), torch.tensor([0], dtype=objA.dtype), 0 
        # L2
        if "L2" in metrics:
            l2dist = torch.norm((objA-objB), p=2, dim=0)
        # Linf
        if "Linf" in metrics:
            linfdist = torch.norm((objA-objB), torch.inf, dim=0)
        # Chamfer
        if "CD" in metrics:
            emd_cd = EMD_CD(objB, objA, batch_size=args.batch_size, verbose=False)
            cd, emd = emd_cd['MMD-CD'].item(), emd_cd['MMD-EMD'].item()
        return l2dist, linfdist, cd
    
    layer_dict = {}
    for i, (la, lb) in enumerate(zip(layerA, layerB)):
        l2, linf, cd = get_distances(layerA[la], layerB[lb])
        layer_dict[i] = [l2.mean().item(), linf.mean().item(), cd]
    return layer_dict
    
# Convert np.inf & np.nan to str before json dump
def convert_numpy_objects(dict_to_convert):
    new = {}
    for k, v in dict_to_convert.items():
        if isinstance(v, dict):
            new[k] = convert_numpy_objects(v)
        else:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                new[k] = str(v)
            elif isinstance(v, np.ndarray):
                new[k] = v.tolist()
            else:
                new[k] = v
    return new

def test(args, io):
    test_loader = DataLoader(ModelNet40Attack(path=attaked_data_path), num_workers=4,
                            batch_size=args.batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    # Classification Model
    if args.model == 'pointnet2':
        model = PointNet2_cls().to(device)
        input_dim_dict = {0:3, 1:128, 2:256, 3:1024, 4:None}
        input_num_dict = {0:1024, 1:512, 2:128, 3:1, 4:None}
    elif args.model == 'pointnet':
        model = PointNet_cls().to(device)
        input_dim_dict = {0:3, 1:64, 2:128, 3:1024, 4:None}
        input_num_dict = {0:1024, 1:1024, 2:1024, 3:1024, 4:None}
    elif args.model == 'dgcnn':
        model = DGCNN_cls().to(device)
        input_dim_dict = {0:3, 1:64, 2:64, 3:128, 4:256}
        input_num_dict = {0:1024, 1:1024, 2:1024, 3:1024, 4:1024}
    elif args.model == 'curvenet':
        model = CurveNet_cls().to(device)
        input_dim_dict = {0:3, 1:64, 2:128, 3:256, 4:512}
        input_num_dict = {0:1024, 1:1024, 2:1024, 3:256, 4:64}
    elif args.model == 'pct':
        model = PCT_cls().to(device)
        input_dim_dict = {0:3, 1:128, 2:256, 3:256, 4:256}
        input_num_dict = {0:1024, 1:512, 2:256, 3:256, 4:256}
    elif args.model == 'pointmlp':
        model = PointMLP_cls().to(device)
        input_dim_dict = {0:3, 1:64, 2:128, 3:256, 4:512}
        input_num_dict = {0:1024, 1:1024, 2:512, 3:256, 4:128}
    else:
        raise Exception("Not implemented")
    model.eval()
    
    attacked_num_dict = input_num_dict.copy()
    if args.attack=="drop": attacked_num_dict[0]=824
    if args.attack=="add": attacked_num_dict[0]=1224
    
    # Autoencoders
    ae_list = []
    for i in range(5):
        lay_path = pretrained_denoiser_paths[f"layer{i}"]
        if lay_path == "Identity":
            ae = Identity()
        else:
            ckpt = torch.load(lay_path)
            ae = AutoEncoder(ckpt['args']).to(args.device)
            ae.load_state_dict(ckpt['state_dict'])
            ae.eval()
        ae_list.append(ae)
    denoiser = Multiple_Layer_Denoiser(models=ae_list, t_list=t_list)
    
    # Create empty lists to keep all predictions
    num_classes = 40    
    clean_preds_list = torch.tensor([]).view(0,num_classes)
    attacked_preds_list = torch.tensor([]).view(0,num_classes)
    denoised_preds_list = torch.tensor([]).view(0,num_classes)
    defended_preds_list = torch.tensor([]).view(0,num_classes)
    true_label_list = torch.tensor([]).view(0,1)
    
    
    # Dummy forward pass to get layer shapes
    dummy_batch = next(iter(test_loader))
    _, clean_layers = get_logits(dummy_batch['pointcloud'], model, dummy_batch['shift'], dummy_batch['scale'], device)
    _, attacked_layers = get_logits(dummy_batch['attacked'], model, dummy_batch['shift'], dummy_batch['scale'], device)
    # print([lay.shape for lay in clean_layers])
    
    # Create empty tensors to keep layer data(for distance measurements)
    layer_len = len(clean_layers)
    clean_layers_dict = {i:torch.tensor([]).view(0, layer_data.shape[2], layer_data.shape[1]) for i,layer_data in enumerate(clean_layers)}
    attacked_layers_dict = {i:torch.tensor([]).view(0, layer_data.shape[2], layer_data.shape[1]) for i,layer_data in enumerate(attacked_layers)}
    denoised_layers_dict = {i:torch.tensor([]).view(0, layer_data.shape[2], layer_data.shape[1]) for i,layer_data in enumerate(clean_layers)}
    defended_layers_dict = {i:torch.tensor([]).view(0, layer_data.shape[2], layer_data.shape[1]) for i,layer_data in enumerate(attacked_layers)}
    
    # Number of point clouds to test the attack
    stop_iter = args.test_size // args.batch_size
    # stop_iter = 5
    
    for i, batch in enumerate(test_loader):
        # print(f"batch {i}/{len(test_loader)}")
        
        # Stop at specified batch number
        if i == stop_iter: break
        
        data = batch['pointcloud']
        data_attack = batch['attacked']
        shift = batch['shift']
        scale = batch['scale']
        label = batch["cate"]
        
        # Prediction on original data
        clean_logits, clean_layers = get_logits(data, model, shift, scale, device)
        clean_preds_list = torch.cat((clean_preds_list, clean_logits), dim=0)
        for i in range(layer_len):
            # print(clean_layers[i].transpose(1, 2).shape)
            # print(clean_layers_dict[i].shape)
            clean_layers_dict[i] = torch.cat((clean_layers_dict[i], clean_layers[i].transpose(1, 2)), dim=0)
        
        # Prediction on attacked data
        attacked_logits, attacked_layers = get_logits(data_attack, model, shift, scale, device)
        attacked_preds_list = torch.cat((attacked_preds_list, attacked_logits), dim=0)
        for i in range(layer_len):
            attacked_layers_dict[i] = torch.cat((attacked_layers_dict[i], attacked_layers[i].transpose(1, 2)), dim=0)
        
        # Defensive Prediction on original data
        denoised_logits, denoised_layers = get_logits(data, model, shift, scale, device, denoiser=denoiser)
        denoised_preds_list = torch.cat((denoised_preds_list, denoised_logits), dim=0)
        for i in range(layer_len):
            denoised_layers_dict[i] = torch.cat((denoised_layers_dict[i], denoised_layers[i].transpose(1, 2)), dim=0)
            
        # Defensive Prediction on attacked data
        defended_logits, defended_layers = get_logits(data_attack, model, shift, scale, device, denoiser=denoiser)
        defended_preds_list = torch.cat((defended_preds_list, defended_logits), dim=0)
        for i in range(layer_len):
            defended_layers_dict[i] = torch.cat((defended_layers_dict[i], defended_layers[i].transpose(1, 2)), dim=0)
            
        # Label
        true_label_list = torch.cat((true_label_list, label), dim=0)
    
    metrics = None
    if args.model=="pointnet": metrics = ["L2", "Linf"] # Skip CD for pointnet, it takes too long
    if args.attack in ["add","drop"]: metrics = [] # Cannot compute metrics due to different num of points
    else: metrics = ["L2", "Linf", "CD"] # Otherwise, get all metrics
    
    attacked_dist = get_layer_distances(clean_layers_dict, attacked_layers_dict, metrics=metrics)
    denoised_dist = get_layer_distances(clean_layers_dict, denoised_layers_dict, metrics=metrics)
    defended_dist = get_layer_distances(clean_layers_dict, defended_layers_dict, metrics=metrics)
    
    # Print out Results
    clean_acc, clean_loss = get_stats(true_label_list, clean_preds_list)
    attacked_acc, attacked_loss = get_stats(true_label_list, attacked_preds_list)
    denoised_acc, denoised_loss = get_stats(true_label_list, denoised_preds_list)
    defended_acc, defended_loss = get_stats(true_label_list, defended_preds_list)
    
    # JSON Dump Dictionary
    result_dict = {}
    
    # Layer Distances
    sub_dict = {}
    for i in range(layer_len):
        sub_dict[f"Layer {i} distances"] = {"L2": attacked_dist[i][0], "Linf":attacked_dist[i][1], "CD": attacked_dist[i][2]}
    result_dict[f"Clean-Attacked distances"] = sub_dict
    
    sub_dict = {}
    for i in range(layer_len):
        sub_dict[f"Layer {i} distances"] = {"L2": denoised_dist[i][0], "Linf":denoised_dist[i][1], "CD": denoised_dist[i][2]}
    result_dict[f"Clean-Denoised distances"] = sub_dict
        
    sub_dict = {}
    for i in range(layer_len):
        sub_dict[f"Layer {i} distances"] = {"L2": defended_dist[i][0], "Linf":defended_dist[i][1], "CD": defended_dist[i][2]}
    result_dict[f"Clean-Defended distances"] = sub_dict
        
    print(f"\n T_List: {t_list} \n")
    print("Clean Acc:", clean_acc)
    print("Clean Loss:", clean_loss)
    # print("-"*30)
    
    # print("Attacked dists")
    # for i in range(layer_len):
    #     print(f"Layer {i}:", "L2:", attacked_dist[i][0], "Linf:", attacked_dist[i][1], "CD:", attacked_dist[i][2])
    # print("Attacked Acc:", attacked_acc)
    # print("Attacked Loss:", attacked_loss)
    # print("-"*30)
    
    # print("Denoised dists")
    # for i in range(layer_len):
    #     print(f"Layer {i}:", "L2:", denoised_dist[i][0], "Linf:", denoised_dist[i][1], "CD:", denoised_dist[i][2])
    # print("Denoised Acc:", denoised_acc)
    # print("Denoised Loss:", denoised_loss)
    # print("-"*30)
    
    # print("Defended dists")
    # for i in range(layer_len):
    #     print(f"Layer {i}:", "L2:", defended_dist[i][0], "Linf:", defended_dist[i][1], "CD:", defended_dist[i][2])
    print("Defended Acc:", defended_acc)
    print("Defended Loss:", defended_loss)
    print()
    
    geo_mean = np.sqrt((denoised_acc**2 + defended_acc**2)/2)
    print("Geo-Mean Acc:", geo_mean)
    print("-"*30)
    
    # Losses
    result_dict["Clean_loss"] = clean_loss
    result_dict["Attacked_loss"] = attacked_loss
    result_dict["Denoised_loss"] = denoised_loss
    result_dict["Defended_loss"] = defended_loss
    # Accuracies
    result_dict["Clean_acc"] = clean_acc
    result_dict["Attacked_acc"] = attacked_acc
    result_dict["Denoised_acc"] = denoised_acc
    result_dict["Defended_acc"] = defended_acc
    result_dict["Geo-Mean_acc"] = geo_mean

    # Save Predictions for each element
    result_dict["Preds"] = {
        "True_labels": true_label_list.squeeze().numpy(),
        "Clean_preds": clean_preds_list.max(dim=1)[1].squeeze().numpy(),
        "Attacked_preds": attacked_preds_list.max(dim=1)[1].squeeze().numpy(),
        "Denoised_preds": denoised_preds_list.max(dim=1)[1].squeeze().numpy(),
        "Defended_preds": defended_preds_list.max(dim=1)[1].squeeze().numpy(),
    }
    
    savedir = os.path.join(*[args.save_path, args.attack, args.model, "t__" + str("_".join([str(i) for i in t_list]))]) if args.save_path is not None else None
    filename = os.path.join(savedir, datetime.today().strftime('RESULT_%Y_%m_%d__%H_%M_%S')) if savedir is not None else None
    
    os.makedirs(savedir, exist_ok=True)
    
    args_dict = vars(args)
    with open(args.attaked_json_path, "r") as file:
        args_dict["attack_args"] = json.load(file)
    
    args_dict |= {"result_path":filename}
    args_dict |= result_dict
    with open(filename+'.json', 'w') as fp:
        json.dump(convert_numpy_objects(args_dict), fp, indent=4)
        fp.close()

io = IOStream('outputs/' + args.exp_name + '/run.log')

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

test(args, io)
