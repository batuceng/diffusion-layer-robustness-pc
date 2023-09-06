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

model_path_dict = {
    "dgcnn": [
        "uhem_trained/dgcnn/layer0/2023_08_26__15_13_30/ckpt_0.003807_1818000.pt",
        "uhem_trained/dgcnn/layer1/2023_08_26__15_06_48/ckpt_0.570095_1998000.pt",
        "uhem_trained/dgcnn/layer2/2023_08_26__15_07_21/ckpt_1.605012_1344000.pt",
        "uhem_trained/dgcnn/layer3/2023_08_26__15_07_43/ckpt_3.458023_1854000.pt",
        "uhem_trained/dgcnn/layer4/2023_08_26__15_08_06/ckpt_18.761648_1347000.pt",
    ],
    "curvenet": [
        "uhem_trained/dgcnn/layer0/2023_08_26__15_13_30/ckpt_0.003807_1818000.pt",
        "uhem_trained/curvenet/layer1/2023_08_15__15_43_52/ckpt_0.254451_1707000.pt",
        "uhem_trained/curvenet/layer2/2023_08_15__15_44_44/ckpt_1.777029_1662000.pt",
        "uhem_trained/curvenet/layer3/2023_08_12__20_45_22/ckpt_4.668281_1104000.pt",
        "uhem_trained/curvenet/layer4/2023_08_12__20_43_02/ckpt_5.987106_945000.pt",
    ],
    "pct": [
        "uhem_trained/dgcnn/layer0/2023_08_26__15_13_30/ckpt_0.003807_1818000.pt",
        "uhem_trained/pct/layer1/2023_08_15__15_35_59/ckpt_10.050566_9000.pt",
        "uhem_trained/pct/layer2/2023_08_15__15_40_13/ckpt_76.067963_1518000.pt",
        "uhem_trained/pct/layer3/2023_08_15__15_41_46/ckpt_100.240242_24000.pt",
        "uhem_trained/pct/layer4/2023_08_15__15_42_20/ckpt_181.535049_1740000.pt",
    ],
    "pointmlp": [
        "uhem_trained/dgcnn/layer0/2023_08_26__15_13_30/ckpt_0.003807_1818000.pt",
        "uhem_trained/pointmlp/layer1/2023_08_12__19_42_21/ckpt_0.004289_1239000.pt",
        "uhem_trained/pointmlp/layer2/2023_08_12__19_41_08/ckpt_1.437761_1293000.pt",
        "uhem_trained/pointmlp/layer3/2023_08_12__19_40_21/ckpt_3.167697_1251000.pt",
        "uhem_trained/pointmlp/layer4/2023_08_12__19_40_09/ckpt_13.268216_15000.pt",
    ],
    "pointnet2": [
        "uhem_trained/dgcnn/layer0/2023_08_26__15_13_30/ckpt_0.003807_1818000.pt",
        "uhem_trained/pointnet2/layer1/2023_08_12__20_34_27/ckpt_11.080982_1089000.pt"
        "uhem_trained/pointnet2/layer1/2023_08_15__16_06_41/ckpt_10.943207_2283000.pt",
        "uhem_trained/pointnet2/layer2/2023_08_12__20_35_16/ckpt_15.662710_1131000.pt",
        "uhem_trained/pointnet2/layer2/2023_08_15__16_07_51/ckpt_15.582665_2652000.pt",
        "uhem_trained/pointnet2/layer3/2023_08_12__20_35_49/ckpt_14666061332873216.000000_27000.pt",
    ],
    "pointnet": [
        "uhem_trained/dgcnn/layer0/2023_08_26__15_13_30/ckpt_0.003807_1818000.pt",
        "uhem_trained/pointnet/layer1/2023_08_12__20_19_11/ckpt_0.354332_2862000.pt",
        "uhem_trained/pointnet/layer2/2023_08_12__20_20_10/ckpt_1.809803_2706000.pt",
        "uhem_trained/pointnet/layer3/2023_08_12__20_20_57/ckpt_53.780865_1026000.pt",
    ],
}

attacked_data_list = [
"data_attacked/add/curvenet/ATK_2023_09_04__14_31_22.pt",
"data_attacked/add/curvenet/ATK_2023_09_04__20_35_40.pt",
"data_attacked/add/dgcnn/ATK_2023_09_04__14_25_51.pt",
"data_attacked/add/pct/ATK_2023_09_04__14_27_41.pt",
"data_attacked/add/pointnet2/ATK_2023_09_04__14_23_18.pt",
"data_attacked/add/pointnet/ATK_2023_09_04__14_23_00.pt",
"data_attacked/cw/curvenet/ATK_2023_09_04__17_01_08.pt",
"data_attacked/cw/curvenet/ATK_2023_09_04__21_25_56.pt",
"data_attacked/cw/dgcnn/ATK_2023_09_04__16_27_26.pt",
"data_attacked/cw/pct/ATK_2023_09_04__16_30_39.pt",
"data_attacked/cw/pointnet2/ATK_2023_09_04__16_22_23.pt",
"data_attacked/cw/pointnet/ATK_2023_09_04__16_21_49.pt",
"data_attacked/drop/curvenet/ATK_2023_08_18__16_45_05.pt",
"data_attacked/drop/dgcnn/ATK_2023_08_18__16_39_59.pt",
"data_attacked/drop/pct/ATK_2023_08_18__16_41_23.pt",
"data_attacked/drop/pointmlp/ATK_2023_08_22__14_51_27.pt",
"data_attacked/drop/pointnet2/ATK_2023_08_18__16_37_32.pt",
"data_attacked/drop/pointnet/ATK_2023_08_18__16_37_15.pt",
"data_attacked/knn/curvenet/ATK_2023_09_04__15_50_20.pt",
"data_attacked/knn/curvenet/ATK_2023_09_04__20_45_54.pt",
"data_attacked/knn/dgcnn/ATK_2023_09_04__14_59_46.pt",
"data_attacked/knn/pct/ATK_2023_09_04__15_03_21.pt",
"data_attacked/knn/pointnet2/ATK_2023_09_04__14_41_10.pt",
"data_attacked/knn/pointnet/ATK_2023_09_04__14_38_05.pt",
"data_attacked/pgd/curvenet/ATK_2023_08_18__16_12_20.pt",
"data_attacked/pgd/dgcnn/ATK_2023_08_18__15_58_47.pt",
"data_attacked/pgdl2/curvenet/ATK_2023_08_18__16_30_20.pt",
"data_attacked/pgdl2/dgcnn/ATK_2023_08_18__16_26_09.pt",
"data_attacked/pgdl2/pointmlp/ATK_2023_08_22__15_45_25.pt",
"data_attacked/pgdl2/pointnet2/ATK_2023_08_18__16_19_48.pt",
"data_attacked/pgdl2/pointnet/ATK_2023_08_18__16_19_14.pt",
"data_attacked/pgd/pct/ATK_2023_08_18__16_02_45.pt",
"data_attacked/pgd/pointmlp/ATK_2023_08_22__15_03_43.pt",
"data_attacked/pgd/pointnet2/ATK_2023_08_18__15_52_25.pt",
"data_attacked/pgd/pointnet/ATK_2023_08_18__15_51_52.pt"
]


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t_list', type=str, default='5,5,5,5,5')

parser.add_argument('--save-path', type=str, default='./dist_results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp-logs', type=str, default='experiment_logs')
parser.add_argument('--test_size', type=int, default=np.inf)

# Datasets and loaders
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--scale-mode', type=str, default='shape_unit')

# DGCNN arguments
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'pointnet2', 'dgcnn', 'curvenet', 'pct', 'pointmlp'],
                    help='Model to use, [pointnet, pointnet2, dgcnn, curvenet, pct, pointmlp]')
parser.add_argument('--dataset', type=str, default='modelnet40attack', metavar='N',
                    choices=['modelnet40attack'])
parser.add_argument('--num-points', type=int, default=1024,
                        help='num of points to use')
parser.add_argument('--no-cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--attack', type=str, default="cw")


model_dict = {
    'curvenet': CurveNet_cls, #0.938412
    'pct':      PCT_cls, #0.931524
    'pointmlp': PointMLP_cls, #0.940032
    'dgcnn':    DGCNN_cls, #0.928687
    'pointnet2':PointNet2_cls, #0.911264
    'pointnet': PointNet_cls, #0.896677
}

args = parser.parse_args()
seed_all(args.seed)

args.data_attacked = list(filter(lambda x: args.model in x and args.attack in x, attacked_data_list))[0]
args.json_path = args.data_attacked.split(".pt")[0] + ".json"

args.t_list = [int(i.strip()) for i in args.t_list.split(",")]

args.available_models = model_path_dict[args.model]

@torch.no_grad()
def get_logits(data, model, shift, scale, device, denoiser=Identity()):
    input = (data.to(device) * scale.view(-1,1,1).to(device) + shift.to(device)).detach().permute(0,2,1)
    model = model.to(device)
    logits, layers = model.forward_denoised(input, denoiser)
    return logits.clone().detach().cpu(), [l.clone().detach().cpu() for l in layers]

@torch.no_grad()
def get_stats(labels, logits):
    preds = logits.max(dim=1)[1]
    acc = accuracy_score(labels.squeeze().numpy(), preds.squeeze().numpy())
    loss = F.cross_entropy(logits.to(torch.float64), labels.squeeze().to(torch.long))
    return acc, loss.item()

@torch.no_grad()
def get_layer_distances(layerA, layerB):
    # ObjA Clean Pc, ObjB Denoised PC
    def get_distances(objA, objB):
        # L2
        l2dist = torch.norm((objA-objB), p=2, dim=0)
        # Linf
        linfdist = torch.norm((objA-objB), torch.inf, dim=0)
        # Chamfer
        metrics = EMD_CD(objB, objA, batch_size=args.batch_size)
        cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
        
        return l2dist, linfdist, cd
    
    layer_dict = {}
    for i, (la, lb) in enumerate(zip(layerA, layerB)):
        l2, linf, cd = get_distances(layerA[la], layerB[lb])
        layer_dict[i] = [l2.mean().item(), linf.mean().item(), cd]

    return layer_dict
    

def test(args, io):

    test_loader = DataLoader(ModelNet40Attack(path=args.data_attacked),
                            batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    # Classification Model
    if args.model == 'pointnet2':
        model = PointNet2_cls().to(device)
        input_dim_dict = {0:3, 1:128, 2:256, 3:1024, 4:None}
        input_num_dict = {0:1024, 1:512, 2:128, 3:128, 4:None}
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
        input_num_dict = {0:1024, 1:1024, 2:256, 3:128, 4:64}
    else:
        raise Exception("Not implemented")
    model.eval()
    
    # Autoencoders
    ae_list = []
    for lay in args.available_models:
        if lay == "Identity":
            ae = Identity()
        else:
            ckpt = torch.load(lay)
            ae = AutoEncoder(ckpt['args']).to(args.device)
            ae.load_state_dict(ckpt['state_dict'])
            ae.eval()
        ae_list.append(ae)
    denoiser = Multiple_Layer_Denoiser(models=ae_list, t_list=args.t_list)
    
    num_classes = 40    
    clean_preds_list = torch.tensor([]).view(0,num_classes)
    attacked_preds_list = torch.tensor([]).view(0,num_classes)
    denoised_preds_list = torch.tensor([]).view(0,num_classes)
    defended_preds_list = torch.tensor([]).view(0,num_classes)
    true_label_list = torch.tensor([]).view(0,1)
    
    layer_len = len(list(filter(lambda x: x is not None, input_dim_dict.values())))
    clean_layers_dict = {i:torch.tensor([]).view(0, input_num_dict[i], input_dim_dict[i]) for i in range(layer_len)}
    attacked_layers_dict = {i:torch.tensor([]).view(0, input_num_dict[i], input_dim_dict[i]) for i in range(layer_len)}
    denoised_layers_dict = {i:torch.tensor([]).view(0, input_num_dict[i], input_dim_dict[i]) for i in range(layer_len)}
    defended_layers_dict = {i:torch.tensor([]).view(0, input_num_dict[i], input_dim_dict[i]) for i in range(layer_len)}
    
    # clean_layers_list = torch.tensor([]).view(0, *list(input_dim_dict.values()))
    # attacked_layers_list = torch.tensor([]).view(0, *list(input_dim_dict.values()))
    # denoised_layers_list = torch.tensor([]).view(0, *list(input_dim_dict.values()))
    # defended_layers_list = torch.tensor([]).view(0, *list(input_dim_dict.values()))
    
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
        
        # print(data.shape)
        # Prediction on original data
        clean_logits, clean_layers = get_logits(data, model, shift, scale, device)
        clean_preds_list = torch.cat((clean_preds_list, clean_logits), dim=0)
        for i in range(layer_len):
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
    
    attacked_dist = get_layer_distances(clean_layers_dict, attacked_layers_dict)
    denoised_dist = get_layer_distances(clean_layers_dict, denoised_layers_dict)
    defended_dist = get_layer_distances(clean_layers_dict, defended_layers_dict)
    
    # Print out Results
    clean_acc, clean_loss = get_stats(true_label_list, clean_preds_list)
    attacked_acc, attacked_loss = get_stats(true_label_list, attacked_preds_list)
    denoised_acc, denoised_loss = get_stats(true_label_list, denoised_preds_list)
    defended_acc, defended_loss = get_stats(true_label_list, defended_preds_list)
    
    
    result_dict = {}
    
    print(f"\n T_List: {args.t_list} \n")
    print("Clean Acc:", clean_acc)
    print("Clean Loss:", clean_loss)
    print("-"*30)
    
    print("Attacked dists")
    for i in input_dim_dict:
        print(f"Layer {i}:", "L2:", attacked_dist[i][0], "Linf:", attacked_dist[i][1], "CD:", attacked_dist[i][2])
        result_dict[f"Layer {i} distances"] = {"L2": attacked_dist[i][0], "Linf":attacked_dist[i][1], "CD": attacked_dist[i][2]}
    print("Attacked Acc:", attacked_acc)
    print("Attacked Loss:", attacked_loss)
    print("-"*30)
    
    print("Denoised dists")
    for i in input_dim_dict:
        print(f"Layer {i}:", "L2:", denoised_dist[i][0], "Linf:", denoised_dist[i][1], "CD:", denoised_dist[i][2])
        result_dict[f"Layer {i} distances"] = {"L2": denoised_dist[i][0], "Linf":denoised_dist[i][1], "CD": denoised_dist[i][2]}
    print("Denoised Acc:", denoised_acc)
    print("Denoised Loss:", denoised_loss)
    print("-"*30)
    
    print("Defended dists")
    for i in input_dim_dict:
        print(f"Layer {i}:", "L2:", defended_dist[i][0], "Linf:", defended_dist[i][1], "CD:", defended_dist[i][2])
        result_dict[f"Layer {i} distances"] = {"L2": defended_dist[i][0], "Linf":defended_dist[i][1], "CD": defended_dist[i][2]}
    print("Defended Acc:", defended_acc)
    print("Defended Loss:", defended_loss)
    print()
    
    geo_mean = np.sqrt((denoised_acc**2 + defended_acc**2)/2)
    print("Geo-Mean Acc:", geo_mean)
    
    result_dict["Clean_loss"] = clean_loss
    result_dict["Attacked_loss"] = attacked_loss
    result_dict["Denoised_loss"] = denoised_loss
    result_dict["Defended_loss"] = defended_loss
    
    result_dict["Clean_acc"] = clean_acc
    result_dict["Attacked_acc"] = attacked_acc
    result_dict["Denoised_acc"] = denoised_acc
    result_dict["Defended_acc"] = defended_acc
    result_dict["Geo-Mean_acc"] = geo_mean

    
    savedir = os.path.join(*[args.save_path, args.model, args.data_attacked.split("/")[1], "T__" + str("_".join([str(i) for i in args.t_list]))]) if args.save_path is not None else None
    filename = os.path.join(savedir, datetime.today().strftime('RESULT_%Y_%m_%d__%H_%M_%S')) if savedir is not None else None
    
    os.makedirs(savedir, exist_ok=True)
    
    args_dict = vars(args)
    with open(args.json_path, "r") as file:
        args_dict["attack_args"] = json.load(file)
    
    args_dict |= {"data_path":filename}
    args_dict |= result_dict
    with open(filename+'.json', 'w') as fp:
        json.dump(args_dict, fp)
        fp.close()

io = IOStream('outputs/' + args.exp_name + '/run.log')

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

test(args, io)
